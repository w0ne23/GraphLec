"""
slide_extractor.py
==================
PPT 기반 강의 영상에서 scene + 필기 완료 시점 프레임을 추출합니다.

용어:
  - scene: 영상 타임라인에서 연속적으로 등장하는 하나의 방문 구간
           (기존 slide_index 의미)
  - slide: 원본 장표 identity
           (재방문 / 애니메이션 분리 scene들을 하나로 묶는 논리 단위)

Input : input/lecture.mp4
Output: output_slides/
        ├── slide_001_base.jpg      # 슬라이드 최초 등장 프레임
        ├── slide_001_annot_01.jpg  # 필기 안정화 캡처
        ├── slide_002_base.jpg      # 전환 or PPT 애니메이션(새 콘텐츠) → 새 base
        └── ...

슬라이드 idx가 증가하는 경우:
  1. Cut 전환 (직전 프레임 대비 급격한 MSE + phash 변화)
  2. Fade 전환 (sliding window 내 oldest ↔ current 누적 변화)
  3. Slide base 이중 비교 (필기로 오염된 prev_frame 대신 클린한 base phash 비교)
  4. PPT 애니메이션 (슬라이드 내 대규모 콘텐츠 변화 → 새 base로 분리)

Usage:
    python slide_extractor.py --input input/lecture.mp4 --output output_slides/
    python slide_extractor.py --input input/lecture.mp4 --output output_slides/ --debug
    python slide_extractor.py --input input/lecture.mp4 --tune
"""

import cv2
import numpy as np
import imagehash
import os
import platform
import ctypes
import shutil
import subprocess
import tempfile
from PIL import Image
from pathlib import Path
from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import argparse
import json
import logging
import math

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# 외부 라이브러리 노이즈 로그 억제
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("google.ai.generativelanguage").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.WARNING)

# ──────────────────────────────────────────────
# 설정값 (튜닝 포인트)
# ──────────────────────────────────────────────
class Config:
    # ── 슬라이드 전환 감지 (Cut) ─────────────────────────────────────
    SLIDE_CHANGE_MSE_THRESHOLD   = 500   # MSE 임계 (직전 프레임 대비)
    SLIDE_CHANGE_HASH_THRESHOLD  = 10    # phash hamming distance (0~64)

    # ── 슬라이드 전환 감지 (Fade - Sliding Window) ───────────────────
    FADE_WINDOW_SEC              = 1.0   # 페이드 감지 윈도우 크기 (초)
    FADE_MSE_THRESHOLD           = 400   # oldest ↔ current MSE 임계
    FADE_HASH_THRESHOLD          = 8     # oldest ↔ current phash 거리 임계

    # ── Slide base 이중 비교 ─────────────────────────────────────────
    # prev_frame이 필기로 오염되는 경우를 보완 (유사 레이아웃 슬라이드 구분)
    BASE_HASH_THRESHOLD          = 8     # slide_base_phash ↔ current phash 거리 임계

    # ── 중복 슬라이드 감지 (후처리) ──────────────────────────────────
    # compute_phash_hires (256비트, hash_size=16) 기준. 최대 거리 = 256.
    # 실행 후 로그의 슬라이드 간 거리 분포를 보고 튜닝:
    #   - 실제 동일 슬라이드 쌍의 dist → 이 값보다 크게
    #   - 실제 다른 슬라이드 쌍의 dist → 이 값보다 작게
    DUPLICATE_HASH_THRESHOLD     = 30    # 초기값, 로그 확인 후 조정 필요

    # ── 필기 감지 ────────────────────────────────────────────────────
    ANNOT_DIFF_THRESHOLD         = 15    # 픽셀 변화 판정 절댓값 임계
    ANNOT_CUMULATIVE_RATIO       = 0.0005  # base 대비 0.05% 이상 변화 → 필기 시작
    ANNOT_INSTANT_RATIO          = 0.0001  # 직전 프레임 대비 변화 → 펜 움직임 여부

    # ── PPT 애니메이션 감지 (새 텍스트/이미지 등장 → 새 base) ─────────
    # 안정화 완료 시 새 slide_idx + base 파일로 저장
    # 튜닝: --tune 모드에서 누적 diff p99 이상 값 참고
    ANIM_CUMULATIVE_RATIO        = 0.05   # base 대비 5% 이상 변화 → 애니메이션

    # ── 안정화 판단 ──────────────────────────────────────────────────
    STABILITY_WINDOW_SEC         = 0.7
    MIN_ANNOT_DURATION_SEC       = 0.2

    # ── 처리 성능 ────────────────────────────────────────────────────
    PROCESS_EVERY_N_FRAMES       = 2
    # 전역 판정/annotation 감지는 이 해상도로 수행한다.
    # 후처리 duplicate 판정용 phash_hires보다 더 작은 폭을 사용해도 충분한 경우가 많다.
    DECISION_RESIZE_WIDTH        = int(os.getenv("GRAPHLEC_SLIDE_DECISION_WIDTH", "768"))
    RESIZE_WIDTH                 = 960
    DECODE_BACKEND               = os.getenv("GRAPHLEC_SLIDE_DECODE_BACKEND", "auto")
    FFMPEG_HWACCEL               = os.getenv("GRAPHLEC_FFMPEG_HWACCEL", "cuda")
    # 서버/로컬 공통 정책: 슬라이드 추출은 항상 5분 단위 청크 병렬 처리
    EXTRACT_CHUNK_SEC            = 300.0
    EXTRACT_CHUNK_OVERLAP_SEC    = 3.0
    EXTRACT_WORKERS              = int(os.getenv("GRAPHLEC_SLIDE_EXTRACT_WORKERS", "0"))


# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────
def compute_mse(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    if frame_a.ndim == 2:
        a = frame_a.astype(np.float32)
    else:
        a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if frame_b.ndim == 2:
        b = frame_b.astype(np.float32)
    else:
        b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.mean((a - b) ** 2))


def compute_phash(frame: np.ndarray) -> imagehash.ImageHash:
    """실시간 슬라이드 전환 감지용 (64비트, 속도 우선)"""
    if frame.ndim == 2:
        pil_img = Image.fromarray(frame)
    else:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return imagehash.phash(pil_img)


def compute_phash_int(frame: np.ndarray) -> int:
    return int(str(compute_phash(frame)), 16)


def phash_distance_int(a: int, b: int) -> int:
    return int(a ^ b).bit_count()


def compute_phash_hires(frame: np.ndarray) -> imagehash.ImageHash:
    """중복 슬라이드 후처리 감지용 (256비트, 정밀도 우선).

    PPT 템플릿처럼 레이아웃이 동일한 슬라이드들은 64비트 phash로는
    콘텐츠 차이를 구분하기 어렵다. hash_size=16 (256비트)으로 세밀한
    콘텐츠 차이를 포착한다. 최대 거리는 256.
    임계값 튜닝 기준: 실제 동일 슬라이드의 거리를 로그로 확인 후 설정.
    """
    if frame.ndim == 2:
        pil_img = Image.fromarray(frame)
    else:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return imagehash.phash(pil_img, hash_size=16)


def resize_frame(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def count_changed_pixels(frame_a: np.ndarray, frame_b: np.ndarray, threshold: int) -> float:
    diff = cv2.absdiff(frame_a, frame_b)
    if diff.ndim == 2:
        max_diff = diff
    else:
        max_diff = np.max(diff, axis=2)
    return np.sum(max_diff > threshold) / max_diff.size


def to_decision_frame(frame: np.ndarray, width: int) -> np.ndarray:
    small = resize_frame(frame, width)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def _video_metadata(input_path: str) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"영상 파일을 열 수 없습니다: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if fps <= 0.0 or width <= 0 or height <= 0:
        raise RuntimeError(f"영상 메타데이터를 읽지 못했습니다: {input_path}")
    return fps, total_frames, width, height


def _ffmpeg_hwaccels() -> set[str]:
    if shutil.which("ffmpeg") is None:
        return set()
    try:
        result = subprocess.run(
            ["ffmpeg", "-hwaccels"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return set()

    accels: set[str] = set()
    for line in (result.stdout or "").splitlines():
        token = line.strip().lower()
        if token and token.isascii() and " " not in token and token != "hardware acceleration methods:":
            accels.add(token)
    return accels


def _cuda_runtime_available() -> bool:
    system = platform.system().lower()
    if system == "darwin":
        return False

    nvidia_markers = (
        "/dev/nvidiactl",
        "/dev/nvidia0",
        "/proc/driver/nvidia/version",
    )
    if any(Path(marker).exists() for marker in nvidia_markers):
        return True

    if shutil.which("nvidia-smi") is not None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0 and (result.stdout or "").strip():
                return True
        except Exception:
            pass

    try:
        ctypes.CDLL("libcuda.so.1")
        return True
    except OSError:
        return False


def _resolve_decode_backend(preferred_backend: str) -> tuple[str, str | None]:
    backend = (preferred_backend or "auto").strip().lower()
    hwaccels = _ffmpeg_hwaccels()
    system = platform.system().lower()
    cuda_usable = "cuda" in hwaccels and _cuda_runtime_available()

    if backend == "ffmpeg-cuda":
        if cuda_usable:
            return "ffmpeg", "cuda"
        return "opencv", None

    if backend == "ffmpeg-videotoolbox":
        if "videotoolbox" in hwaccels:
            return "ffmpeg", "videotoolbox"
        return "opencv", None

    if backend == "auto":
        if cuda_usable:
            return "ffmpeg", "cuda"
        if system == "darwin" and "videotoolbox" in hwaccels:
            return "ffmpeg", "videotoolbox"
        return "opencv", None

    return "opencv", None


def _iter_processed_frames_opencv(input_path: str, cfg: Config, fps: float):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"영상 파일을 열 수 없습니다: {input_path}")

    try:
        frame_no = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1
            if frame_no % cfg.PROCESS_EVERY_N_FRAMES != 0:
                continue
            yield frame_no, frame_no / fps, frame
    finally:
        cap.release()


def _iter_processed_frames_opencv_range(
    input_path: str,
    cfg: Config,
    fps: float,
    start_sec: float,
    end_sec: float,
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"영상 파일을 열 수 없습니다: {input_path}")

    start_frame = max(0, int(start_sec * fps))
    end_frame = max(start_frame, int(end_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame

    try:
        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % cfg.PROCESS_EVERY_N_FRAMES != 0:
                continue
            yield frame_idx, frame_idx / fps, frame
    finally:
        cap.release()


def _iter_processed_frames_ffmpeg_hwaccel(
    input_path: str,
    cfg: Config,
    fps: float,
    width: int,
    height: int,
    hwaccel: str,
):
    select_filter = (
        f"select='not(mod(n+1\\,{cfg.PROCESS_EVERY_N_FRAMES}))',format=bgr24"
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-hwaccel",
        hwaccel,
        "-i",
        input_path,
        "-vf",
        select_filter,
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10 ** 8,
    )

    frame_size = width * height * 3
    frame_no = cfg.PROCESS_EVERY_N_FRAMES

    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if not raw:
                break
            if len(raw) != frame_size:
                raise RuntimeError("ffmpeg rawvideo 출력이 중간에 잘렸습니다.")

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
            yield frame_no, frame_no / fps, frame
            frame_no += cfg.PROCESS_EVERY_N_FRAMES
    finally:
        if proc.stdout:
            proc.stdout.close()

    stderr = b""
    if proc.stderr is not None:
        stderr = proc.stderr.read()
        proc.stderr.close()
    ret = proc.wait()
    if ret != 0:
        err_text = stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg hwaccel 디코드 실패 (hwaccel={hwaccel}, exit={ret}): {err_text}")


def _iter_processed_frames_ffmpeg_hwaccel_range(
    input_path: str,
    cfg: Config,
    fps: float,
    width: int,
    height: int,
    hwaccel: str,
    start_sec: float,
    end_sec: float,
):
    select_filter = f"select='not(mod(n+1\\,{cfg.PROCESS_EVERY_N_FRAMES}))',format=bgr24"
    duration = max(0.0, end_sec - start_sec)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_sec:.3f}",
        "-hwaccel",
        hwaccel,
        "-i",
        input_path,
        "-t",
        f"{duration:.3f}",
        "-vf",
        select_filter,
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10 ** 8,
    )

    frame_size = width * height * 3
    start_frame = max(0, int(start_sec * fps))
    absolute_frame_no = start_frame + 1
    while absolute_frame_no % cfg.PROCESS_EVERY_N_FRAMES != 0:
        absolute_frame_no += 1

    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if not raw:
                break
            if len(raw) != frame_size:
                raise RuntimeError("ffmpeg rawvideo 출력이 중간에 잘렸습니다.")
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
            timestamp = absolute_frame_no / fps
            yield absolute_frame_no, timestamp, frame
            absolute_frame_no += cfg.PROCESS_EVERY_N_FRAMES
    finally:
        if proc.stdout:
            proc.stdout.close()

    stderr = b""
    if proc.stderr is not None:
        stderr = proc.stderr.read()
        proc.stderr.close()
    ret = proc.wait()
    if ret != 0:
        err_text = stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg hwaccel 디코드 실패 (hwaccel={hwaccel}, exit={ret}): {err_text}")


def _frame_iterator(
    input_path: str,
    cfg: Config,
    fps: float,
    width: int,
    height: int,
    decode_backend: str,
):
    resolved_backend, hwaccel = _resolve_decode_backend(decode_backend or cfg.DECODE_BACKEND)

    if resolved_backend == "ffmpeg" and hwaccel:
        try:
            log.info(f"프레임 디코드 백엔드: ffmpeg ({hwaccel})")
            return (
                _iter_processed_frames_ffmpeg_hwaccel(input_path, cfg, fps, width, height, hwaccel),
                f"ffmpeg-{hwaccel}",
            )
        except FileNotFoundError:
            log.warning("ffmpeg를 찾지 못해 OpenCV 디코드로 폴백합니다.")
        except Exception as e:
            log.warning(f"ffmpeg hwaccel 초기화 실패로 OpenCV 디코드로 폴백합니다: {e}")

    log.info("프레임 디코드 백엔드: opencv")
    return _iter_processed_frames_opencv(input_path, cfg, fps), "opencv"


def _frame_iterator_range(
    input_path: str,
    cfg: Config,
    fps: float,
    width: int,
    height: int,
    decode_backend: str,
    start_sec: float,
    end_sec: float,
):
    resolved_backend, hwaccel = _resolve_decode_backend(decode_backend or cfg.DECODE_BACKEND)

    if resolved_backend == "ffmpeg" and hwaccel:
        try:
            log.info(
                f"청크 프레임 디코드 백엔드: ffmpeg ({hwaccel}) [{start_sec:.2f}s ~ {end_sec:.2f}s]"
            )
            return (
                _iter_processed_frames_ffmpeg_hwaccel_range(
                    input_path, cfg, fps, width, height, hwaccel, start_sec, end_sec
                ),
                f"ffmpeg-{hwaccel}",
            )
        except FileNotFoundError:
            log.warning("ffmpeg를 찾지 못해 OpenCV 디코드로 폴백합니다.")
        except Exception as e:
            log.warning(f"ffmpeg hwaccel 초기화 실패로 OpenCV 디코드로 폴백합니다: {e}")

    log.info(f"청크 프레임 디코드 백엔드: opencv [{start_sec:.2f}s ~ {end_sec:.2f}s]")
    return _iter_processed_frames_opencv_range(input_path, cfg, fps, start_sec, end_sec), "opencv"


# ──────────────────────────────────────────────
# 슬라이드 전환 감지기 (Cut + Fade + Base 이중 비교)
# ──────────────────────────────────────────────
class SlideChangeDetector:
    """
    세 가지 방식으로 슬라이드 전환을 감지:

      1. Cut:  prev_frame ↔ current MSE + phash 급등
      2. Base 이중 비교:  slide_base_phash ↔ current 비교
         - prev_frame이 필기로 오염되어 phash가 왜곡되는 경우 보완
         - 유사 레이아웃 슬라이드도 구분 가능
      3. Fade: sliding window의 oldest ↔ current 누적 변화
         - 연속 프레임 간 diff가 작아 cut을 통과하는 페이드 감지

    슬라이드 전환(또는 PPT 애니메이션 new base) 시 반드시 reset() 호출.
    """
    def __init__(self, cfg: Config, fps: float):
        self.cfg = cfg
        buf_size = max(2, int(cfg.FADE_WINDOW_SEC * fps / cfg.PROCESS_EVERY_N_FRAMES))
        self.frame_buffer     = deque(maxlen=buf_size)
        self.prev_frame       = None
        self.prev_phash       = None
        self.slide_base_phash = None

    def reset(self, frame: np.ndarray):
        """슬라이드 전환 후 호출: 버퍼·base 초기화"""
        phash = compute_phash_int(frame)
        self.prev_frame       = frame.copy()
        self.prev_phash       = phash
        self.slide_base_phash = phash
        self.frame_buffer.clear()
        self.frame_buffer.append((frame.copy(), phash))

    def is_slide_change(self, frame: np.ndarray, curr_phash: int | None = None) -> bool:
        if self.prev_frame is None:
            self.reset(frame)
            return False

        if curr_phash is None:
            curr_phash = compute_phash_int(frame)

        # ── 1. Cut 전환 ────────────────────────────────────────────────
        mse = compute_mse(self.prev_frame, frame)
        if mse >= self.cfg.SLIDE_CHANGE_MSE_THRESHOLD:
            if phash_distance_int(self.prev_phash, curr_phash) >= self.cfg.SLIDE_CHANGE_HASH_THRESHOLD:
                self._update(frame, curr_phash)
                return True

        # ── 2. Base 이중 비교 ──────────────────────────────────────────
        if self.slide_base_phash is not None:
            if phash_distance_int(self.slide_base_phash, curr_phash) >= self.cfg.BASE_HASH_THRESHOLD:
                if mse >= self.cfg.SLIDE_CHANGE_MSE_THRESHOLD * 0.5:
                    self._update(frame, curr_phash)
                    return True

        # ── 3. Fade 전환 ───────────────────────────────────────────────
        if len(self.frame_buffer) == self.frame_buffer.maxlen:
            oldest_frame, oldest_phash = self.frame_buffer[0]
            if compute_mse(oldest_frame, frame) >= self.cfg.FADE_MSE_THRESHOLD:
                if phash_distance_int(oldest_phash, curr_phash) >= self.cfg.FADE_HASH_THRESHOLD:
                    self._update(frame, curr_phash)
                    return True

        self._update(frame, curr_phash)
        return False

    def _update(self, frame: np.ndarray, phash: int | None = None):
        if phash is None:
            phash = compute_phash_int(frame)
        self.prev_frame = frame.copy()
        self.prev_phash = phash
        self.frame_buffer.append((frame.copy(), phash))


# ──────────────────────────────────────────────
# 필기 안정화 감지기
# ──────────────────────────────────────────────
class AnnotationStabilityDetector:
    """
    슬라이드 내 변화를 두 단계 임계값으로 구분:

      소규모 변화 (ANNOT_RATIO ≤ ratio < ANIM_RATIO)
        → 필기 → CAPTURE_ANNOT → annot_XX 저장

      대규모 변화 (ratio ≥ ANIM_RATIO)
        → PPT 애니메이션 → NEW_BASE
          → main loop에서 슬라이드 전환과 동일하게 처리 (새 slide_idx + base 저장)

    상태 머신:
      STABLE    → (ratio ≥ ANIM_RATIO)  → ANIMATING
               → (ratio ≥ ANNOT_RATIO) → WRITING
      WRITING   → (안정화) → CAPTURE_ANNOT → STABLE (base 갱신)
      WRITING   → (ratio가 ANIM_RATIO 초과) → ANIMATING 격상
      ANIMATING → (안정화) → NEW_BASE       → (main loop이 reset 호출)
    """
    def __init__(self, cfg: Config, fps: float):
        self.cfg = cfg
        self.stability_frames = int(cfg.STABILITY_WINDOW_SEC * fps / cfg.PROCESS_EVERY_N_FRAMES)
        self.min_annot_frames = int(cfg.MIN_ANNOT_DURATION_SEC * fps / cfg.PROCESS_EVERY_N_FRAMES)
        self.reset()

    def reset(self, base_frame: np.ndarray = None):
        self.state            = "STABLE"
        self.base_frame       = base_frame
        self.prev_frame       = base_frame
        self.stable_count     = 0
        self.writing_count    = 0
        self.last_annot_frame_no = None
        self.last_annot_timestamp = None

    def process(
        self,
        frame: np.ndarray,
        frame_no: int | None = None,
        timestamp: float | None = None,
    ) -> str:
        """
        반환값:
          "CAPTURE_ANNOT"  - 필기 안정화 완료 → annot_XX 저장
          "NEW_BASE"       - 애니메이션 안정화 완료 → 새 slide_idx + base 저장
          "NONE"
        """
        if self.base_frame is None or self.prev_frame is None:
            self.prev_frame = frame.copy()
            return "NONE"

        cumulative_ratio = count_changed_pixels(
            self.base_frame, frame, self.cfg.ANNOT_DIFF_THRESHOLD
        )
        instant_ratio = count_changed_pixels(
            self.prev_frame, frame, self.cfg.ANNOT_DIFF_THRESHOLD
        )
        is_active = instant_ratio >= self.cfg.ANNOT_INSTANT_RATIO

        result = "NONE"

        if self.state == "STABLE":
            if cumulative_ratio >= self.cfg.ANIM_CUMULATIVE_RATIO:
                self.state         = "ANIMATING"
                self.writing_count = 1
                self.stable_count  = 0
                self.last_annot_frame_no = frame_no
                self.last_annot_timestamp = timestamp
            elif cumulative_ratio >= self.cfg.ANNOT_CUMULATIVE_RATIO:
                self.state         = "WRITING"
                self.writing_count = 1
                self.stable_count  = 0
                self.last_annot_frame_no = frame_no
                self.last_annot_timestamp = timestamp

        elif self.state in ("WRITING", "ANIMATING"):
            # WRITING 도중 변화량이 커지면 ANIMATING으로 격상
            if self.state == "WRITING" and cumulative_ratio >= self.cfg.ANIM_CUMULATIVE_RATIO:
                self.state = "ANIMATING"

            if cumulative_ratio >= self.cfg.ANNOT_CUMULATIVE_RATIO:
                self.writing_count += 1
                if is_active:
                    self.stable_count = 0
                    self.last_annot_frame_no = frame_no
                    self.last_annot_timestamp = timestamp
                else:
                    self.stable_count += 1

                if self.stable_count >= self.stability_frames:
                    if self.state == "ANIMATING":
                        result = "NEW_BASE"
                        # main loop에서 reset()을 호출하므로 여기선 상태만 초기화
                    else:
                        result = "CAPTURE_ANNOT"
                        # 다음 필기 이벤트를 위해 base를 현재 안정 상태로 갱신
                        self.base_frame = frame.copy()
                    self.state         = "STABLE"
                    self.stable_count  = 0
                    self.writing_count = 0

        self.prev_frame = frame.copy()
        return result

    def get_capture_frame_no(self, current_frame_no: int) -> int:
        return self.last_annot_frame_no if self.last_annot_frame_no is not None else current_frame_no

    def get_capture_timestamp(self, current_timestamp: float) -> float:
        return self.last_annot_timestamp if self.last_annot_timestamp is not None else current_timestamp


# ──────────────────────────────────────────────
# 메인 파이프라인
# ──────────────────────────────────────────────
def _run_slide_decision_pass(
    frame_iter,
    cfg: Config,
    fps: float,
    duration: float,
    debug: bool = False,
):
    slide_detector = SlideChangeDetector(cfg, fps)
    annot_detector = AnnotationStabilityDetector(cfg, fps)

    slide_idx   = 0
    annot_idx   = 0
    processed_frames = 0
    first_frame = True
    metadata    = []
    progress_interval = max(1, int((duration * fps / cfg.PROCESS_EVERY_N_FRAMES) / 20)) if duration > 0 else 500

    def register_new_base(frame_no, small, timestamp, reason):
        """scene 전환 or PPT 애니메이션 → 새 scene(base) 메타 저장 공통 처리"""
        nonlocal slide_idx, annot_idx
        slide_idx += 1
        annot_idx  = 0
        fname = f"slide_{slide_idx:03d}_base.jpg"
        metadata.append(_meta(fname, slide_idx, timestamp, "base", annot_index=0, frame_no=frame_no))
        log.info(f"[씬 {slide_idx}] base ({reason}) @ {timestamp:.2f}s")
        slide_detector.reset(small)
        annot_detector.reset(base_frame=small)

    for item in frame_iter:
        if len(item) >= 4:
            frame_no, timestamp, small, curr_phash = item[:4]
        else:
            frame_no, timestamp, small = item
            curr_phash = None
        processed_frames += 1

        # ── scene 전환 감지 (Cut / Fade / Base 이중 비교) ──────────
        if first_frame or slide_detector.is_slide_change(small, curr_phash=curr_phash):
            if not first_frame:
                # 전환 직전 진행 중이던 필기 강제 캡처
                if annot_detector.state == "WRITING" \
                        and annot_detector.writing_count >= annot_detector.min_annot_frames:
                    annot_idx += 1
                    fname = f"slide_{slide_idx:03d}_annot_{annot_idx:02d}.jpg"
                    capture_frame_no = annot_detector.get_capture_frame_no(frame_no)
                    capture_ts = annot_detector.get_capture_timestamp(timestamp)
                    metadata.append(
                        _meta(
                            fname,
                            slide_idx,
                            capture_ts,
                            "annotation",
                            annot_index=annot_idx,
                            frame_no=capture_frame_no,
                        )
                    )
                    log.info(f"  [강제 캡처] {fname} @ {capture_ts:.2f}s")

            first_frame = False
            register_new_base(frame_no, small, timestamp, "slide_change")
            continue

        # ── 필기 / 애니메이션 감지 ──────────────────────────────────
        event = annot_detector.process(small, frame_no=frame_no, timestamp=timestamp)

        if event == "CAPTURE_ANNOT":
            annot_idx += 1
            capture_frame_no = annot_detector.get_capture_frame_no(frame_no)
            capture_ts = annot_detector.get_capture_timestamp(timestamp)
            fname = f"slide_{slide_idx:03d}_annot_{annot_idx:02d}.jpg"
            metadata.append(
                _meta(
                    fname,
                    slide_idx,
                    capture_ts,
                    "annotation",
                    annot_index=annot_idx,
                    frame_no=capture_frame_no,
                )
            )
            log.info(f"  [필기 완료] {fname} @ {capture_ts:.2f}s")

        elif event == "NEW_BASE":
            # PPT 애니메이션으로 새 콘텐츠 등장 → 새 scene/base로 처리
            capture_frame_no = annot_detector.get_capture_frame_no(frame_no)
            register_new_base(capture_frame_no, small, timestamp, "animation_base")

        if debug and frame_no % (int(fps) * 10) == 0:
            log.debug(f"  처리 중: {timestamp:.1f}s / {duration:.1f}s")
        elif processed_frames % progress_interval == 0:
            ratio = min(100.0, (timestamp / duration) * 100.0) if duration > 0 else 0.0
            log.info(
                f"  [전역 판정 진행] sampled_frames={processed_frames}, "
                f"time={timestamp:.1f}s/{duration:.1f}s ({ratio:.1f}%)"
            )

    log.info(f"  처리 프레임={processed_frames}")
    return metadata


def _extract_slides_core(
    input_path: str,
    output_dir: str,
    debug: bool = False,
    decode_backend: str | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
    run_postprocess: bool = True,
):
    fps, total_frames, frame_width, frame_height = _video_metadata(input_path)
    duration = total_frames / fps if total_frames > 0 else 0.0

    log.info(f"영상 로드: {input_path}")
    log.info(f"  FPS={fps:.2f}, 총 프레임={total_frames}, 길이={duration:.1f}초")
    log.info(f"  해상도={frame_width}x{frame_height}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    if start_sec is None or end_sec is None:
        frame_iter, active_backend = _frame_iterator(
            input_path,
            cfg,
            fps,
            frame_width,
            frame_height,
            decode_backend or cfg.DECODE_BACKEND,
        )
    else:
        frame_iter, active_backend = _frame_iterator_range(
            input_path,
            cfg,
            fps,
            frame_width,
            frame_height,
            decode_backend or cfg.DECODE_BACKEND,
            start_sec,
            end_sec,
        )

    decision_width = max(160, min(cfg.DECISION_RESIZE_WIDTH, cfg.RESIZE_WIDTH))
    decision_iter = (
        (frame_no, timestamp, to_decision_frame(frame, decision_width))
        for frame_no, timestamp, frame in frame_iter
    )
    metadata = _run_slide_decision_pass(
        frame_iter=decision_iter,
        cfg=cfg,
        fps=fps,
        duration=duration,
        debug=debug,
    )

    _materialize_metadata_frames(input_path, out_path, metadata)
    metadata = add_slide_time_ranges(metadata, duration)
    metadata = mark_visual_duplicates(metadata, out_path, cfg)
    metadata = finalize_scene_slide_metadata(metadata)
    log_scene_slide_summary(metadata)

    if not run_postprocess:
        return metadata

    meta_path = out_path / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    scene_slide_map_path = out_path / "scene_slide_map.json"
    with open(scene_slide_map_path, "w", encoding="utf-8") as f:
        json.dump(build_scene_slide_map(metadata), f, ensure_ascii=False, indent=2)

    canonical_slide_annotations_path = out_path / "canonical_slide_annotations.json"
    with open(canonical_slide_annotations_path, "w", encoding="utf-8") as f:
        json.dump(build_canonical_slide_annotations(metadata), f, ensure_ascii=False, indent=2)
    log.info(f"  디코드 백엔드={active_backend}")
    log.info(f"\n완료: scene {len({m['scene_index'] for m in metadata})}개, 총 {len(metadata)}개 프레임 저장 → {output_dir}")
    return metadata


def _extract_sampled_frames_chunk_worker(
    input_path: str,
    chunk_dir: str,
    decode_backend: str,
    start_sec: float,
    end_sec: float,
):
    cfg = Config()
    fps, total_frames, frame_width, frame_height = _video_metadata(input_path)
    duration = total_frames / fps if total_frames > 0 else 0.0
    chunk_path = Path(chunk_dir)
    chunk_path.mkdir(parents=True, exist_ok=True)

    frame_iter, active_backend = _frame_iterator_range(
        input_path,
        cfg,
        fps,
        frame_width,
        frame_height,
        decode_backend or cfg.DECODE_BACKEND,
        start_sec,
        end_sec,
    )

    sampled_fps = max(1.0, fps / max(1, cfg.PROCESS_EVERY_N_FRAMES))
    decision_width = max(160, min(cfg.DECISION_RESIZE_WIDTH, cfg.RESIZE_WIDTH))
    small_height = int(frame_height * (decision_width / frame_width))
    video_filename = "sampled_frames.avi"
    video_path = chunk_path / video_filename
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        sampled_fps,
        (decision_width, small_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"샘플 프레임 비디오를 열 수 없습니다: {video_path}")

    manifest: list[dict] = []
    processed_frames = 0
    try:
        for frame_no, timestamp, frame in frame_iter:
            processed_frames += 1
            small = resize_frame(frame, decision_width)
            decision_small = cv2.GaussianBlur(
                cv2.cvtColor(small, cv2.COLOR_BGR2GRAY),
                (3, 3),
                0,
            )
            writer.write(small)
            manifest.append({
                "frame_no": frame_no,
                "timestamp_sec": round(timestamp, 4),
                "phash_int": compute_phash_int(decision_small),
            })
    finally:
        writer.release()

    payload = {
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_sec": duration,
        "decode_backend": active_backend,
        "processed_frames": processed_frames,
        "decision_resize_width": decision_width,
        "video_filename": video_filename,
        "frames": manifest,
    }
    manifest_path = chunk_path / "sampled_frames.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(manifest_path)


def _ordered_sampled_frames(manifest_paths: list[Path]):
    seen_frame_nos: set[int] = set()
    for manifest_path in manifest_paths:
        chunk_dir = manifest_path.parent
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        video_filename = payload.get("video_filename")
        if not video_filename:
            raise ValueError(f"video_filename이 없는 sampled manifest입니다: {manifest_path}")

        video_path = chunk_dir / video_filename
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"샘플 프레임 비디오를 열 수 없습니다: {video_path}")

        try:
            for item in payload.get("frames", []):
                ret, frame = cap.read()
                if not ret or frame is None:
                    raise RuntimeError(f"샘플 프레임 비디오를 끝까지 읽지 못했습니다: {video_path}")
                frame_no = int(item["frame_no"])
                if frame_no in seen_frame_nos:
                    continue
                seen_frame_nos.add(frame_no)
                decision_frame = cv2.GaussianBlur(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    (3, 3),
                    0,
                )
                phash_int = item.get("phash_int")
                if phash_int is None:
                    phash_int = compute_phash_int(decision_frame)
                yield (
                    frame_no,
                    float(item["timestamp_sec"]),
                    decision_frame,
                    int(phash_int),
                )
        finally:
            cap.release()


def _group_chunk_metadata(metadata: list[dict], source_dir: Path) -> list[list[dict]]:
    groups: list[list[dict]] = []
    current: list[dict] = []
    current_slide_idx = None
    for item in metadata:
        normalized = dict(item)
        normalized["_source_dir"] = str(source_dir)
        slide_idx = item["slide_index"]
        if current_slide_idx is None or slide_idx != current_slide_idx:
            if current:
                groups.append(current)
            current = [normalized]
            current_slide_idx = slide_idx
        else:
            current.append(normalized)
    if current:
        groups.append(current)
    return groups


def _item_source_path(item: dict) -> Path:
    return Path(item["_source_dir"]) / item["filename"]


def _group_representative_path(group: list[dict]) -> Path:
    annotations = [item for item in group if item.get("capture_type") == "annotation"]
    target = annotations[-1] if annotations else group[0]
    return _item_source_path(target)


def _group_base_path(group: list[dict]) -> Path:
    return _item_source_path(group[0])


def _image_hash_for_merge(path: Path, resize_width: int) -> imagehash.ImageHash:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    return compute_phash_hires(resize_frame(img, resize_width))


def _groups_match_for_merge(
    prev_group: list[dict],
    curr_group: list[dict],
    cfg: Config,
) -> bool:
    prev_end = max(item["timestamp_sec"] for item in prev_group)
    curr_start = min(item["timestamp_sec"] for item in curr_group)
    if curr_start - prev_end > max(1.0, cfg.EXTRACT_CHUNK_OVERLAP_SEC + 0.5):
        return False

    prev_rep = _image_hash_for_merge(_group_representative_path(prev_group), cfg.RESIZE_WIDTH)
    curr_rep = _image_hash_for_merge(_group_representative_path(curr_group), cfg.RESIZE_WIDTH)
    if (prev_rep - curr_rep) < cfg.DUPLICATE_HASH_THRESHOLD:
        return True

    prev_base = _image_hash_for_merge(_group_base_path(prev_group), cfg.RESIZE_WIDTH)
    curr_base = _image_hash_for_merge(_group_base_path(curr_group), cfg.RESIZE_WIDTH)
    return (prev_base - curr_base) < cfg.DUPLICATE_HASH_THRESHOLD


def _merge_group_frames(prev_group: list[dict], curr_group: list[dict]) -> list[dict]:
    seen = {
        (item.get("capture_type"), round(float(item.get("timestamp_sec", 0.0)), 2))
        for item in prev_group
    }
    merged = list(prev_group)
    for item in curr_group:
        key = (item.get("capture_type"), round(float(item.get("timestamp_sec", 0.0)), 2))
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    merged.sort(key=lambda x: (x["timestamp_sec"], 0 if x["capture_type"] == "base" else 1))
    return merged


def _copy_merged_groups(merged_groups: list[list[dict]], out_path: Path) -> list[dict]:
    for stale in out_path.glob("slide_*.jpg"):
        stale.unlink(missing_ok=True)

    metadata: list[dict] = []
    for new_slide_idx, group in enumerate(merged_groups, start=1):
        annot_idx = 0
        for item in sorted(group, key=lambda x: (x["timestamp_sec"], 0 if x["capture_type"] == "base" else 1)):
            capture_type = item["capture_type"]
            if capture_type == "base":
                fname = f"slide_{new_slide_idx:03d}_base.jpg"
            else:
                annot_idx += 1
                fname = f"slide_{new_slide_idx:03d}_annot_{annot_idx:02d}.jpg"
            shutil.copy2(_item_source_path(item), out_path / fname)
            metadata.append(_meta(fname, new_slide_idx, item["timestamp_sec"], capture_type))
    return metadata


def _chunk_specs(duration: float, cfg: Config, workers: int) -> list[dict]:
    if duration <= cfg.EXTRACT_CHUNK_SEC:
        return []

    chunk_sec = max(30.0, cfg.EXTRACT_CHUNK_SEC)
    overlap = max(0.5, min(cfg.EXTRACT_CHUNK_OVERLAP_SEC, chunk_sec / 4))
    specs: list[dict] = []
    chunk_count = max(1, math.ceil(duration / chunk_sec))
    for idx in range(chunk_count):
        core_start = idx * chunk_sec
        core_end = min(duration, (idx + 1) * chunk_sec)
        start_sec = 0.0 if idx == 0 else max(0.0, core_start - overlap)
        end_sec = duration if idx == chunk_count - 1 else min(duration, core_end + overlap)
        specs.append({
            "chunk_index": idx,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "core_start_sec": core_start,
            "core_end_sec": core_end,
        })
    return specs


def _extract_chunk_worker(
    input_path: str,
    chunk_dir: str,
    decode_backend: str,
    start_sec: float,
    end_sec: float,
    debug: bool = False,
):
    metadata = _extract_slides_core(
        input_path=input_path,
        output_dir=chunk_dir,
        debug=debug,
        decode_backend=decode_backend,
        start_sec=start_sec,
        end_sec=end_sec,
        run_postprocess=False,
    )
    chunk_meta_path = Path(chunk_dir) / "chunk_metadata.json"
    with open(chunk_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return str(chunk_meta_path)


def extract_slides(
    input_path: str,
    output_dir: str,
    debug: bool = False,
    decode_backend: str | None = None,
    extract_workers: int | None = None,
):
    cfg = Config()
    requested_workers = cfg.EXTRACT_WORKERS if extract_workers is None else int(extract_workers)
    fps, total_frames, _, _ = _video_metadata(input_path)
    duration = total_frames / fps if total_frames > 0 else 0.0

    specs = _chunk_specs(duration, cfg, requested_workers)
    if not specs:
        log.info(
            f"슬라이드 추출 단일 청크 실행: duration={duration:.1f}s <= "
            f"chunk_sec={cfg.EXTRACT_CHUNK_SEC:.1f}s"
        )
        return _extract_slides_core(input_path, output_dir, debug=debug, decode_backend=decode_backend)

    if requested_workers <= 0:
        workers = len(specs)
    else:
        # 5분 초과로 청크가 2개 이상 생긴 경우에는 항상 병렬로 처리한다.
        # 서버 환경에서 잘못된 설정값(예: 1)으로 병렬성이 꺼지는 일을 막는다.
        workers = min(max(2, requested_workers), len(specs))

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    log.info(
        f"슬라이드 추출 청크 병렬 실행: duration={duration:.1f}s, "
        f"chunk_sec={cfg.EXTRACT_CHUNK_SEC:.1f}s, requested_workers={requested_workers}, "
        f"workers={workers}, chunks={len(specs)}"
        + (" (auto)" if requested_workers <= 0 else "")
    )

    with tempfile.TemporaryDirectory(prefix="graphlec_slide_chunks_") as temp_root:
        temp_root_path = Path(temp_root)
        manifest_paths: dict[int, Path] = {}

        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {}
            for spec in specs:
                chunk_dir = temp_root_path / f"chunk_{spec['chunk_index']:03d}"
                chunk_dir.mkdir(parents=True, exist_ok=True)
                log.info(
                    f"  [청크 시작 {spec['chunk_index'] + 1}/{len(specs)}] "
                    f"{spec['start_sec']:.1f}s ~ {spec['end_sec']:.1f}s"
                )
                future = executor.submit(
                    _extract_sampled_frames_chunk_worker,
                    input_path,
                    str(chunk_dir),
                    decode_backend or cfg.DECODE_BACKEND,
                    spec["start_sec"],
                    spec["end_sec"],
                )
                future_map[future] = (spec, chunk_dir)

            pending = set(future_map.keys())
            completed = 0
            while pending:
                done, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                if not done:
                    log.info(
                        f"  [청크 대기 중] completed={completed}/{len(specs)}, "
                        f"running={len(pending)}"
                    )
                    continue

                for future in done:
                    spec, _ = future_map[future]
                    manifest_path = Path(future.result())
                    manifest_paths[spec["chunk_index"]] = manifest_path
                    completed += 1
                    try:
                        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                        log.info(
                            f"  [청크 완료 {completed}/{len(specs)} | idx={spec['chunk_index'] + 1}] "
                            f"backend={payload.get('decode_backend')} "
                            f"frames={payload.get('processed_frames')} "
                            f"range={spec['start_sec']:.1f}s~{spec['end_sec']:.1f}s"
                        )
                    except Exception:
                        log.info(
                            f"  [청크 완료 {completed}/{len(specs)} | idx={spec['chunk_index'] + 1}] "
                            f"range={spec['start_sec']:.1f}s~{spec['end_sec']:.1f}s"
                        )

        ordered_manifests = [manifest_paths[idx] for idx in sorted(manifest_paths)]
        total_sampled_frames = 0
        for manifest_path in ordered_manifests:
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                total_sampled_frames += int(payload.get("processed_frames", 0) or 0)
            except Exception:
                pass
        log.info(f"전역 판정 시작: ordered_chunks={len(ordered_manifests)}, sampled_frames={total_sampled_frames}")
        frame_iter = _ordered_sampled_frames(ordered_manifests)
        metadata = _run_slide_decision_pass(
            frame_iter=frame_iter,
            cfg=cfg,
            fps=fps,
            duration=duration,
            debug=debug,
        )

        _materialize_metadata_frames(input_path, out_path, metadata)
        metadata = add_slide_time_ranges(metadata, duration)
        metadata = mark_visual_duplicates(metadata, out_path, cfg)
        metadata = finalize_scene_slide_metadata(metadata)
        log_scene_slide_summary(metadata)

        meta_path = out_path / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        scene_slide_map_path = out_path / "scene_slide_map.json"
        with open(scene_slide_map_path, "w", encoding="utf-8") as f:
            json.dump(build_scene_slide_map(metadata), f, ensure_ascii=False, indent=2)

        canonical_slide_annotations_path = out_path / "canonical_slide_annotations.json"
        with open(canonical_slide_annotations_path, "w", encoding="utf-8") as f:
            json.dump(build_canonical_slide_annotations(metadata), f, ensure_ascii=False, indent=2)

        log.info(f"\n완료: scene {len({m['scene_index'] for m in metadata})}개, 총 {len(metadata)}개 프레임 저장 → {output_dir}")
        return metadata


def _save(frame: np.ndarray, path: Path):
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])


def _materialize_metadata_frames(input_path: str, out_path: Path, metadata: list[dict]):
    for stale in out_path.glob("slide_*.jpg"):
        stale.unlink(missing_ok=True)

    frame_targets: dict[int, list[str]] = {}
    for item in metadata:
        frame_no = int(item.get("frame_no", 0) or 0)
        if frame_no <= 0:
            raise ValueError(f"frame_no가 없는 metadata 항목입니다: {item}")
        frame_targets.setdefault(frame_no, []).append(item["filename"])

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"영상 파일을 열 수 없습니다: {input_path}")

    try:
        for frame_no in sorted(frame_targets):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_no - 1))
            ret, frame = cap.read()
            if not ret or frame is None:
                raise RuntimeError(f"frame_no={frame_no} 원본 프레임을 추출하지 못했습니다.")
            for fname in frame_targets[frame_no]:
                _save(frame, out_path / fname)
    finally:
        cap.release()


def _meta(
    fname: str,
    slide_idx: int,
    timestamp: float,
    capture_type: str,
    annot_index: int = 0,
    frame_no: int | None = None,
) -> dict:
    return {
        "filename":      fname,
        "slide_index":   slide_idx,
        "scene_index":   slide_idx,
        "timestamp_sec": round(timestamp, 2),
        "annot_index":   int(annot_index),
        "frame_no":      int(frame_no) if frame_no is not None else None,
        "capture_type":  capture_type,  # base | annotation
    }


# ──────────────────────────────────────────────
# 후처리: 슬라이드 단위 시간 구간 추가
# ──────────────────────────────────────────────
def add_slide_time_ranges(metadata: list, video_duration: float) -> list:
    """
    각 프레임 레코드에 scene/slide 시간 필드 추가.

    - scene_start_sec : 해당 scene_index의 base 프레임 타임스탬프
    - scene_end_sec   : 다음 scene_index의 시작 시각 (마지막 scene은 영상 길이)
    - slide_start_sec / slide_end_sec 는 기존 호환 필드로 유지

    slide_classifier에서 오디오 침묵 구간과 교차할 때 이 구간을 기준으로 사용한다.
    """
    # slide_index → base 타임스탬프 수집
    slide_starts: dict[int, float] = {}
    for m in metadata:
        idx = m["slide_index"]
        if m["capture_type"] == "base" and idx not in slide_starts:
            slide_starts[idx] = m["timestamp_sec"]

    sorted_indices = sorted(slide_starts.keys())

    # 각 슬라이드의 종료 시각 = 다음 슬라이드 시작 시각
    slide_ends: dict[int, float] = {}
    for i, idx in enumerate(sorted_indices):
        if i + 1 < len(sorted_indices):
            slide_ends[idx] = slide_starts[sorted_indices[i + 1]]
        else:
            slide_ends[idx] = round(video_duration, 2)

    for m in metadata:
        idx = m["slide_index"]
        m["scene_start_sec"] = slide_starts.get(idx)
        m["scene_end_sec"]   = slide_ends.get(idx)
        m["slide_start_sec"] = slide_starts.get(idx)
        m["slide_end_sec"]   = slide_ends.get(idx)

    return metadata


# ──────────────────────────────────────────────
# 후처리: 같은 slide(재등장/애니메이션 계열) 그룹 표시
# ──────────────────────────────────────────────
def mark_visual_duplicates(metadata: list, out_path: Path, cfg: Config) -> list:
    """
    모든 슬라이드의 대표 프레임(base + last_annot)을 풀에 쌓고
    전체 쌍(all-pairs)을 비교하여 같은 슬라이드 그룹을 표시한다.

    프레임 풀 구성:
      - 각 slide_index 별로 base 프레임 + last_annot 프레임(있으면) 수집
      - 레이블: "base{idx}" / "annot{idx}"

    비교:
      - 풀 내 모든 쌍을 phash(256비트) 비교
      - 동일 slide_index 간 쌍은 건너뜀
      - dist < DUPLICATE_HASH_THRESHOLD(현재 30) → 같은 슬라이드로 간주

    여기서 "같은 slide"는 애니메이션 단계/재등장(revisit)을 포함한
    같은 원본 장표 계열을 의미한다.
    """
    from collections import defaultdict

    # slide_index별 프레임 그룹화
    groups: dict[int, list] = defaultdict(list)
    for m in metadata:
        groups[m["slide_index"]].append(m)

    # 프레임 풀 구성: label → (slide_index, filename)
    pool: dict[str, tuple[int, str]] = {}
    for idx in sorted(groups.keys()):
        frames     = groups[idx]
        base_list  = [f for f in frames if f["capture_type"] == "base"]
        annot_list = [f for f in frames if f["capture_type"] == "annotation"]

        if base_list:
            pool[f"base{idx}"] = (idx, base_list[0]["filename"])
        if annot_list:
            pool[f"annot{idx}"] = (idx, annot_list[-1]["filename"])

    # phash 계산 (256비트)
    phashes: dict[str, imagehash.ImageHash] = {}
    for label, (_, fname) in pool.items():
        img = cv2.imread(str(out_path / fname))
        if img is not None:
            phashes[label] = compute_phash_hires(resize_frame(img, cfg.RESIZE_WIDTH))
        else:
            log.warning(f"  [중복 감지] 이미지 로드 실패: {fname}")

    # 전체 쌍 비교 — slide_index별 같은 슬라이드 관계 수집
    labels = sorted(phashes.keys())
    # duplicate_map[idx] = 이 슬라이드와 같은 슬라이드로 판정된 다른 slide_index 집합
    duplicate_map: dict[int, set[int]] = defaultdict(set)

    log.info("\n──────── 슬라이드 간 phash 거리 전체 비교 (같은 슬라이드 판정용) ────────")
    log.info(f"  DUPLICATE_HASH_THRESHOLD = {cfg.DUPLICATE_HASH_THRESHOLD}  (256비트 기준, 최대 256)")
    log.info(f"  {'프레임 쌍':<30} {'dist':>5}  {'판정'}")
    log.info(f"  {'-'*30}  {'-'*5}  {'-'*10}")

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            la, lb  = labels[i], labels[j]
            idx_a   = pool[la][0]
            idx_b   = pool[lb][0]

            # 같은 슬라이드 내 base↔annot 쌍은 건너뜀
            if idx_a == idx_b:
                continue

            dist = phashes[la] - phashes[lb]
            flag = "★ 같은 슬라이드" if dist < cfg.DUPLICATE_HASH_THRESHOLD else ""

            log.info(f"  {la:<14} ↔ {lb:<14}  {dist:>5}  {flag}")

            if dist < cfg.DUPLICATE_HASH_THRESHOLD:
                duplicate_map[idx_a].add(idx_b)
                duplicate_map[idx_b].add(idx_a)

    log.info("──────────────────────────────────────────────────────────────\n")

    # ── union-find로 전이적 같은 슬라이드 그룹 확정 ────────────────────── #
    all_indices = list(groups.keys())
    parent = {idx: idx for idx in all_indices}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for idx_a, neighbors in duplicate_map.items():
        for idx_b in neighbors:
            union(idx_a, idx_b)

    # 루트별로 그룹 구성
    from collections import defaultdict as _defaultdict
    dup_groups: dict[int, set[int]] = _defaultdict(set)
    for idx in all_indices:
        dup_groups[find(idx)].add(idx)

    # 각 scene의 slide family 산출
    group_of: dict[int, set[int]] = {}
    for members in dup_groups.values():
        for idx in members:
            group_of[idx] = members

    # 같은 slide family 내 scene 방문 순서도 함께 기록한다.
    family_visit_order: dict[int, int] = {}
    family_prev_visit: dict[int, int | None] = {}
    family_next_visit: dict[int, int | None] = {}
    for members in dup_groups.values():
        ordered = sorted(members)
        for pos, idx in enumerate(ordered, start=1):
            family_visit_order[idx] = pos
            family_prev_visit[idx] = ordered[pos - 2] if pos > 1 else None
            family_next_visit[idx] = ordered[pos] if pos < len(ordered) else None

    # 호환성을 위해 duplicate_of는 유지하되,
    # 의미는 "같은 슬라이드 계열의 다른 slide_index"로 본다.
    for m in metadata:
        idx = m["slide_index"]
        members = sorted(group_of.get(idx, {idx}))
        others = [x for x in members if x != idx]
        m["duplicate_of"] = others
        m["scene_group"] = members
        m["scene_canonical"] = members[0]
        m["scene_group_size"] = len(members)
        m["same_slide_group"] = members
        m["same_slide_canonical"] = members[0]
        m["same_slide_group_size"] = len(members)
        m["same_slide_visit_order"] = family_visit_order.get(idx, 1)
        m["same_slide_is_revisit"] = family_visit_order.get(idx, 1) > 1
        m["same_slide_previous"] = family_prev_visit.get(idx)
        m["same_slide_next"] = family_next_visit.get(idx)
        m["slide_group"] = members
        m["slide_canonical_index"] = members[0]
        m["slide_group_size"] = len(members)
        m["slide_visit_order"] = family_visit_order.get(idx, 1)
        m["slide_is_revisit"] = family_visit_order.get(idx, 1) > 1
        m["previous_scene_index"] = family_prev_visit.get(idx)
        m["next_scene_index"] = family_next_visit.get(idx)

    return metadata


def finalize_scene_slide_metadata(metadata: list[dict]) -> list[dict]:
    """
    scene 기준 로컬 annot 번호와 slide 기준 누적 annot 번호를 함께 기록한다.

    - annot_index: scene 내부 로컬 번호 (기존 유지)
    - slide_annot_index: 같은 canonical slide 전체에서의 누적 번호
    - scene_annotation_count: 해당 scene의 annot 개수
    - slide_annotation_count_total: 해당 canonical slide 전체 annot 개수
    - scene_annotation_start_index / end_index:
        해당 scene이 canonical slide 누적 annot에서 차지하는 범위
    """
    from collections import defaultdict

    by_scene: dict[int, list[dict]] = defaultdict(list)
    for item in metadata:
        by_scene[int(item.get("scene_index", item.get("slide_index", 0)) or 0)].append(item)

    by_slide: dict[int, list[tuple[int, list[dict]]]] = defaultdict(list)
    for scene_idx, items in by_scene.items():
        slide_idx = int(
            items[0].get("slide_canonical_index")
            or items[0].get("same_slide_canonical")
            or scene_idx
        )
        by_slide[slide_idx].append((scene_idx, items))

    scene_ranges: dict[int, tuple[int, int]] = {}
    slide_totals: dict[int, int] = {}

    for slide_idx, scene_groups in by_slide.items():
        scene_groups.sort(key=lambda pair: min(float(x.get("timestamp_sec", 0.0) or 0.0) for x in pair[1]))
        cumulative = 0
        for scene_idx, items in scene_groups:
            annots = sorted(
                [x for x in items if x.get("capture_type") == "annotation"],
                key=lambda x: (
                    int(x.get("annot_index", 0) or 0),
                    float(x.get("timestamp_sec", 0.0) or 0.0),
                ),
            )
            start = cumulative + 1 if annots else 0
            for offset, annot in enumerate(annots, start=1):
                annot["scene_annot_index"] = int(annot.get("annot_index", 0) or 0)
                annot["slide_annot_index"] = cumulative + offset
            cumulative += len(annots)
            end = cumulative if annots else 0
            scene_ranges[scene_idx] = (start, end)
        slide_totals[slide_idx] = cumulative

    slide_number_lookup = _build_slide_number_lookup(metadata)

    for scene_idx, items in by_scene.items():
        slide_idx = int(
            items[0].get("slide_canonical_index")
            or items[0].get("same_slide_canonical")
            or scene_idx
        )
        scene_annots = [x for x in items if x.get("capture_type") == "annotation"]
        start, end = scene_ranges.get(scene_idx, (0, 0))
        for item in items:
            item["slide_number"] = slide_number_lookup.get(slide_idx, slide_idx)
            item["scene_annotation_count"] = len(scene_annots)
            item["slide_annotation_count_total"] = slide_totals.get(slide_idx, 0)
            item["scene_annotation_start_index"] = start
            item["scene_annotation_end_index"] = end
            item["scene_local_annot_index"] = int(item.get("annot_index", 0) or 0)
            if item.get("capture_type") != "annotation":
                item["slide_annot_index"] = 0
                item["scene_annot_index"] = 0

    return metadata


def _build_slide_number_lookup(metadata: list[dict]) -> dict[int, int]:
    from collections import defaultdict

    by_scene: dict[int, list[dict]] = defaultdict(list)
    for item in metadata:
        by_scene[int(item.get("scene_index", item.get("slide_index", 0)) or 0)].append(item)

    ordered_pairs: list[tuple[float, int]] = []
    for scene_idx in sorted(by_scene):
        items = by_scene[scene_idx]
        base = next((x for x in items if x.get("capture_type") == "base"), items[0])
        slide_idx = int(base.get("slide_canonical_index") or base.get("same_slide_canonical") or scene_idx)
        ts = float(base.get("scene_start_sec", base.get("slide_start_sec", base.get("timestamp_sec", 0.0))) or 0.0)
        ordered_pairs.append((ts, slide_idx))

    lookup: dict[int, int] = {}
    for _, slide_idx in sorted(ordered_pairs, key=lambda x: x[0]):
        if slide_idx not in lookup:
            lookup[slide_idx] = len(lookup) + 1
    return lookup


def log_scene_slide_summary(metadata: list[dict]):
    scene_slide_map = build_scene_slide_map(metadata)
    mappings = scene_slide_map.get("mappings", [])
    if not mappings:
        return

    log.info("\n──────── slide ↔ scene 타임라인 ────────")
    timeline = [
        f"slide {row['slide_number']:03d}: scene {row['scene_index']:03d}"
        for row in mappings
    ]
    for i in range(0, len(timeline), 4):
        log.info("  " + " / ".join(timeline[i:i + 4]))

    log.info("──────── 상세 매핑 ────────")
    for row in mappings:
        scene_range = (
            f"{row['scene_annotation_start_index']}~{row['scene_annotation_end_index']}"
            if row["scene_annotation_start_index"] and row["scene_annotation_end_index"]
            else "-"
        )
        log.info(
            f"  [{row['scene_start_formatted']} ~ {row['scene_end_formatted']}] "
            f"scene {row['scene_index']:03d} -> slide {row['slide_number']:03d} "
            f"(canonical {row['slide_canonical_index']:03d}, "
            f"visit {row['slide_visit_order']}/{row['slide_group_size']}, "
            f"scene_annots={row['scene_annotation_count']}, "
            f"slide_annots={scene_range}, slide_total={row['slide_annotation_count_total']})"
        )
    log.info("────────────────────────────────────\n")


def _fmt_hms(sec: float) -> str:
    total = max(0, int(round(float(sec or 0.0))))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_scene_slide_map(metadata: list[dict]) -> dict:
    from collections import defaultdict

    by_scene: dict[int, list[dict]] = defaultdict(list)
    for item in metadata:
        by_scene[int(item.get("scene_index", item.get("slide_index", 0)) or 0)].append(item)

    slide_number_lookup = _build_slide_number_lookup(metadata)

    mappings: list[dict] = []
    for scene_idx in sorted(by_scene):
        items = by_scene[scene_idx]
        base = next((x for x in items if x.get("capture_type") == "base"), items[0])
        slide_idx = int(base.get("slide_canonical_index") or base.get("same_slide_canonical") or scene_idx)
        slide_number = int(base.get("slide_number", slide_number_lookup.get(slide_idx, slide_idx)) or slide_idx)
        scene_start = float(base.get("scene_start_sec", base.get("slide_start_sec", base.get("timestamp_sec", 0.0))) or 0.0)
        scene_end = float(base.get("scene_end_sec", base.get("slide_end_sec", scene_start)) or scene_start)
        mappings.append({
            "scene_index": scene_idx,
            "slide_number": slide_number,
            "slide_canonical_index": slide_idx,
            "slide_group": list(base.get("slide_group", base.get("same_slide_group", [slide_idx]))),
            "slide_group_size": int(base.get("slide_group_size", base.get("same_slide_group_size", 1)) or 1),
            "slide_visit_order": int(base.get("slide_visit_order", base.get("same_slide_visit_order", 1)) or 1),
            "slide_is_revisit": bool(base.get("slide_is_revisit", base.get("same_slide_is_revisit", False))),
            "previous_scene_index": base.get("previous_scene_index", base.get("same_slide_previous")),
            "next_scene_index": base.get("next_scene_index", base.get("same_slide_next")),
            "scene_start_sec": scene_start,
            "scene_end_sec": scene_end,
            "scene_start_formatted": _fmt_hms(scene_start),
            "scene_end_formatted": _fmt_hms(scene_end),
            "scene_annotation_count": int(base.get("scene_annotation_count", 0) or 0),
            "slide_annotation_count_total": int(base.get("slide_annotation_count_total", 0) or 0),
            "scene_annotation_start_index": int(base.get("scene_annotation_start_index", 0) or 0),
            "scene_annotation_end_index": int(base.get("scene_annotation_end_index", 0) or 0),
            "base_filename": base.get("filename"),
        })

    unique_slides = sorted({row["slide_canonical_index"] for row in mappings})
    return {
        "summary": {
            "total_scenes": len(mappings),
            "total_slides": len(unique_slides),
        },
        "timeline": [
            {
                "order": i + 1,
                "scene_index": row["scene_index"],
                "slide_number": row["slide_number"],
                "slide_canonical_index": row["slide_canonical_index"],
            }
            for i, row in enumerate(mappings)
        ],
        "mappings": mappings,
    }


def build_canonical_slide_annotations(metadata: list[dict]) -> dict:
    from collections import defaultdict

    by_scene: dict[int, list[dict]] = defaultdict(list)
    for item in metadata:
        by_scene[int(item.get("scene_index", item.get("slide_index", 0)) or 0)].append(item)

    slide_number_lookup = _build_slide_number_lookup(metadata)

    by_slide: dict[int, list[dict]] = defaultdict(list)
    for scene_idx, items in by_scene.items():
        base = next((x for x in items if x.get("capture_type") == "base"), items[0])
        slide_idx = int(base.get("slide_canonical_index") or base.get("same_slide_canonical") or scene_idx)
        by_slide[slide_idx].append(items)

    slides_payload: list[dict] = []
    total_annotations = 0

    for slide_idx in sorted(by_slide):
        scene_groups = sorted(
            by_slide[slide_idx],
            key=lambda items: min(float(x.get("timestamp_sec", 0.0) or 0.0) for x in items),
        )
        slide_number = slide_number_lookup.get(slide_idx, slide_idx)

        visits: list[dict] = []
        all_annotations: list[dict] = []

        for items in scene_groups:
            base = next((x for x in items if x.get("capture_type") == "base"), items[0])
            scene_idx = int(base.get("scene_index", base.get("slide_index", 0)) or 0)
            annots = sorted(
                [x for x in items if x.get("capture_type") == "annotation"],
                key=lambda x: (
                    int(x.get("annot_index", 0) or 0),
                    float(x.get("timestamp_sec", 0.0) or 0.0),
                ),
            )
            visit_entry = {
                "scene_index": scene_idx,
                "slide_number": slide_number,
                "visit_order": int(base.get("slide_visit_order", base.get("same_slide_visit_order", 1)) or 1),
                "is_revisit": bool(base.get("slide_is_revisit", base.get("same_slide_is_revisit", False))),
                "scene_start_sec": float(base.get("scene_start_sec", base.get("slide_start_sec", base.get("timestamp_sec", 0.0))) or 0.0),
                "scene_end_sec": float(base.get("scene_end_sec", base.get("slide_end_sec", base.get("timestamp_sec", 0.0))) or 0.0),
                "scene_start_formatted": _fmt_hms(base.get("scene_start_sec", base.get("slide_start_sec", base.get("timestamp_sec", 0.0)))),
                "scene_end_formatted": _fmt_hms(base.get("scene_end_sec", base.get("slide_end_sec", base.get("timestamp_sec", 0.0)))),
                "base_filename": base.get("filename"),
                "scene_annotation_count": len(annots),
                "scene_annotation_start_index": int(base.get("scene_annotation_start_index", 0) or 0),
                "scene_annotation_end_index": int(base.get("scene_annotation_end_index", 0) or 0),
                "annotations": [],
            }

            for annot in annots:
                annot_entry = {
                    "filename": annot.get("filename"),
                    "scene_index": scene_idx,
                    "slide_number": slide_number,
                    "slide_canonical_index": slide_idx,
                    "timestamp_sec": float(annot.get("timestamp_sec", 0.0) or 0.0),
                    "timestamp_formatted": _fmt_hms(annot.get("timestamp_sec", 0.0)),
                    "annot_index": int(annot.get("annot_index", 0) or 0),
                    "scene_annot_index": int(annot.get("scene_annot_index", annot.get("annot_index", 0)) or 0),
                    "slide_annot_index": int(annot.get("slide_annot_index", 0) or 0),
                }
                visit_entry["annotations"].append(annot_entry)
                all_annotations.append(annot_entry)
                total_annotations += 1

            visits.append(visit_entry)

        slides_payload.append({
            "slide_number": slide_number,
            "slide_canonical_index": slide_idx,
            "scene_indices": [visit["scene_index"] for visit in visits],
            "visit_count": len(visits),
            "total_annotation_count": len(all_annotations),
            "visits": visits,
            "all_annotations": all_annotations,
        })

    return {
        "summary": {
            "total_slides": len(slides_payload),
            "total_annotations": total_annotations,
        },
        "slides": slides_payload,
    }


# ──────────────────────────────────────────────
# 임계값 튜닝 도우미 (--tune 모드)
# ──────────────────────────────────────────────
def tune_thresholds(input_path: str, sample_sec: float = 30.0):
    """
    영상 앞부분 sample_sec초를 분석하여 diff 분포를 출력.
    튜닝 기준:
      ANNOT_INSTANT_RATIO:    p50 ~ p95 사이
      ANNOT_CUMULATIVE_RATIO: p95 ~ p99 사이
      ANIM_CUMULATIVE_RATIO:  p99 이상 (필기보다 훨씬 큰 값)
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(sample_sec * fps)

    instant_ratios    = []
    cumulative_ratios = []
    prev_small        = None
    base_small        = None

    cfg = Config()
    frame_no = 0
    while frame_no < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        if frame_no % cfg.PROCESS_EVERY_N_FRAMES != 0:
            continue
        decision_width = max(160, min(cfg.DECISION_RESIZE_WIDTH, cfg.RESIZE_WIDTH))
        small = resize_frame(frame, decision_width)
        if base_small is None:
            base_small = small.copy()
        if prev_small is not None:
            instant_ratios.append(count_changed_pixels(prev_small, small, cfg.ANNOT_DIFF_THRESHOLD))
            cumulative_ratios.append(count_changed_pixels(base_small, small, cfg.ANNOT_DIFF_THRESHOLD))
        prev_small = small

    cap.release()

    ir = np.array(instant_ratios)
    cr = np.array(cumulative_ratios)

    print("\n===== 임계값 튜닝 가이드 =====")
    print(f"[순간 diff]   p50={np.percentile(ir,50):.5f}  p95={np.percentile(ir,95):.5f}  max={ir.max():.5f}")
    print(f"  → ANNOT_INSTANT_RATIO 권장: p50~p95 (현재: {cfg.ANNOT_INSTANT_RATIO})")
    print()
    print(f"[누적 diff]   p50={np.percentile(cr,50):.5f}  p95={np.percentile(cr,95):.5f}"
          f"  p99={np.percentile(cr,99):.5f}  max={cr.max():.5f}")
    print(f"  → ANNOT_CUMULATIVE_RATIO 권장: p95~p99  (현재: {cfg.ANNOT_CUMULATIVE_RATIO})")
    print(f"  → ANIM_CUMULATIVE_RATIO  권장: p99 이상 (현재: {cfg.ANIM_CUMULATIVE_RATIO})")
    print("================================\n")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPT 강의 영상 슬라이드 추출기")
    parser.add_argument("--input",  "-i", default="input/lecture.mp4",    help="입력 영상 경로")
    parser.add_argument("--output", "-o", default="output_slides/", help="출력 디렉토리")
    parser.add_argument("--debug",  action="store_true",             help="디버그 로그 출력")
    parser.add_argument("--tune",   action="store_true",             help="임계값 튜닝 모드")
    parser.add_argument(
        "--decode-backend",
        choices=["opencv", "ffmpeg-cuda", "ffmpeg-videotoolbox", "auto"],
        default=Config.DECODE_BACKEND,
        help="프레임 디코드 백엔드 선택 (default: 환경변수 GRAPHLEC_SLIDE_DECODE_BACKEND 또는 opencv)",
    )
    args = parser.parse_args()

    if args.tune:
        tune_thresholds(args.input)
    else:
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        extract_slides(args.input, args.output, debug=args.debug, decode_backend=args.decode_backend)
