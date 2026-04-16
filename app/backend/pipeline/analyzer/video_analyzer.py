"""
영상 품질 메트릭 추출 모듈

영상 파일에서 해상도, 프레임레이트, 비트레이트, 선명도, 노이즈 레벨을 추출합니다.

Output (JSON):
{
  "metadata": { "video_path", "duration", ... },
  "quality": {
    "resolution":   { "width", "height", "label" },
    "framerate":    { "fps", "label" },
    "bitrate":      { "kbps", "label" },
    "sharpness":    { "score", "label" },
    "noise_level":  { "score", "label" }
  },
  "verdict": {
    "passed": true/false,
    "issues": ["해상도 기준 미달", ...]
  }
}

사용법:
  from extract_video_quality import VideoQualityExtractor
  extractor = VideoQualityExtractor()
  result = extractor.run("lecture.mp4")

필요 패키지:
  pip install opencv-python numpy
  ffprobe (ffmpeg 설치 시 포함)
"""

import cv2
import json
import time
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


# ============================================================================ #
#  품질 기준값 (조정 가능)                                                        #
# ============================================================================ #

QUALITY_STANDARDS = {
    "resolution": {
        "min_width": 1024,
        "min_height": 768,
        "label_map": {
            (3840, 2160): "4K",
            (1920, 1080): "FHD",
            (1280, 720):  "HD",
            (854, 480):   "SD",
        }
    },
    "framerate": {
        "min_fps": 15.0,      # 슬라이드 영상은 15fps면 충분
    },
    "bitrate": {
        "min_kbps": 300,      # 슬라이드는 정적이라 낮은 비트레이트로도 충분
    },
    "sharpness": {
        "min_score": 100.0,   # Laplacian 분산 기준
    },
    "noise_level": {
        "max_score": 50.0,    # 슬라이드 전환 시 프레임 차이가 크므로 허용 범위 확대
    }
}


# ============================================================================ #
#  해상도 추출                                                                   #
# ============================================================================ #

def extract_resolution(cap: cv2.VideoCapture) -> Dict:
    """해상도 추출"""
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 해상도 라벨 결정
    label = "Unknown"
    label_map = QUALITY_STANDARDS["resolution"]["label_map"]
    for (w, h), lbl in label_map.items():
        if width >= w and height >= h:
            label = lbl
            break
    else:
        label = "Sub-SD"

    passed = (
        width  >= QUALITY_STANDARDS["resolution"]["min_width"] and
        height >= QUALITY_STANDARDS["resolution"]["min_height"]
    )

    return {
        "width": width,
        "height": height,
        "label": label,
        "passed": passed
    }


# ============================================================================ #
#  프레임레이트 추출                                                              #
# ============================================================================ #

def extract_framerate(cap: cv2.VideoCapture) -> Dict:
    """프레임레이트 추출"""
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps >= 60:
        label = "60fps+"
    elif fps >= 30:
        label = "30fps"
    elif fps >= 24:
        label = "24fps"
    else:
        label = f"{fps:.1f}fps (낮음)"

    passed = fps >= QUALITY_STANDARDS["framerate"]["min_fps"]

    return {
        "fps": round(fps, 2),
        "label": label,
        "passed": passed
    }


# ============================================================================ #
#  비트레이트 추출 (ffprobe)                                                     #
# ============================================================================ #

def extract_bitrate(video_path: str) -> Dict:
    """
    ffprobe를 사용한 비트레이트 추출
    ffprobe 없을 경우 파일 크기 기반 근사치 사용
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                video_path
            ],
            capture_output=True, text=True, timeout=30
        )
        data = json.loads(result.stdout)
        bit_rate = int(data["format"].get("bit_rate", 0))
        kbps = bit_rate // 1000
        source = "ffprobe"

    except Exception:
        # ffprobe 없을 경우: 파일 크기 / 영상 길이로 근사
        try:
            file_size = Path(video_path).stat().st_size  # bytes
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            duration = total_frames / fps if fps > 0 else 1
            kbps = int((file_size * 8) / duration / 1000)
            source = "estimated"
        except Exception as e:
            return {"kbps": 0, "label": "측정 불가", "passed": False, "source": "error"}

    if kbps >= 8000:
        label = "고품질 (8Mbps+)"
    elif kbps >= 4000:
        label = "고품질 (4Mbps+)"
    elif kbps >= 1000:
        label = "보통 (1Mbps+)"
    else:
        label = f"저품질 ({kbps}kbps)"

    passed = kbps >= QUALITY_STANDARDS["bitrate"]["min_kbps"]

    return {
        "kbps": kbps,
        "label": label,
        "passed": passed,
        "source": source
    }


# ============================================================================ #
#  선명도 측정 (Laplacian 분산)                                                  #
# ============================================================================ #

def extract_sharpness(cap: cv2.VideoCapture, sample_count: int = 10) -> Dict:
    """
    Laplacian 분산으로 선명도 측정
    높을수록 선명 (100 이상 권장)

    여러 프레임을 샘플링하여 평균값 사용
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps

    # 균등 간격으로 샘플 프레임 선택
    sample_positions = [
        int(total_frames * i / (sample_count + 1))
        for i in range(1, sample_count + 1)
    ]

    scores = []
    for pos in sample_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        scores.append(score)

    # 원래 위치로 복원
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if not scores:
        return {"score": 0.0, "label": "측정 불가", "passed": False}

    avg_score = float(np.mean(scores))
    min_score = QUALITY_STANDARDS["sharpness"]["min_score"]

    if avg_score >= 500:
        label = "매우 선명"
    elif avg_score >= 200:
        label = "선명"
    elif avg_score >= 100:
        label = "보통"
    elif avg_score >= 50:
        label = "약간 흐림"
    else:
        label = "흐림"

    passed = avg_score >= min_score

    return {
        "score": round(avg_score, 2),
        "label": label,
        "passed": passed,
        "sample_count": len(scores)
    }


# ============================================================================ #
#  노이즈 레벨 측정 (프레임 간 표준편차)                                           #
# ============================================================================ #

def extract_noise_level(cap: cv2.VideoCapture, sample_count: int = 10) -> Dict:
    """
    정적 구간에서 프레임 간 픽셀 표준편차로 노이즈 측정
    낮을수록 노이즈 적음 (15 이하 권장)

    방법: 연속된 두 프레임의 차이 평균으로 노이즈 근사
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 영상 앞 1/4 ~ 3/4 구간에서 샘플링 (인트로/아웃트로 제외)
    start = total_frames // 4
    end   = total_frames * 3 // 4
    interval = max(1, (end - start) // sample_count)

    noise_scores = []
    prev_frame = None

    for i in range(sample_count):
        pos = start + i * interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
        gray_small = cv2.resize(gray, (320, 240))

        if prev_frame is not None:
            diff = np.abs(gray_small - prev_frame)
            noise_scores.append(float(np.std(diff)))

        prev_frame = gray_small

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if not noise_scores:
        return {"score": 0.0, "label": "측정 불가", "passed": True}

    avg_score = float(np.mean(noise_scores))
    max_score = QUALITY_STANDARDS["noise_level"]["max_score"]

    if avg_score <= 5:
        label = "매우 낮음 (우수)"
    elif avg_score <= 10:
        label = "낮음 (양호)"
    elif avg_score <= 15:
        label = "보통"
    elif avg_score <= 25:
        label = "높음 (주의)"
    else:
        label = "매우 높음 (불량)"

    passed = avg_score <= max_score

    return {
        "score": round(avg_score, 2),
        "label": label,
        "passed": passed,
        "sample_count": len(noise_scores)
    }


# ============================================================================ #
#  메인 추출기                                                                   #
# ============================================================================ #

class VideoQualityExtractor:
    """영상 품질 메트릭 통합 추출기"""

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def run(self, video_path: str) -> Dict:
        """
        전체 품질 메트릭 추출

        Args:
            video_path: 입력 영상 경로

        Returns:
            result: { "metadata", "quality", "verdict" }
        """
        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"🎬 영상 품질 메트릭 추출")
        print(f"{'='*60}")
        print(f"  영상: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"영상 파일을 열 수 없습니다: {video_path}")

        fps        = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration   = total_frames / fps if fps > 0 else 0

        # 각 메트릭 추출
        print("\n측정 중...")
        print("  - 해상도...")
        resolution  = extract_resolution(cap)

        print("  - 프레임레이트...")
        framerate   = extract_framerate(cap)

        print("  - 비트레이트...")
        bitrate     = extract_bitrate(video_path)

        print("  - 선명도...")
        sharpness   = extract_sharpness(cap)

        print("  - 노이즈 레벨...")
        noise_level = extract_noise_level(cap)

        cap.release()

        # 종합 판정
        issues = []
        if not resolution["passed"]:
            issues.append(f"해상도 기준 미달 ({resolution['width']}x{resolution['height']}, 최소 1280x720 권장)")
        if not framerate["passed"]:
            issues.append(f"프레임레이트 기준 미달 ({framerate['fps']}fps, 최소 24fps 권장)")
        if not bitrate["passed"]:
            issues.append(f"비트레이트 기준 미달 ({bitrate['kbps']}kbps, 최소 1000kbps 권장)")
        if not sharpness["passed"]:
            issues.append(f"선명도 기준 미달 (score: {sharpness['score']}, 최소 100 권장)")
        if not noise_level["passed"]:
            issues.append(f"노이즈 레벨 기준 초과 (score: {noise_level['score']}, 최대 15 권장)")

        result = {
            "metadata": {
                "video_path": video_path,
                "duration": round(duration, 2),
                "total_frames": total_frames,
                "processing_time": round(time.time() - start_time, 2)
            },
            "quality": {
                "resolution":  resolution,
                "framerate":   framerate,
                "bitrate":     bitrate,
                "sharpness":   sharpness,
                "noise_level": noise_level
            },
            "verdict": {
                "passed": len(issues) == 0,
                "issues": issues
            }
        }

        # 결과 저장
        output_path = Path(self.output_dir) / "video_quality.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # 결과 출력
        print(f"\n{'='*60}")
        print(f"📊 품질 측정 결과")
        print(f"{'='*60}")
        print(f"  해상도    : {resolution['width']}x{resolution['height']} ({resolution['label']}) {'✓' if resolution['passed'] else '✗'}")
        print(f"  프레임레이트: {framerate['fps']}fps ({framerate['label']}) {'✓' if framerate['passed'] else '✗'}")
        print(f"  비트레이트 : {bitrate['kbps']}kbps ({bitrate['label']}) {'✓' if bitrate['passed'] else '✗'}")
        print(f"  선명도    : {sharpness['score']} ({sharpness['label']}) {'✓' if sharpness['passed'] else '✗'}")
        print(f"  노이즈레벨 : {noise_level['score']} ({noise_level['label']}) {'✓' if noise_level['passed'] else '✗'}")
        print(f"\n  최종 판정 : {'✅ 통과' if result['verdict']['passed'] else '❌ 미달'}")
        if issues:
            for issue in issues:
                print(f"    • {issue}")
        print(f"\n  저장 위치 : {output_path}")
        print(f"  처리 시간 : {result['metadata']['processing_time']}초")

        return result


# ============================================================================ #
#  단독 실행                                                                     #
# ============================================================================ #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="영상 품질 메트릭 추출")
    parser.add_argument("-v", "--video", required=True, help="입력 영상 경로")
    parser.add_argument("-o", "--output", default="./output", help="출력 폴더")
    args = parser.parse_args()

    extractor = VideoQualityExtractor(output_dir=args.output)
    extractor.run(args.video)
