# Final Issue Verification Scoring

## Purpose

Final issue verification receives issue candidates that have already been classified into one category:

- `factual_error`
- `temporal_error`
- `scope_overclaim`
- `confusing_explanation`

Each verifier batch contains only one slide and one category. The model must not reclassify the issue. It only decides whether the issue passes the assigned category gate after reading the slide context.

The final verifier no longer uses one shared prompt with a swapped rubric. It dispatches to four independent prompt paths:

- A-only factual error verifier
- B-only temporal/currentness verifier
- C-only scope overclaim verifier
- D-only confusing explanation verifier

## Model Output

A/B/D verifiers output three scores:

```json
{
  "judgments": [
    {
      "id": "input id",
      "judgment": "valid_issue | partially_resolved | not_issue | insufficient_context",
      "is_valid_issue": 0.0,
      "category_severity": 0.0,
      "context_unresolved": 0.0,
      "reason": "short rationale",
      "minimal_fix": "optional correction"
    }
  ]
}
```

`context_resolution` may remain in legacy artifacts for display, but new model output and scoring use `context_unresolved`.

C `scope_overclaim` currently uses a compact output because the final score is based only on `is_valid_issue`:

```json
{
  "judgments": [
    {
      "id": "input id",
      "judgment": "valid_issue | partially_resolved | not_issue | insufficient_context",
      "is_valid_issue": 0.0,
      "reason": "short rationale",
      "evidence": "short context evidence"
    }
  ]
}
```

## Score Definitions

`is_valid_issue` means how strongly the issue passes the assigned category criteria.

`category_severity` means how important the issue is inside that category, assuming it is valid.

`context_unresolved` means how much of the issue remains after reading the provided lecture context.

```text
0.0 = fully resolved by context
1.0 = not resolved at all
```

Context unresolved must be interpreted from the whole lecture context, not from keyword matching. If the surrounding lecture naturally narrows a phrase into an educational simplification, contrastive explanation, management role, mediation role, permission-control role, or representative path, the model should assign low `context_unresolved`. Advanced implementation exceptions, stricter taxonomy, or "could be phrased more precisely" are not enough to keep an issue.

## Category-Specific Server Formula

The server computes each model score with category-specific weights:

```text
A factual_error:
  0.55 * is_valid_issue
+ 0.25 * category_severity
+ 0.20 * context_unresolved

B temporal_error:
  0.45 * is_valid_issue
+ 0.25 * category_severity
+ 0.30 * context_unresolved
  cap: 0.79

C scope_overclaim:
  is_valid_issue

D confusing_explanation:
  0.40 * is_valid_issue
+ 0.20 * category_severity
+ 0.40 * context_unresolved
  cap: 0.79
  if judgment != valid_issue, cap: 0.39
```

Meaning:

- an issue can be valid and severe
- but if the context clearly resolves it, `context_unresolved` should be low
- if context does not resolve it, `context_unresolved` should be high
- context is one weighted factor, not a multiplicative gate
- B/C/D do not auto-confirm by default; they stay below the confirmed threshold unless later pipeline stages add stronger grounding

## Judgment Labels

Model labels should align with scores:

```text
valid_issue:
- is_valid_issue >= 0.70
- category_severity >= 0.50
- context_unresolved >= 0.60

partially_resolved:
- is_valid_issue >= 0.50
- context_unresolved > 0.20
- context_unresolved < 0.60

not_issue:
- is_valid_issue < 0.50
- or context_unresolved <= 0.20

insufficient_context:
- actual utterance, slide context, or temporal basis is insufficient
```

`partially_resolved` does not mean automatically kept. It means context weakens the issue, and the server formula lowers the score through a lower `context_unresolved`.

For `confusing_explanation`, `partially_resolved`, `not_issue`, and `insufficient_context` are capped below the professor-check threshold. D-type items should surface only when a concrete misconception remains clearly after context.

## Final Aggregation

Model scores are combined by configured model weights:

```text
final_severity_score =
  sum(final_model_score * model_weight)
```

Default model weights:

```text
gpt-5.4=0.35, claude=0.45, grok=0.20
```

## Status Thresholds

```text
score >= 0.80  -> confirmed
score > 0.40   -> professor_check
score <= 0.40  -> rejected
```
