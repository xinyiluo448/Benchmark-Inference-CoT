from dataclasses import dataclass
from typing import Optional, Tuple

from bench_runner.runner import extract_last_number, numeric_equal


@dataclass
class EvalResult:
    correct: Optional[int]  # 1/0 or None if not applicable
    reference: str


def evaluate(metric: str, prediction: str, reference: str) -> EvalResult:
    """
    Evaluate a single prediction/reference pair with a metric key.
    Supported metrics:
      - exact: case-insensitive exact match with light label normalization (positive/negative detection)
      - numeric: compare last numbers (float), fallback to exact
      - none: skip evaluation
    """
    metric = (metric or "none").lower()
    pred_raw = prediction.strip()
    ref_raw = reference.strip()
    pred_norm = pred_raw.lower()
    ref_norm = ref_raw.lower()

    if metric == "exact":
        # Light heuristic: map common sentiment words to canonical labels if unambiguous.
        def normalize_label(text: str) -> str:
            has_pos = "positive" in text
            has_neg = "negative" in text
            if has_pos and not has_neg:
                return "positive"
            if has_neg and not has_pos:
                return "negative"
            return text

        pred_label = normalize_label(pred_norm)
        ref_label = normalize_label(ref_norm)
        return EvalResult(correct=int(pred_label == ref_label), reference=reference)

    if metric == "numeric":
        pred_num = extract_last_number(pred_raw)
        ref_num = extract_last_number(ref_raw)
        if pred_num and ref_num and numeric_equal(pred_num, ref_num):
            return EvalResult(correct=1, reference=reference)
        return EvalResult(correct=int(pred_norm == ref_norm), reference=reference)

    # default: no evaluation
    return EvalResult(correct=None, reference=reference)


# Default metric mapping for built-in tasks
DEFAULT_TASK_METRIC = {
    "classification-sst2": "exact",
    "reasoning-gsm8k": "numeric",
    "summarization-xsum": "rouge",  # handled via --eval-rouge; per-sample correct is None
}
