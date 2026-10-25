"""
Evaluation metrics for calibration and benchmarking.

Provides precision, recall, F1, confusion matrices, and classification reports
for measuring pipeline accuracy against ground-truth datasets. Also provides
BenchmarkResult for structured benchmark output.

No external dependencies — uses only Python standard library.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Binary classification metrics
# ---------------------------------------------------------------------------


def precision(tp: int, fp: int) -> float:
    """Precision = TP / (TP + FP). Returns 0.0 when denominator is zero."""
    denom = tp + fp
    return tp / denom if denom > 0 else 0.0


def recall(tp: int, fn: int) -> float:
    """Recall = TP / (TP + FN). Returns 0.0 when denominator is zero."""
    denom = tp + fn
    return tp / denom if denom > 0 else 0.0


def f1_score(prec: float, rec: float) -> float:
    """F1 = 2 * precision * recall / (precision + recall). Returns 0.0 when both are zero."""
    denom = prec + rec
    return 2 * prec * rec / denom if denom > 0 else 0.0


def accuracy(correct: int, total: int) -> float:
    """Accuracy = correct / total.  Returns 0.0 when total is zero."""
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def confusion_matrix(predictions: list, actuals: list) -> dict:
    """
    Binary confusion matrix.

    Parameters
    ----------
    predictions : list[bool]
        Predicted labels (True = positive).
    actuals : list[bool]
        Ground-truth labels (True = positive).

    Returns
    -------
    dict with keys "tp", "fp", "tn", "fn".

    Raises
    ------
    ValueError
        If predictions and actuals have different lengths.
    """
    if len(predictions) != len(actuals):
        raise ValueError(f"predictions ({len(predictions)}) and actuals ({len(actuals)}) must have the same length")
    tp = fp = tn = fn = 0
    for pred, actual in zip(predictions, actuals):
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and actual:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


# ---------------------------------------------------------------------------
# Multi-class classification report
# ---------------------------------------------------------------------------


def classification_report(
    predictions: list,
    actuals: list,
    labels: list | None = None,
) -> dict:
    """
    Per-class and aggregate precision / recall / F1.

    Parameters
    ----------
    predictions : list[str]
        Predicted labels.
    actuals : list[str]
        Ground-truth labels.
    labels : list[str] | None
        Label names.  If None, inferred from the union of predictions + actuals.

    Returns
    -------
    dict of the form::

        {
            "per_class": {
                "label_a": {"precision": ..., "recall": ..., "f1": ..., "support": ...},
                ...
            },
            "macro_avg": {"precision": ..., "recall": ..., "f1": ...},
            "micro_avg": {"precision": ..., "recall": ..., "f1": ...},
            "accuracy": ...,
            "total": ...
        }
    """
    if len(predictions) != len(actuals):
        raise ValueError(f"predictions ({len(predictions)}) and actuals ({len(actuals)}) must have the same length")

    if labels is None:
        labels = sorted(set(predictions) | set(actuals))

    per_class: dict[str, dict] = {}
    micro_tp = micro_fp = micro_fn = 0

    for label in labels:
        tp = sum(1 for p, a in zip(predictions, actuals) if p == label and a == label)
        fp = sum(1 for p, a in zip(predictions, actuals) if p == label and a != label)
        fn = sum(1 for p, a in zip(predictions, actuals) if p != label and a == label)
        support = sum(1 for a in actuals if a == label)

        p = precision(tp, fp)
        r = recall(tp, fn)
        f = f1_score(p, r)

        per_class[label] = {"precision": p, "recall": r, "f1": f, "support": support}
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

    # Macro average (unweighted mean of per-class metrics)
    n_labels = len(labels) if labels else 1
    macro_p = sum(c["precision"] for c in per_class.values()) / n_labels
    macro_r = sum(c["recall"] for c in per_class.values()) / n_labels
    macro_f = f1_score(macro_p, macro_r)

    # Micro average (global TP/FP/FN)
    micro_p = precision(micro_tp, micro_fp)
    micro_r = recall(micro_tp, micro_fn)
    micro_f = f1_score(micro_p, micro_r)

    # Overall accuracy
    correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
    total = len(predictions)

    return {
        "per_class": per_class,
        "macro_avg": {"precision": macro_p, "recall": macro_r, "f1": macro_f},
        "micro_avg": {"precision": micro_p, "recall": micro_r, "f1": micro_f},
        "accuracy": correct / total if total > 0 else 0.0,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Threshold sweep helper
# ---------------------------------------------------------------------------


def threshold_sweep(
    scores: list[float],
    actuals: list[bool],
    thresholds: list[float] | None = None,
) -> list[dict]:
    """
    Evaluate binary classification at multiple score thresholds.

    Parameters
    ----------
    scores : list[float]
        Predicted scores (higher = more positive).
    actuals : list[bool]
        Ground-truth labels.
    thresholds : list[float] | None
        Thresholds to evaluate.  If None, uses [0.1, 0.2, ..., 0.9].

    Returns
    -------
    list[dict], each with "threshold", "precision", "recall", "f1", "tp", "fp", "tn", "fn".
    """
    if thresholds is None:
        thresholds = [round(t / 10, 1) for t in range(1, 10)]

    results = []
    for thresh in thresholds:
        preds = [s >= thresh for s in scores]
        cm = confusion_matrix(preds, actuals)
        p = precision(cm["tp"], cm["fp"])
        r = recall(cm["tp"], cm["fn"])
        f = f1_score(p, r)
        results.append(
            {
                "threshold": thresh,
                "precision": p,
                "recall": r,
                "f1": f,
                **cm,
            }
        )
    return results


# ---------------------------------------------------------------------------
# BenchmarkResult — structured benchmark output
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Structured output from a calibration benchmark run."""

    name: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metrics: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    details: list[dict[str, Any]] = field(default_factory=list)

    @property
    def summary(self) -> str:
        """Human-readable one-line summary."""
        parts = [f"Benchmark: {self.name}"]
        for key, val in self.metrics.items():
            if isinstance(val, float):
                parts.append(f"{key}={val:.3f}")
            else:
                parts.append(f"{key}={val}")
        return " | ".join(parts)


def save_report(result: BenchmarkResult, path: str) -> None:
    """Save a BenchmarkResult to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False)


def load_report(path: str) -> BenchmarkResult:
    """Load a BenchmarkResult from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return BenchmarkResult(**data)


# ---------------------------------------------------------------------------
# JSONL dataset loader
# ---------------------------------------------------------------------------


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file (one JSON object per line)."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
