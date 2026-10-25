"""
Tests for src/core/metrics.py — evaluation metrics for calibration.
"""

import json
import os
import tempfile

import pytest

from src.core.metrics import (
    BenchmarkResult,
    accuracy,
    classification_report,
    confusion_matrix,
    f1_score,
    load_jsonl,
    load_report,
    precision,
    recall,
    save_report,
    threshold_sweep,
)

# ── precision / recall / f1 / accuracy ──────────────────────────────────


class TestPrecision:
    def test_perfect(self):
        assert precision(10, 0) == 1.0

    def test_zero_tp(self):
        assert precision(0, 5) == 0.0

    def test_zero_denominator(self):
        assert precision(0, 0) == 0.0

    def test_typical(self):
        # 8 / (8 + 2) = 0.8
        assert abs(precision(8, 2) - 0.8) < 1e-9


class TestRecall:
    def test_perfect(self):
        assert recall(10, 0) == 1.0

    def test_zero_tp(self):
        assert recall(0, 5) == 0.0

    def test_zero_denominator(self):
        assert recall(0, 0) == 0.0

    def test_typical(self):
        # 6 / (6 + 4) = 0.6
        assert abs(recall(6, 4) - 0.6) < 1e-9


class TestF1Score:
    def test_perfect(self):
        assert f1_score(1.0, 1.0) == 1.0

    def test_zero_both(self):
        assert f1_score(0.0, 0.0) == 0.0

    def test_typical(self):
        # 2 * 0.8 * 0.6 / (0.8 + 0.6) = 0.6857...
        f = f1_score(0.8, 0.6)
        assert abs(f - 2 * 0.8 * 0.6 / 1.4) < 1e-9

    def test_one_zero(self):
        assert f1_score(1.0, 0.0) == 0.0
        assert f1_score(0.0, 1.0) == 0.0


class TestAccuracy:
    def test_perfect(self):
        assert accuracy(10, 10) == 1.0

    def test_zero(self):
        assert accuracy(0, 10) == 0.0

    def test_zero_total(self):
        assert accuracy(0, 0) == 0.0

    def test_typical(self):
        assert abs(accuracy(7, 10) - 0.7) < 1e-9


# ── confusion_matrix ────────────────────────────────────────────────────


class TestConfusionMatrix:
    def test_all_true_positive(self):
        cm = confusion_matrix([True, True, True], [True, True, True])
        assert cm == {"tp": 3, "fp": 0, "tn": 0, "fn": 0}

    def test_all_true_negative(self):
        cm = confusion_matrix([False, False], [False, False])
        assert cm == {"tp": 0, "fp": 0, "tn": 2, "fn": 0}

    def test_mixed(self):
        preds = [True, True, False, False, True]
        actuals = [True, False, True, False, True]
        cm = confusion_matrix(preds, actuals)
        assert cm == {"tp": 2, "fp": 1, "tn": 1, "fn": 1}

    def test_empty(self):
        cm = confusion_matrix([], [])
        assert cm == {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            confusion_matrix([True], [True, False])


# ── classification_report ───────────────────────────────────────────────


class TestClassificationReport:
    def test_binary(self):
        preds = ["pos", "pos", "neg", "neg", "pos"]
        actuals = ["pos", "neg", "neg", "pos", "pos"]
        report = classification_report(preds, actuals)

        assert report["total"] == 5
        assert "pos" in report["per_class"]
        assert "neg" in report["per_class"]

        # pos: tp=2, fp=1, fn=1 → p=2/3, r=2/3, f1=2/3
        pos = report["per_class"]["pos"]
        assert abs(pos["precision"] - 2 / 3) < 1e-9
        assert abs(pos["recall"] - 2 / 3) < 1e-9
        assert pos["support"] == 3

    def test_multiclass(self):
        preds = ["A", "B", "C", "A", "B", "C"]
        actuals = ["A", "B", "C", "B", "C", "A"]
        report = classification_report(preds, actuals, labels=["A", "B", "C"])

        assert report["total"] == 6
        assert len(report["per_class"]) == 3
        # 3 correct out of 6
        assert abs(report["accuracy"] - 0.5) < 1e-9

    def test_perfect_classification(self):
        preds = actuals = ["A", "B", "A", "B"]
        report = classification_report(preds, actuals)

        assert report["accuracy"] == 1.0
        assert report["macro_avg"]["precision"] == 1.0
        assert report["micro_avg"]["recall"] == 1.0

    def test_all_wrong(self):
        preds = ["A", "A", "A"]
        actuals = ["B", "B", "B"]
        report = classification_report(preds, actuals, labels=["A", "B"])

        assert report["accuracy"] == 0.0
        assert report["per_class"]["A"]["precision"] == 0.0
        assert report["per_class"]["B"]["recall"] == 0.0

    def test_empty(self):
        report = classification_report([], [])
        assert report["total"] == 0
        assert report["accuracy"] == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            classification_report(["A"], ["A", "B"])

    def test_labels_inferred(self):
        report = classification_report(["x", "y", "x"], ["x", "x", "y"])
        assert set(report["per_class"].keys()) == {"x", "y"}


# ── threshold_sweep ─────────────────────────────────────────────────────


class TestThresholdSweep:
    def test_default_thresholds(self):
        scores = [0.1, 0.5, 0.9]
        actuals = [False, True, True]
        results = threshold_sweep(scores, actuals)
        assert len(results) == 9  # 0.1 through 0.9
        # At threshold 0.5: preds=[F, T, T] → tp=2, fp=0, fn=0, tn=1
        r05 = [r for r in results if r["threshold"] == 0.5][0]
        assert r05["tp"] == 2
        assert r05["fp"] == 0
        assert r05["precision"] == 1.0

    def test_custom_thresholds(self):
        scores = [0.3, 0.7]
        actuals = [False, True]
        results = threshold_sweep(scores, actuals, thresholds=[0.5])
        assert len(results) == 1
        assert results[0]["threshold"] == 0.5

    def test_all_below_threshold(self):
        scores = [0.1, 0.2, 0.3]
        actuals = [True, True, True]
        results = threshold_sweep(scores, actuals, thresholds=[0.9])
        assert results[0]["recall"] == 0.0


# ── BenchmarkResult ─────────────────────────────────────────────────────


class TestBenchmarkResult:
    def test_creation(self):
        br = BenchmarkResult(name="test_bench")
        assert br.name == "test_bench"
        assert br.metrics == {}
        assert br.thresholds == {}
        assert br.details == []
        assert br.timestamp  # non-empty

    def test_summary(self):
        br = BenchmarkResult(
            name="embed_test",
            metrics={"precision": 0.85, "recall": 0.72, "count": 100},
        )
        s = br.summary
        assert "embed_test" in s
        assert "0.850" in s
        assert "0.720" in s
        assert "100" in s

    def test_save_load_round_trip(self):
        br = BenchmarkResult(
            name="round_trip",
            metrics={"f1": 0.9},
            thresholds={"similarity": 0.35},
            details=[{"id": "n1", "correct": True}],
        )
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            save_report(br, path)
            loaded = load_report(path)
            assert loaded.name == "round_trip"
            assert loaded.metrics["f1"] == 0.9
            assert loaded.thresholds["similarity"] == 0.35
            assert len(loaded.details) == 1
            assert loaded.details[0]["correct"] is True
            assert loaded.timestamp == br.timestamp
        finally:
            os.unlink(path)

    def test_save_creates_valid_json(self):
        br = BenchmarkResult(name="json_check", metrics={"x": 1.5})
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            save_report(br, path)
            with open(path) as f:
                data = json.load(f)
            assert data["name"] == "json_check"
            assert data["metrics"]["x"] == 1.5
        finally:
            os.unlink(path)


# ── load_jsonl ──────────────────────────────────────────────────────────


class TestLoadJsonl:
    def test_basic(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            with open(path, "w") as f:
                f.write('{"a": 1}\n')
                f.write('{"b": 2}\n')
            records = load_jsonl(path)
            assert len(records) == 2
            assert records[0]["a"] == 1
            assert records[1]["b"] == 2
        finally:
            os.unlink(path)

    def test_empty_lines_skipped(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            with open(path, "w") as f:
                f.write('{"x": 1}\n\n\n{"y": 2}\n')
            records = load_jsonl(path)
            assert len(records) == 2
        finally:
            os.unlink(path)

    def test_empty_file(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            records = load_jsonl(path)
            assert records == []
        finally:
            os.unlink(path)
