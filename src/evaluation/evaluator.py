"""
src/evaluation/evaluator.py

Comprehensive evaluation of trained Hinglish sentiment models.

Produces:
  - Classification report (precision, recall, F1 per class + macro/weighted)
  - Confusion matrix (normalized)
  - Error analysis: hardest examples, confidence calibration
  - Model comparison table across different checkpoints
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ID2LABEL, LABEL2ID

LABEL_NAMES = ["Negative", "Neutral", "Positive"]


class HinglishEvaluator:
    """
    Evaluate a trained predictor on a test set and produce rich analysis.

    Usage:
        from inference.predictor import HinglishSentimentPredictor
        predictor = HinglishSentimentPredictor("outputs/best_model")
        evaluator = HinglishEvaluator(predictor)

        # Evaluate on a DataFrame with 'text' and 'label' columns
        report = evaluator.evaluate(test_df)
        evaluator.print_report(report)
        evaluator.save_report(report, "outputs/eval_report.json")
    """

    def __init__(self, predictor):
        self.predictor = predictor

    def evaluate(self, df: pd.DataFrame, text_col: str = "text", label_col: str = "label") -> Dict:
        """
        Run full evaluation on a labeled DataFrame.

        Returns dict with all metrics and analysis.
        """
        print(f"Evaluating on {len(df)} samples...")

        # Get predictions
        results = self.predictor.predict_batch(df[text_col].tolist(), show_progress=True)

        y_true = df[label_col].tolist()
        y_pred = [r["label"] for r in results]
        confidences = [r["confidence"] for r in results]

        # ── Core metrics ──────────────────────────────────────────────────────
        report_dict = classification_report(
            y_true, y_pred,
            labels=LABEL_NAMES,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(y_true, y_pred, labels=LABEL_NAMES)
        cm_normalized = (cm.astype(float) / cm.sum(axis=1, keepdims=True)).round(3)

        macro_f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        accuracy   = accuracy_score(y_true, y_pred)

        # ── Error analysis ────────────────────────────────────────────────────
        errors = []
        correct = []
        for i, (r, true_label) in enumerate(zip(results, y_true)):
            entry = {
                "index":       i,
                "text":        r["text"],
                "true_label":  true_label,
                "pred_label":  r["label"],
                "confidence":  r["confidence"],
                "scores":      r["scores"],
                "is_correct":  r["label"] == true_label,
            }
            if not entry["is_correct"]:
                errors.append(entry)
            else:
                correct.append(entry)

        # Hardest correct (high confidence + correct) — well-calibrated examples
        hard_errors = sorted(errors, key=lambda x: x["confidence"], reverse=True)[:20]

        # Low-confidence correct — uncertain but right
        uncertain_correct = sorted(correct, key=lambda x: x["confidence"])[:10]

        # ── Confidence calibration ────────────────────────────────────────────
        conf_buckets = self._calibration_stats(y_true, y_pred, confidences)

        return {
            "accuracy":          round(accuracy, 4),
            "macro_f1":          round(macro_f1, 4),
            "weighted_f1":       round(weighted_f1, 4),
            "per_class":         {
                label: {
                    "precision": round(report_dict[label]["precision"], 4),
                    "recall":    round(report_dict[label]["recall"], 4),
                    "f1":        round(report_dict[label]["f1-score"], 4),
                    "support":   report_dict[label]["support"],
                }
                for label in LABEL_NAMES
            },
            "confusion_matrix":             cm.tolist(),
            "confusion_matrix_normalized":  cm_normalized.tolist(),
            "label_names":       LABEL_NAMES,
            "total_samples":     len(df),
            "total_errors":      len(errors),
            "error_rate":        round(len(errors) / len(df), 4),
            "hard_errors":       hard_errors,
            "uncertain_correct": uncertain_correct,
            "calibration":       conf_buckets,
            "all_predictions":   results,
        }

    def _calibration_stats(
        self, y_true: list, y_pred: list, confidences: list, n_bins: int = 5
    ) -> List[Dict]:
        """
        Compute accuracy per confidence bucket to check calibration.
        A well-calibrated model: 90% confident → ~90% accurate.
        """
        bins = np.linspace(0, 1, n_bins + 1)
        buckets = []
        for low, high in zip(bins[:-1], bins[1:]):
            mask = [low <= c < high for c in confidences]
            if not any(mask):
                continue
            bucket_true = [y for y, m in zip(y_true, mask) if m]
            bucket_pred = [y for y, m in zip(y_pred, mask) if m]
            bucket_conf = [c for c, m in zip(confidences, mask) if m]
            acc = accuracy_score(bucket_true, bucket_pred)
            buckets.append({
                "range":       f"{low:.0%}–{high:.0%}",
                "count":       len(bucket_true),
                "avg_conf":    round(float(np.mean(bucket_conf)), 3),
                "accuracy":    round(acc, 3),
                "calibration_gap": round(abs(float(np.mean(bucket_conf)) - acc), 3),
            })
        return buckets

    def print_report(self, report: Dict):
        """Pretty-print evaluation results to stdout."""
        print("\n" + "=" * 60)
        print("  EVALUATION RESULTS — Hinglish Sentiment Engine")
        print("=" * 60)
        print(f"  Samples:    {report['total_samples']}")
        print(f"  Accuracy:   {report['accuracy']:.1%}")
        print(f"  Macro F1:   {report['macro_f1']:.4f}")
        print(f"  Weighted F1:{report['weighted_f1']:.4f}")
        print(f"  Error rate: {report['error_rate']:.1%}")
        print()
        print(f"  {'Label':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
        print(f"  {'-'*48}")
        for label in LABEL_NAMES:
            pc = report["per_class"][label]
            print(f"  {label:<12} {pc['precision']:>10.4f} {pc['recall']:>8.4f} {pc['f1']:>8.4f} {pc['support']:>9}")

        print()
        print("  Confusion Matrix (rows=true, cols=pred):")
        print(f"  {'':12}", end="")
        for label in LABEL_NAMES:
            print(f"  {label[:7]:>7}", end="")
        print()
        for i, row in enumerate(report["confusion_matrix_normalized"]):
            print(f"  {LABEL_NAMES[i]:<12}", end="")
            for val in row:
                print(f"  {val:>7.3f}", end="")
            print()

        print()
        print("  Calibration:")
        for bucket in report["calibration"]:
            gap_indicator = "✓" if bucket["calibration_gap"] < 0.05 else "⚠"
            print(f"  {gap_indicator} {bucket['range']:12} acc={bucket['accuracy']:.3f}  "
                  f"avg_conf={bucket['avg_conf']:.3f}  n={bucket['count']}")

        print()
        print("  Top error examples (highest-confidence mistakes):")
        for err in report["hard_errors"][:5]:
            print(f"  ✗ True:{err['true_label']:<9} Pred:{err['pred_label']:<9} "
                  f"Conf:{err['confidence']:.3f} | {err['text'][:60]}")
        print("=" * 60 + "\n")

    def save_report(self, report: Dict, path: str):
        """Save full report to JSON (without large prediction list for readability)."""
        slim = {k: v for k, v in report.items() if k != "all_predictions"}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(slim, f, indent=2, ensure_ascii=False)
        print(f"Report saved to: {path}")

        # Save predictions as CSV
        csv_path = path.replace(".json", "_predictions.csv")
        preds_df = pd.DataFrame(report["all_predictions"])
        preds_df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"Predictions saved to: {csv_path}")


def compare_models(model_paths: List[str], test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare multiple model checkpoints on the same test set.
    Useful for ablation studies.

    Returns a DataFrame with rows=models, cols=metrics.
    """
    from inference.predictor import HinglishSentimentPredictor

    rows = []
    for path in model_paths:
        print(f"\n--- Evaluating: {path} ---")
        predictor = HinglishSentimentPredictor(path)
        evaluator = HinglishEvaluator(predictor)
        report    = evaluator.evaluate(test_df)

        row = {"model": os.path.basename(path)}
        row["accuracy"]     = report["accuracy"]
        row["macro_f1"]     = report["macro_f1"]
        row["weighted_f1"]  = report["weighted_f1"]
        for label in LABEL_NAMES:
            row[f"f1_{label.lower()}"] = report["per_class"][label]["f1"]
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    print("\n\n=== MODEL COMPARISON ===")
    print(comparison_df.to_string(index=False))
    return comparison_df


if __name__ == "__main__":
    print("Evaluator module loaded. Import HinglishEvaluator and pass a predictor + test DataFrame.")