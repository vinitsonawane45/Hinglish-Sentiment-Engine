"""
scripts/evaluate.py

Evaluate a trained Hinglish sentiment model on the test set.

Usage:
  python scripts/evaluate.py
  python scripts/evaluate.py --model_path outputs/best_model
  python scripts/evaluate.py --compare outputs/checkpoint-200 outputs/checkpoint-400
"""

import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Paths
from src.inference.predictor import HinglishSentimentPredictor
from src.evaluation.evaluator import HinglishEvaluator, compare_models


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Hinglish Sentiment Model")
    parser.add_argument("--model_path", default=Paths.BEST_MODEL, help="Path to trained model")
    parser.add_argument("--test_file",  default=os.path.join(Paths.DATA_PROC, "test.csv"))
    parser.add_argument("--output",     default=os.path.join(Paths.OUTPUTS, "eval_report.json"))
    parser.add_argument("--compare",    nargs="+", help="Paths to multiple models to compare")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load test data
    if not os.path.exists(args.test_file):
        print(f"Test file not found: {args.test_file}")
        print("Run: python scripts/preprocess.py first.")
        sys.exit(1)

    test_df = pd.read_csv(args.test_file)
    print(f"Test set: {len(test_df)} samples")

    if args.compare:
        # Multi-model comparison
        print(f"\nComparing {len(args.compare)} models...")
        comparison = compare_models(args.compare, test_df)
        csv_path = os.path.join(Paths.OUTPUTS, "model_comparison.csv")
        comparison.to_csv(csv_path, index=False)
        print(f"\nComparison saved to: {csv_path}")
    else:
        # Single model evaluation
        if not os.path.exists(args.model_path):
            print(f"Model not found: {args.model_path}")
            print("Run: python scripts/train.py first.")
            sys.exit(1)

        predictor = HinglishSentimentPredictor(args.model_path)
        evaluator = HinglishEvaluator(predictor)
        report    = evaluator.evaluate(test_df)

        evaluator.print_report(report)
        evaluator.save_report(report, args.output)


if __name__ == "__main__":
    main()