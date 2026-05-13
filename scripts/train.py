"""
scripts/train.py

Fine-tune a multilingual BERT model for Hinglish sentiment analysis.

Usage:
  # Default (HingMBERT, 5 epochs)
  python scripts/train.py

  # Custom model + hyperparameters
  python scripts/train.py --model hing-mbert --epochs 5 --lr 2e-5 --batch 16

  # Disable W&B logging
  WANDB_DISABLED=true python scripts/train.py

  # Enable fp16 for faster training on GPU
  python scripts/train.py --fp16
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import TrainingConfig, SUPPORTED_MODELS, Paths
from src.model.trainer import train
from src.preprocessing.dataset_builder import create_demo_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train Hinglish Sentiment Engine")
    parser.add_argument(
        "--model", default="hing-mbert",
        choices=list(SUPPORTED_MODELS.keys()),
        help="Model backbone to fine-tune (default: hing-mbert)"
    )
    parser.add_argument("--epochs",    type=int,   default=5,    help="Training epochs")
    parser.add_argument("--lr",        type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch",     type=int,   default=16,   help="Batch size per device")
    parser.add_argument("--max_len",   type=int,   default=128,  help="Max token length")
    parser.add_argument("--fp16",      action="store_true",      help="Use fp16 mixed precision")
    parser.add_argument("--bf16",      action="store_true",      help="Use bf16 mixed precision")
    parser.add_argument("--push_hub",  action="store_true",      help="Push model to HF Hub")
    parser.add_argument("--hub_id",    default="",               help="HuggingFace Hub model ID")
    parser.add_argument("--demo",      action="store_true",      help="Quick demo with toy data")
    parser.add_argument("--no_wandb",  action="store_true",      help="Disable W&B logging")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"

    print("\n=== Hinglish Sentiment Engine — Training ===")
    print(f"  Model:   {args.model} ({SUPPORTED_MODELS[args.model]})")
    print(f"  Epochs:  {args.epochs}")
    print(f"  LR:      {args.lr}")
    print(f"  Batch:   {args.batch}")
    print(f"  fp16:    {args.fp16}")
    print()

    # Build config
    cfg = TrainingConfig(
        model_name=SUPPORTED_MODELS[args.model],
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch,
        max_length=args.max_len,
        fp16=args.fp16,
        bf16=args.bf16,
        push_to_hub=args.push_hub,
        hub_model_id=args.hub_id or f"your-username/hinglish-sentiment-{args.model}",
        report_to="none" if args.no_wandb else "wandb",
    )

    # Dataset
    if args.demo:
        print("Using toy demo dataset (--demo flag)...")
        dataset_dict = create_demo_dataset()
        class_weights = None
    else:
        dataset_dict = None  # Will load from disk (data/processed/dataset)

        # Load class weights if precomputed
        weights_path = os.path.join(Paths.DATA_PROC, "class_weights.json")
        if os.path.exists(weights_path):
            with open(weights_path) as f:
                class_weights = json.load(f)["weights"]
            print(f"Using class weights: {[round(w, 3) for w in class_weights]}")
        else:
            class_weights = None
            print("No class weights found — using uniform loss.")

    # Train!
    best_model_path = train(
        config=cfg,
        dataset_dict=dataset_dict,
        class_weights=class_weights,
    )

    print(f"\n✓ Training complete. Best model: {best_model_path}")
    print("\nNext steps:")
    print("  python scripts/evaluate.py")
    print("  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()