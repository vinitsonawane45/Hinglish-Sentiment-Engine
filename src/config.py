"""
Central configuration for the Hinglish Sentiment Engine.
All hyperparameters, paths, and constants live here.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


# ─── Label mapping ────────────────────────────────────────────────────────────

LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = 3

# SemEval-2020 uses these raw label strings
SEMEVAL_LABEL_MAP = {
    "negative": "Negative",
    "neutral":  "Neutral",
    "positive": "Positive",
    # handle varied casing
    "Negative": "Negative",
    "Neutral":  "Neutral",
    "Positive": "Positive",
}

# ─── Supported model backbones ────────────────────────────────────────────────

SUPPORTED_MODELS = {
    # Best choice: already pre-trained on 52M Hinglish tokens
    "hing-mbert":    "l3cube-pune/hing-mbert",
    # Strong multilingual baseline
    "xlm-roberta":   "FacebookAI/xlm-roberta-base",
    # Vanilla multilingual BERT (your true baseline)
    "mbert":         "google-bert/bert-base-multilingual-cased",
    # Google's Indian-language specialist
    "muril":         "google/muril-base-cased",
    # HingRoBERTa — another L3Cube option
    "hing-roberta":  "l3cube-pune/hing-roberta",
}

DEFAULT_MODEL = "hing-mbert"

# ─── Training hyperparameters ─────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    # Model
    model_name: str = SUPPORTED_MODELS[DEFAULT_MODEL]
    num_labels: int = NUM_LABELS

    # Data
    max_length: int = 128          # Most tweets < 50 tokens, 128 is safe ceiling
    train_file: str = "data/processed/train.csv"
    val_file: str   = "data/processed/val.csv"
    test_file: str  = "data/processed/test.csv"

    # Training
    output_dir: str = "outputs/checkpoints"
    num_epochs: int = 5
    batch_size: int = 16           # Use 8 if OOM on free Colab
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1      # 10% of steps for warmup
    gradient_accumulation_steps: int = 2  # effective batch = 32
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "linear"

    # Regularisation
    dropout_prob: float = 0.1
    label_smoothing: float = 0.05  # Helps with noisy social media labels

    # Evaluation
    metric_for_best_model: str = "eval_macro_f1"
    greater_is_better: bool = True
    eval_steps: int = 100
    save_steps: int = 100
    load_best_model_at_end: bool = True

    # Logging
    logging_steps: int = 50
    report_to: str = "wandb"       # Set to "none" to disable W&B

    # Hardware
    fp16: bool = False             # Set True if GPU supports it (speeds up 2x)
    bf16: bool = False             # Prefer bf16 on A100/H100
    dataloader_num_workers: int = 4
    seed: int = 42

    # HuggingFace Hub push
    push_to_hub: bool = False
    hub_model_id: str = "your-username/hinglish-sentiment-mbert"


@dataclass
class InferenceConfig:
    model_path: str = "outputs/best_model"
    max_length: int = 128
    batch_size: int = 32
    device: str = "auto"           # "auto" | "cpu" | "cuda" | "mps"


# ─── Paths ────────────────────────────────────────────────────────────────────

class Paths:
    ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_RAW     = os.path.join(ROOT, "data", "raw")
    DATA_PROC    = os.path.join(ROOT, "data", "processed")
    DATA_ANN     = os.path.join(ROOT, "data", "annotated")
    OUTPUTS      = os.path.join(ROOT, "outputs")
    CHECKPOINTS  = os.path.join(OUTPUTS, "checkpoints")
    BEST_MODEL   = os.path.join(OUTPUTS, "best_model")
    LOGS         = os.path.join(OUTPUTS, "logs")

    # Raw data files (after download)
    SEMEVAL_TRAIN = os.path.join(DATA_RAW, "semeval_train.csv")
    SEMEVAL_VAL   = os.path.join(DATA_RAW, "semeval_val.csv")
    SEMEVAL_TEST  = os.path.join(DATA_RAW, "semeval_test.csv")

    @classmethod
    def make_all(cls):
        """Create all directories if they don't exist."""
        for attr in dir(cls):
            val = getattr(cls, attr)
            if isinstance(val, str) and not attr.startswith("_"):
                # Only create if it looks like a directory (no file extension)
                if "." not in os.path.basename(val):
                    os.makedirs(val, exist_ok=True)