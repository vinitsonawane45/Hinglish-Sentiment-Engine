"""
src/model/trainer.py

Fine-tunes a multilingual BERT-family model for Hinglish sentiment classification.

Architecture:
    [mBERT / HingMBERT / XLM-RoBERTa]
              ↓
    CLS token hidden state (768-dim)
              ↓
    Dropout(0.1)
              ↓
    Linear(768 → 3)
              ↓
    Softmax → {Negative, Neutral, Positive}

Key features:
  - Class-weighted loss for imbalanced labels
  - W&B integration for experiment tracking
  - HuggingFace Trainer with early stopping
  - Auto-push to HF Hub after training
"""

import os
import sys
import json
import numpy as np
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    set_seed,
)
from datasets import DatasetDict, load_from_disk
import evaluate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TrainingConfig, LABEL2ID, ID2LABEL, NUM_LABELS, Paths


# ─── Custom Trainer with class-weighted loss ──────────────────────────────────

class WeightedLossTrainer(Trainer):
    """
    Extends HuggingFace Trainer to use class-weighted CrossEntropyLoss.
    Critical for Hinglish datasets where Neutral >> Negative/Positive.
    """

    def __init__(self, class_weights: Optional[list] = None, **kwargs):
        super().__init__(**kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fn = CrossEntropyLoss(weight=weights, label_smoothing=0.05)
        else:
            loss_fn = CrossEntropyLoss(label_smoothing=0.05)

        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ─── Metrics ──────────────────────────────────────────────────────────────────

def build_compute_metrics():
    """
    Returns a compute_metrics function for the Trainer.
    Reports: accuracy, macro F1, per-class F1.
    """
    accuracy_metric = evaluate.load("accuracy")
    f1_metric       = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        acc = accuracy_metric.compute(predictions=predictions, references=labels)
        macro_f1 = f1_metric.compute(
            predictions=predictions, references=labels, average="macro"
        )
        per_class_f1 = f1_metric.compute(
            predictions=predictions, references=labels, average=None
        )

        result = {
            "eval_accuracy":  round(acc["accuracy"], 4),
            "eval_macro_f1":  round(macro_f1["f1"], 4),
            "eval_f1_negative": round(per_class_f1["f1"][0], 4),
            "eval_f1_neutral":  round(per_class_f1["f1"][1], 4),
            "eval_f1_positive": round(per_class_f1["f1"][2], 4),
        }
        return result

    return compute_metrics


# ─── Tokenization ─────────────────────────────────────────────────────────────

def tokenize_dataset(dataset_dict: DatasetDict, tokenizer, max_length: int = 128) -> DatasetDict:
    """
    Tokenize the full DatasetDict. Returns tokenized version with input_ids,
    attention_mask, and labels columns.
    """
    def tokenize_fn(batch):
        tokens = tokenizer(
            batch["text"],
            padding=False,         # Dynamic padding via DataCollatorWithPadding
            truncation=True,
            max_length=max_length,
        )
        tokens["labels"] = batch["label_id"]
        return tokens

    tokenized = dataset_dict.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text", "label", "label_id"],
        desc="Tokenizing",
    )
    tokenized.set_format("torch")
    return tokenized


# ─── Main training function ───────────────────────────────────────────────────

def train(
    config: Optional[TrainingConfig] = None,
    dataset_dict: Optional[DatasetDict] = None,
    class_weights: Optional[list] = None,
) -> str:
    """
    Full training pipeline. Returns path to the best saved checkpoint.

    Args:
        config:        TrainingConfig dataclass. Uses defaults if None.
        dataset_dict:  Pre-built DatasetDict. Loads from disk if None.
        class_weights: Class weights for loss. Computed from data if None.

    Returns:
        str: Path to best model checkpoint.
    """
    cfg = config or TrainingConfig()
    Paths.make_all()
    set_seed(cfg.seed)

    print(f"\n{'='*60}")
    print(f"  Hinglish Sentiment Engine — Training")
    print(f"  Model: {cfg.model_name}")
    print(f"  Epochs: {cfg.num_epochs} | LR: {cfg.learning_rate} | Batch: {cfg.batch_size}")
    print(f"{'='*60}\n")

    # ── Load tokenizer ──────────────────────────────────────────────────────
    print(f"Loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # ── Load or receive dataset ─────────────────────────────────────────────
    if dataset_dict is None:
        processed_path = os.path.join(Paths.DATA_PROC, "dataset")
        if os.path.exists(processed_path):
            print(f"Loading processed dataset from: {processed_path}")
            dataset_dict = load_from_disk(processed_path)
        else:
            raise FileNotFoundError(
                f"No processed dataset found at {processed_path}. "
                "Run scripts/preprocess.py first."
            )

    print(f"Dataset: {dataset_dict}")

    # ── Tokenize ─────────────────────────────────────────────────────────────
    tokenized = tokenize_dataset(dataset_dict, tokenizer, cfg.max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model: {cfg.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        hidden_dropout_prob=cfg.dropout_prob,
        attention_probs_dropout_prob=cfg.dropout_prob,
        ignore_mismatched_sizes=True,
    )

    # Freeze base layers for first epoch (optional — uncomment to enable)
    # for name, param in model.base_model.named_parameters():
    #     if "embeddings" in name:
    #         param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total | {trainable:,} trainable")

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        evaluation_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        logging_dir=Paths.LOGS,
        logging_steps=cfg.logging_steps,
        report_to=cfg.report_to,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        dataloader_num_workers=cfg.dataloader_num_workers,
        seed=cfg.seed,
        save_total_limit=3,
        push_to_hub=cfg.push_to_hub,
        hub_model_id=cfg.hub_model_id if cfg.push_to_hub else None,
    )

    # ── Build trainer ─────────────────────────────────────────────────────────
    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.001,
            )
        ],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nStarting training...\n")
    train_result = trainer.train()

    # ── Save best model ───────────────────────────────────────────────────────
    os.makedirs(Paths.BEST_MODEL, exist_ok=True)
    trainer.save_model(Paths.BEST_MODEL)
    tokenizer.save_pretrained(Paths.BEST_MODEL)

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best model saved to: {Paths.BEST_MODEL}")
    print(f"  Test Macro F1:  {test_metrics.get('test_macro_f1', 'N/A')}")
    print(f"  Test Accuracy:  {test_metrics.get('test_accuracy', 'N/A')}")
    print(f"{'='*60}\n")

    # Save config for reproducibility
    cfg_dict = {k: v for k, v in cfg.__dict__.items()}
    with open(os.path.join(Paths.BEST_MODEL, "training_config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2)

    # Push to hub
    if cfg.push_to_hub:
        print(f"Pushing model to HuggingFace Hub: {cfg.hub_model_id}")
        trainer.push_to_hub()

    return Paths.BEST_MODEL


if __name__ == "__main__":
    # Quick demo with toy data
    from preprocessing.dataset_builder import create_demo_dataset

    print("Running training demo with toy dataset...")
    demo_cfg = TrainingConfig(
        model_name="google-bert/bert-base-multilingual-cased",
        num_epochs=2,
        batch_size=4,
        eval_steps=10,
        save_steps=10,
        logging_steps=5,
        report_to="none",
    )

    demo_data = create_demo_dataset()
    best_path = train(config=demo_cfg, dataset_dict=demo_data)
    print(f"Demo training done. Model at: {best_path}")