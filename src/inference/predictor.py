"""
src/inference/predictor.py

Production-ready inference pipeline for Hinglish sentiment prediction.

Features:
  - Single text & batch prediction
  - Confidence scores for all 3 classes
  - Attention weight extraction (for visualization)
  - Auto device selection (CUDA / MPS / CPU)
  - Caching for repeated predictions
"""

import os
import sys
import time
from typing import Union, List, Dict, Optional, Tuple
from functools import lru_cache

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ID2LABEL, LABEL2ID, InferenceConfig, Paths
from preprocessing.cleaner import HinglishCleaner, detect_script_ratio


def get_device(preference: str = "auto") -> torch.device:
    """Select best available device."""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


# ─── Main predictor class ─────────────────────────────────────────────────────

class HinglishSentimentPredictor:
    """
    Load a fine-tuned model and run inference on Hinglish text.

    Usage:
        predictor = HinglishSentimentPredictor("outputs/best_model")

        # Single prediction
        result = predictor.predict("yaar ye movie bahut acchi thi!")
        print(result)
        # {
        #   "text": "yaar ye movie bahut acchi thi!",
        #   "cleaned_text": "yaar ye movie bahut acchi thi",
        #   "label": "Positive",
        #   "confidence": 0.921,
        #   "scores": {"Negative": 0.031, "Neutral": 0.048, "Positive": 0.921},
        #   "script_ratio": {"devanagari": 0.0, "latin": 0.87, "other": 0.13},
        #   "inference_ms": 23.4
        # }

        # Batch prediction
        results = predictor.predict_batch(["text1", "text2", ...])
    """

    def __init__(
        self,
        model_path: str = None,
        config: Optional[InferenceConfig] = None,
        clean_input: bool = True,
    ):
        cfg = config or InferenceConfig()
        self.model_path  = model_path or cfg.model_path
        self.max_length  = cfg.max_length
        self.batch_size  = cfg.batch_size
        self.device      = get_device(cfg.device)
        self.clean_input = clean_input
        self.cleaner     = HinglishCleaner()

        print(f"Loading model from: {self.model_path}")
        print(f"Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path
        )
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded. Labels: {ID2LABEL}")

    def _prepare_text(self, text: str) -> Tuple[str, str]:
        """Returns (original, cleaned)."""
        cleaned = self.cleaner.clean(text) if self.clean_input else text
        if not cleaned:
            cleaned = text  # Fallback to original if cleaning wipes it
        return text, cleaned

    @torch.no_grad()
    def predict(self, text: str, return_attention: bool = False) -> Dict:
        """
        Predict sentiment for a single Hinglish text.

        Args:
            text: Raw input text (Hinglish, any script).
            return_attention: If True, also return attention weights for visualization.

        Returns:
            dict with keys: text, cleaned_text, label, confidence, scores,
                           script_ratio, inference_ms, [attention_weights]
        """
        t0 = time.perf_counter()
        original, cleaned = self._prepare_text(text)

        inputs = self.tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(
            **inputs,
            output_attentions=return_attention,
        )

        logits = outputs.logits[0]
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_id = int(np.argmax(probs))

        result = {
            "text":         original,
            "cleaned_text": cleaned,
            "label":        ID2LABEL[pred_id],
            "confidence":   round(float(probs[pred_id]), 4),
            "scores": {
                ID2LABEL[i]: round(float(p), 4)
                for i, p in enumerate(probs)
            },
            "script_ratio":  detect_script_ratio(cleaned),
            "inference_ms":  round((time.perf_counter() - t0) * 1000, 1),
        }

        if return_attention and outputs.attentions is not None:
            # Return attention from last layer, averaged across heads
            # Shape: (num_layers, batch, heads, seq_len, seq_len)
            last_layer_attn = outputs.attentions[-1][0]  # (heads, seq, seq)
            avg_attn = last_layer_attn.mean(dim=0).cpu().numpy()  # (seq, seq)
            tokens = self.tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0].cpu().tolist()
            )
            result["attention_weights"] = avg_attn.tolist()
            result["tokens"] = tokens

        return result

    @torch.no_grad()
    def predict_batch(self, texts: List[str], show_progress: bool = True) -> List[Dict]:
        """
        Predict sentiment for a list of texts using batched inference.
        Much faster than calling predict() in a loop.
        """
        from tqdm import tqdm

        results = []
        originals = texts
        cleaned_texts = [self._prepare_text(t)[1] for t in texts]

        batches = range(0, len(cleaned_texts), self.batch_size)
        if show_progress:
            batches = tqdm(batches, desc="Predicting", unit="batch")

        for start in batches:
            batch_texts    = cleaned_texts[start : start + self.batch_size]
            batch_originals = originals[start : start + self.batch_size]

            t0 = time.perf_counter()
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            logits = outputs.logits.cpu().numpy()
            probs  = torch.softmax(torch.tensor(logits), dim=-1).numpy()
            pred_ids = np.argmax(probs, axis=-1)

            for i, (orig, cleaned, pred_id, prob_row) in enumerate(
                zip(batch_originals, batch_texts, pred_ids, probs)
            ):
                results.append({
                    "text":         orig,
                    "cleaned_text": cleaned,
                    "label":        ID2LABEL[int(pred_id)],
                    "confidence":   round(float(prob_row[pred_id]), 4),
                    "scores": {
                        ID2LABEL[j]: round(float(p), 4)
                        for j, p in enumerate(prob_row)
                    },
                    "script_ratio":  detect_script_ratio(cleaned),
                    "inference_ms":  round(elapsed_ms / len(batch_texts), 1),
                })

        return results

    def predict_df(self, df, text_col: str = "text"):
        """
        Add prediction columns to a pandas DataFrame in-place.
        Returns the DataFrame with added columns: label, confidence, score_neg, score_neu, score_pos.
        """
        import pandas as pd
        results = self.predict_batch(df[text_col].tolist())
        df = df.copy()
        df["predicted_label"]    = [r["label"] for r in results]
        df["confidence"]         = [r["confidence"] for r in results]
        df["score_negative"]     = [r["scores"]["Negative"] for r in results]
        df["score_neutral"]      = [r["scores"]["Neutral"] for r in results]
        df["score_positive"]     = [r["scores"]["Positive"] for r in results]
        return df


# ─── Lightweight pipeline wrapper (for HuggingFace Spaces) ───────────────────

def load_pipeline(model_path: str, device: int = -1):
    """
    Wrap the model in a HuggingFace pipeline for ultra-simple usage.
    device=-1 means CPU.
    """
    return pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        device=device,
        return_all_scores=True,
    )


# ─── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Demo with a placeholder (use your trained model path)
    MODEL_PATH = Paths.BEST_MODEL

    if not os.path.exists(MODEL_PATH):
        print(f"No model found at {MODEL_PATH}.")
        print("Run scripts/train.py first, OR use the demo below with a public model.")
        # Fallback: use raw HingMBERT (without sentiment fine-tuning)
        MODEL_PATH = "l3cube-pune/hing-mbert"

    predictor = HinglishSentimentPredictor(MODEL_PATH)

    test_texts = [
        "yaar ye movie bahut acchi thi, must watch hai!",
        "nhi yar, product bilkul bakwas hai, waste of money",
        "aaj ka match theek tha, na bahut acha na bura",
        "यह service बहुत bekar thi, complaint ka koi response nahi",
        "@someone bhai iska khana ekdum zabardast tha 🔥",
        "sooo boring this movie was, walked out halfway lmao",
    ]

    print("\n" + "=" * 70)
    print("  Hinglish Sentiment Predictions")
    print("=" * 70)

    for text in test_texts:
        result = predictor.predict(text)
        label  = result["label"]
        conf   = result["confidence"]
        emoji  = {"Positive": "😊", "Negative": "😠", "Neutral": "😐"}[label]
        print(f"\n{emoji} [{label:8s}] {conf:.1%} | {text}")
        print(f"   Scores: Neg={result['scores']['Negative']:.3f}  "
              f"Neu={result['scores']['Neutral']:.3f}  "
              f"Pos={result['scores']['Positive']:.3f}  "
              f"({result['inference_ms']}ms)")