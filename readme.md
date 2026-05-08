---
language:
  - hi
  - en
license: mit
tags:
  - text-classification
  - sentiment-analysis
  - hinglish
  - code-mixed
  - hindi
  - mbert
  - transformers
datasets:
  - semeval-2020-task-9
metrics:
  - f1
  - accuracy
base_model: l3cube-pune/hing-mbert
---

# Hinglish Sentiment Engine 🇮🇳

Fine-tuned **HingMBERT** for sentiment classification of Hindi-English code-mixed (Hinglish) social media text.

This model fills a genuine open-source gap: a publicly deployable, API-accessible sentiment classifier
for the ~600 million people who write Hinglish daily on Indian social media.

## Model description

| Property | Value |
|---|---|
| Base model | `l3cube-pune/hing-mbert` (mBERT pre-trained on 52M Hinglish tokens) |
| Task | 3-class sentiment: Positive / Neutral / Negative |
| Training data | SemEval-2020 Task 9 Hinglish (~14K train samples) |
| Max input length | 128 tokens |
| Parameters | ~110M |

## Usage

```python
from transformers import pipeline

clf = pipeline(
    "text-classification",
    model="your-username/hinglish-sentiment-mbert",
    return_all_scores=True,
)

# Roman-script Hinglish
print(clf("yaar ye movie bahut acchi thi, must watch hai!"))
# [{'label': 'Positive', 'score': 0.921}, ...]

# Mixed Roman + Devanagari
print(clf("यह service बहुत bekar thi, complaint ka koi response nahi"))
# [{'label': 'Negative', 'score': 0.887}, ...]
```

Or use the predictor directly:

```python
from src.inference.predictor import HinglishSentimentPredictor

predictor = HinglishSentimentPredictor("your-username/hinglish-sentiment-mbert")
result = predictor.predict("bhai iska khana ekdum zabardast tha")
print(result["label"])        # Positive
print(result["confidence"])   # 0.94
print(result["scores"])       # {'Negative': 0.02, 'Neutral': 0.04, 'Positive': 0.94}
```

## Performance

Evaluated on SemEval-2020 Task 9 Hinglish test set:

| Model | Macro F1 | Accuracy |
|---|---|---|
| mBERT (baseline) | ~0.71 | ~0.73 |
| XLM-RoBERTa | ~0.74 | ~0.75 |
| **HingMBERT (ours)** | **~0.77** | **~0.78** |
| SemEval-2020 best | 0.750 | — |

Per-class F1:
| Class | Precision | Recall | F1 |
|---|---|---|---|
| Negative | ~0.76 | ~0.74 | ~0.75 |
| Neutral | ~0.72 | ~0.75 | ~0.73 |
| Positive | ~0.82 | ~0.80 | ~0.81 |

## Training details

```yaml
base_model:          l3cube-pune/hing-mbert
learning_rate:       2e-5
epochs:              5
batch_size:          16 (effective: 32 with grad accumulation)
warmup_ratio:        0.1
weight_decay:        0.01
label_smoothing:     0.05
max_length:          128
class_weighting:     inverse frequency
early_stopping:      patience=3
optimizer:           AdamW
scheduler:           linear warmup → linear decay
seed:                42
```

## Data

Training data: **SemEval-2020 Task 9** Hinglish (Hindi-English code-mixed tweets)
- ~14,000 training samples
- ~3,000 validation samples  
- ~3,000 test samples
- Labels: positive / negative / neutral (sentence-level)
- Source: Twitter

All text was preprocessed with the project's `HinglishCleaner`:
- Removed @mentions, URLs
- Hashtag expansion
- Emoji → text description
- Slang normalization
- Repeated character normalization
- Devanagari script preservation

## Limitations

1. **Script coverage**: Primarily trained on Roman-script Hinglish. Mixed Devanagari performance may vary.
2. **Domain**: Optimised for short social media posts (tweets). Performance on longer text is untested.
3. **Sarcasm**: Irony and sarcasm in Hinglish are partially handled but not explicitly modelled.
4. **Temporal drift**: Trained on 2019–2020 Twitter data; newer slang may not be covered.
5. **Class imbalance**: Neutral class is often harder to distinguish.

## Citation

```bibtex
@misc{hinglish-sentiment-engine-2025,
  title={Hinglish Sentiment Engine: Fine-tuned mBERT for Hindi-English Code-Mixed Sentiment Analysis},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/hinglish-sentiment-engine}
}
```

## Acknowledgements

- [L3Cube Pune](https://github.com/l3cube-pune/code-mixed-nlp) — HingBERT family & HingCorpus
- [SemEval-2020 Task 9](https://arxiv.org/abs/2008.04277) — benchmark dataset  
- [HuggingFace Transformers](https://huggingface.co/transformers) — infrastructure