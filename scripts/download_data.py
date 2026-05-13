"""
scripts/download_data.py

Downloads SemEval-2020 Task 9 Hinglish sentiment dataset.

The official data lives on GitHub (Hinglish_SentMix_3K / SentiMix).
We also provide instructions for the L3Cube corpus.

Run: python scripts/download_data.py
"""

import os
import sys
import json
import requests
import zipfile
import io
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Paths

# ─── SemEval-2020 Task 9 download URLs ────────────────────────────────────────
# These are the publicly released splits from the competition.

SEMEVAL_SOURCES = {
    # Hindi-English (Hinglish) splits from SemEval 2020 Task 9
    "train": "https://raw.githubusercontent.com/keshav22bansal/BAKSA_IITK/master/Data/Hinglish/Hinglish_train.txt",
    "val":   "https://raw.githubusercontent.com/keshav22bansal/BAKSA_IITK/master/Data/Hinglish/Hinglish_dev.txt",
    "test":  "https://raw.githubusercontent.com/keshav22bansal/BAKSA_IITK/master/Data/Hinglish/Hinglish_test.txt",
}

# Official CodaLab dataset (requires registration):
# https://competitions.codalab.org/competitions/20654
# Download and place files as: data/raw/semeval_train.tsv etc.


def download_file(url: str, dest_path: str, timeout: int = 30) -> bool:
    """Download a single file with progress indication."""
    print(f"  Downloading: {os.path.basename(dest_path)}")
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(resp.content)
        size_kb = len(resp.content) / 1024
        print(f"  ✓ Saved ({size_kb:.1f} KB): {dest_path}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def download_semeval():
    """Download SemEval-2020 Hinglish splits."""
    print("\n=== Downloading SemEval-2020 Task 9 Hinglish Data ===")
    Paths.make_all()

    dest_map = {
        "train": Paths.SEMEVAL_TRAIN,
        "val":   Paths.SEMEVAL_VAL,
        "test":  Paths.SEMEVAL_TEST,
    }

    success = 0
    for split, url in SEMEVAL_SOURCES.items():
        dest = dest_map[split]
        if os.path.exists(dest):
            print(f"  Already exists: {dest} (skipping)")
            success += 1
            continue
        if download_file(url, dest):
            success += 1

    if success < len(SEMEVAL_SOURCES):
        print("\n⚠️  Some downloads failed. Manual download instructions:")
        print("  1. Register at: https://competitions.codalab.org/competitions/20654")
        print("  2. Download the Hinglish dataset ZIP")
        print("  3. Place the TSV files as:")
        print(f"     - data/raw/semeval_train.csv")
        print(f"     - data/raw/semeval_val.csv")
        print(f"     - data/raw/semeval_test.csv")
        print("\n  Each file should have columns: uid, text, label (tab-separated)")
        print("  Labels: positive | negative | neutral")
        _create_sample_data()
    else:
        print(f"\n✓ All {success} SemEval splits downloaded successfully.")


def _create_sample_data():
    """
    Create sample data files so the pipeline works even without real data.
    Replace with real SemEval data for production training.
    """
    print("\n=== Creating sample data for pipeline testing ===")

    sample_train = [
        ("1", "yaar ye movie bahut acchi thi, must watch hai", "positive"),
        ("2", "bhai iska khana ekdum zabardast tha", "positive"),
        ("3", "aaj ka match amazing tha team ne kamaal kar diya", "positive"),
        ("4", "product bahut acha hai packaging bhi nice thi", "positive"),
        ("5", "so happy aaj sab kuch perfect ho gaya finally", "positive"),
        ("6", "superb performance team proud kar diya humein", "positive"),
        ("7", "ekdum mast tha bhai would recommend to everyone", "positive"),
        ("8", "yaar ye product bilkul bakwas hai waste of money", "negative"),
        ("9", "ye movie boring thi walkout karna chahiye tha", "negative"),
        ("10", "khana bahut bekar tha never going back there", "negative"),
        ("11", "delivery bahut late tha product bhi kharab nikla", "negative"),
        ("12", "ye company fraud hai bilkul cheating karti hai", "negative"),
        ("13", "movie mein kuch bhi interesting nahi time waste", "negative"),
        ("14", "disappointed hu expectations se bilkul alag tha", "negative"),
        ("15", "aaj movie dekhi thi kaafi theek thi", "neutral"),
        ("16", "product mile packaging ok thi delivery on time", "neutral"),
        ("17", "ye match tie ho gaya kuch khas nahi hua", "neutral"),
        ("18", "restaurant mein gaye the average khana tha", "neutral"),
        ("19", "abhi decide nahi kiya ki kaisa laga", "neutral"),
        ("20", "theek hai bhai na bahut acha na bahut bura", "neutral"),
        ("21", "yaar ye song bahut catchy hai repeat sun raha hu", "positive"),
        ("22", "bhai iphone ka camera wakai kamaal hai", "positive"),
        ("23", "ye app bilkul useless hai crash hoti rehti hai", "negative"),
        ("24", "customer service ne koi help nahi ki bekar hai", "negative"),
        ("25", "normal experience tha kuch special nahi", "neutral"),
        ("26", "sab log bol rahe the accha hai dekh ke aaunga", "neutral"),
        ("27", "यह फिल्म बहुत अच्छी थी loved every moment", "positive"),
        ("28", "बहुत bekar service complaint ka koi jawab nahi", "negative"),
        ("29", "सुना है नई movie आ rahi hai next week", "neutral"),
        ("30", "yaar concert mein gaya tha ekdum fire tha", "positive"),
        ("31", "itna bura experience pehle kabhi nahi hua tha", "negative"),
        ("32", "average laga mujhe kuch alag nahi tha usme", "neutral"),
    ]

    val_data = sample_train[:8]
    test_data = sample_train[8:16]
    train_data = sample_train[16:]

    def write_tsv(data, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for uid, text, label in data:
                f.write(f"{uid}\t{text}\t{label}\n")
        print(f"  ✓ Written: {path} ({len(data)} samples)")

    write_tsv(train_data, Paths.SEMEVAL_TRAIN)
    write_tsv(val_data, Paths.SEMEVAL_VAL)
    write_tsv(test_data, Paths.SEMEVAL_TEST)

    print("\n⚠️  NOTE: These are sample files for testing the pipeline only.")
    print("  For real model training, download the actual SemEval-2020 dataset.")
    print("  Real dataset: 14,000 train + 3,000 val + 3,000 test samples.\n")


def print_dataset_stats():
    """Print stats for all downloaded raw files."""
    print("\n=== Dataset Statistics ===")
    for split, path in [
        ("train", Paths.SEMEVAL_TRAIN),
        ("val",   Paths.SEMEVAL_VAL),
        ("test",  Paths.SEMEVAL_TEST),
    ]:
        if not os.path.exists(path):
            print(f"  {split}: NOT FOUND")
            continue

        with open(path, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        label_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for line in lines:
            parts = line.split("\t")
            if len(parts) >= 3:
                label = parts[-1].lower()
                if label in label_counts:
                    label_counts[label] += 1

        print(f"  {split}: {len(lines)} samples | "
              f"pos={label_counts['positive']} "
              f"neg={label_counts['negative']} "
              f"neu={label_counts['neutral']}")


if __name__ == "__main__":
    download_semeval()
    print_dataset_stats()
    print("\nNext step: python scripts/preprocess.py")