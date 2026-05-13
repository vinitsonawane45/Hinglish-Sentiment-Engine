"""
src/preprocessing/dataset_builder.py

Loads raw SemEval-2020 data, applies cleaning, creates train/val/test splits,
and saves HuggingFace-compatible datasets to disk.

SemEval-2020 Task 9 Hinglish format (TSV):
    uid  \t  text  \t  label

Labels: positive | negative | neutral
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, ClassLabel, Features, Value
from typing import Optional, Tuple

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LABEL2ID, ID2LABEL, SEMEVAL_LABEL_MAP, Paths
from preprocessing.cleaner import HinglishCleaner


# ─── Loaders for each raw data source ─────────────────────────────────────────

# def load_semeval_tsv(filepath: str) -> pd.DataFrame:
#     """
#     Load SemEval-2020 Task 9 Hinglish TSV file.
#     Columns: uid, text, label
#     """
#     df = pd.read_csv(
#         filepath,
#         sep="\t",
#         header=None,
#         names=["uid", "text", "label"],
#         encoding="utf-8",
#         on_bad_lines="skip",
#     )
#     df = df.dropna(subset=["text", "label"])
#     df["label"] = df["label"].str.strip().map(SEMEVAL_LABEL_MAP)
#     df = df.dropna(subset=["label"])  # Drop rows with unknown labels
#     # print(pd.read_csv(filepath).head())
#     # print(open(filepath, "r", encoding="utf-8").readlines()[:5])
#     return df[["text", "label"]].reset_index(drop=True)
def load_semeval_tsv(filepath: str) -> pd.DataFrame:
    """
    Load Hinglish sentiment dataset.
    Supports both:
    - Real SemEval TSV files
    - Locally generated CSV sample files
    """

    # Try CSV first (your sample files)
    try:
        df = pd.read_csv(
            filepath,
            encoding="utf-8",
        )

        # If proper columns exist, use them
        if {"text", "label"}.issubset(df.columns):

            df = df.dropna(subset=["text", "label"])

            df["label"] = (
                df["label"]
                .astype(str)
                .str.strip()
                .map(SEMEVAL_LABEL_MAP)
            )

            df = df.dropna(subset=["label"])

            return df[["text", "label"]].reset_index(drop=True)

    except Exception:
        pass

    # Fallback to original SemEval TSV format
    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=["uid", "text", "label"],
        encoding="utf-8",
        on_bad_lines="skip",
    )

    df = df.dropna(subset=["text", "label"])

    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .map(SEMEVAL_LABEL_MAP)
    )

    df = df.dropna(subset=["label"])

    return df[["text", "label"]].reset_index(drop=True)

# def load_csv_generic(filepath: str, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
#     """
#     Load any CSV with configurable column names.
#     Useful for your own freshly-collected data.
#     """
#     df = pd.read_csv(filepath, encoding="utf-8")
#     df = df.rename(columns={text_col: "text", label_col: "label"})
#     df["label"] = df["label"].str.strip().map(
#         lambda x: SEMEVAL_LABEL_MAP.get(x, x)
#     )
#     df = df.dropna(subset=["text", "label"])
#     # Keep only valid labels
#     df = df[df["label"].isin(LABEL2ID.keys())]
#     return df[["text", "label"]].reset_index(drop=True)

def load_csv_generic(filepath: str, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """
    Load CSV/TSV sentiment datasets with flexible column handling.
    """

    # Try normal CSV first
    try:
        df = pd.read_csv(filepath, encoding="utf-8")

        # If columns are malformed because of TSV
        if len(df.columns) == 1:
            raise ValueError("Detected TSV format")

    except Exception:
        # Fallback to TSV
        df = pd.read_csv(
            filepath,
            sep="\t",
            encoding="utf-8",
            header=None,
            names=["uid", "text", "label"],
            on_bad_lines="skip",
        )

    # Rename custom columns if needed
    df = df.rename(columns={text_col: "text", label_col: "label"})

    # Ensure required columns exist
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Required columns missing. Found columns: {list(df.columns)}"
        )

    # Clean labels
    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .map(lambda x: SEMEVAL_LABEL_MAP.get(x, x))
    )

    # Remove invalid rows
    df = df.dropna(subset=["text", "label"])

    # Keep only valid labels
    df = df[df["label"].isin(LABEL2ID.keys())]

    return df[["text", "label"]].reset_index(drop=True)


# ─── Main builder ─────────────────────────────────────────────────────────────

class HinglishDatasetBuilder:
    """
    End-to-end pipeline: raw files → cleaned → split → HuggingFace DatasetDict.

    Usage:
        builder = HinglishDatasetBuilder()
        dsd = builder.build(
            train_file="data/raw/semeval_train.csv",
            test_file="data/raw/semeval_test.csv",
            extra_files=["data/annotated/my_data.csv"],
        )
        dsd.save_to_disk("data/processed/dataset")
    """

    def __init__(
        self,
        cleaner: Optional[HinglishCleaner] = None,
        val_size: float = 0.3,
        random_state: int = 42,
        min_text_length: int = 3,
        max_text_length: int = 500,
    ):
        self.cleaner = cleaner or HinglishCleaner()
        self.val_size = val_size
        self.random_state = random_state
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length

    def _load_and_clean(self, filepath: str) -> pd.DataFrame:
        ext = os.path.splitext(filepath)[1].lower()
        if ext in (".tsv",):
            df = load_semeval_tsv(filepath)
        else:
            df = load_csv_generic(filepath)

        # Clean text
        df["text"] = self.cleaner.clean_batch(df["text"].tolist())

        # Filter by length
        df = df[df["text"].str.len() >= self.min_text_length]
        df = df[df["text"].str.len() <= self.max_text_length]

        return df.reset_index(drop=True)

    def build(
        self,
        train_file: str,
        test_file: Optional[str] = None,
        extra_files: Optional[list] = None,
        save_path: Optional[str] = None,
    ) -> DatasetDict:
        """
        Build the full DatasetDict with train / validation / test splits.
        """
        # Load primary training data
        train_df = self._load_and_clean(train_file)
        print(f"Loaded train: {len(train_df)} samples")

        # Load and append any extra (your fresh-collected) data
        if extra_files:
            extra_dfs = [self._load_and_clean(f) for f in extra_files]
            train_df = pd.concat([train_df] + extra_dfs, ignore_index=True)
            print(f"After adding extra data: {len(train_df)} samples")

        # Split off validation set
        train_df, val_df = train_test_split(
            train_df,
            test_size=4,
            stratify=train_df["label"],
            random_state=self.random_state,
        )
        print(f"Train: {len(train_df)} | Val: {len(val_df)}")

        # Load test set (or carve out of train if not provided)
        if test_file and os.path.exists(test_file):
            test_df = self._load_and_clean(test_file)
        else:
            train_df, test_df = train_test_split(
                train_df,
                test_size=0.1,
                stratify=train_df["label"],
                random_state=self.random_state,
            )
            print("No test file provided — carved 10% from train as test set.")

        print(f"Test: {len(test_df)}")

        # Add numeric labels
        for df in [train_df, val_df, test_df]:
            df["label_id"] = df["label"].map(LABEL2ID)

        # Print class distribution
        print("\nClass distribution:")
        for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            dist = df["label"].value_counts(normalize=True).round(3).to_dict()
            print(f"  {split_name}: {dist}")

        # Build HuggingFace DatasetDict
        features = Features({
            "text":     Value("string"),
            "label":    Value("string"),
            "label_id": ClassLabel(num_classes=3, names=["Negative", "Neutral", "Positive"]),
        })

        dataset_dict = DatasetDict({
            "train": Dataset.from_pandas(
                train_df[["text", "label", "label_id"]].reset_index(drop=True),
                features=features,
                preserve_index=False,
            ),

            "validation": Dataset.from_pandas(
                val_df[["text", "label", "label_id"]].reset_index(drop=True),
                features=features,
                preserve_index=False,
            ),

            "test": Dataset.from_pandas(
                test_df[["text", "label", "label_id"]].reset_index(drop=True),
                features=features,
                preserve_index=False,
            ),

        })

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            dataset_dict.save_to_disk(save_path)
            # Also save as CSVs for easy inspection
            train_df.to_csv(os.path.join(save_path, "train.csv"), index=False)
            val_df.to_csv(os.path.join(save_path, "val.csv"), index=False)
            test_df.to_csv(os.path.join(save_path, "test.csv"), index=False)
            print(f"\nDataset saved to: {save_path}")

        return dataset_dict

    def get_class_weights(self, dataset_dict: DatasetDict) -> list:
        """
        Compute inverse-frequency class weights for imbalanced datasets.
        Pass these to the loss function during training.
        """
        from sklearn.utils.class_weight import compute_class_weight
        labels = dataset_dict["train"]["label_id"]
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1, 2]),
            y=np.array(labels),
        )
        print(f"Class weights (Neg, Neu, Pos): {weights.round(3)}")
        return weights.tolist()


# ─── Synthetic data augmentation (for class imbalance) ────────────────────────

def augment_with_backtranslation_placeholder(df: pd.DataFrame, target_label: str, n: int = 200) -> pd.DataFrame:
    """
    Placeholder for back-translation augmentation.
    In production: translate Hinglish → English → back to Hinglish via Google Translate API.
    Here we just demonstrate the data format.
    """
    # TODO: Integrate with deep_translator or googletrans
    # from deep_translator import GoogleTranslator
    # translated = GoogleTranslator(source='auto', target='en').translate(text)
    print(f"[PLACEHOLDER] Would augment {target_label} class by {n} samples via back-translation.")
    return df


# ─── Demo / quick sanity check ────────────────────────────────────────────────

def create_demo_dataset() -> DatasetDict:
    """
    Create a tiny demo dataset with handcrafted Hinglish examples.
    Use this to test the pipeline before real data arrives.
    """
    samples = [
        # Positive
        ("yaar ye movie bahut acchi thi, must watch hai", "Positive"),
        ("bhai iska khana ekdum zabardast tha, phir aaunga", "Positive"),
        ("aaj ka match amazing tha, team ne kamaal kar diya", "Positive"),
        ("product bahut acha hai, packaging bhi nice thi", "Positive"),
        ("so happy aaj, sab kuch perfect ho gaya finally", "Positive"),
        ("यह फिल्म बहुत अच्छी थी, loved every moment", "Positive"),
        ("ekdum mast tha bhai, would recommend to everyone", "Positive"),
        ("superb performance, team proud kar diya humein", "Positive"),
        # Negative
        ("yaar ye product bilkul bakwas hai, waste of money", "Negative"),
        ("nhi yar, ye movie boring thi, walkout karna chahiye tha", "Negative"),
        ("khana bahut bekar tha, never going back there", "Negative"),
        ("delivery bahut late tha aur product bhi kharab nikla", "Negative"),
        ("ye company cheating karti hai, fraud hai bilkul", "Negative"),
        ("movie mein kuch bhi interesting nahi tha, time waste", "Negative"),
        ("बहुत bekar service, complaint ka koi jawab nahi aya", "Negative"),
        ("disappointed hu, expectations se bilkul alag tha", "Negative"),
        # Neutral
        ("aaj movie dekhi thi, kaafi theek thi", "Neutral"),
        ("product mile, packaging ok thi, delivery on time tha", "Neutral"),
        ("ye match tie ho gaya, kuch khas nahi hua", "Neutral"),
        ("restaurant mein gaye the, average khana tha", "Neutral"),
        ("abhi decide nahi kiya ki kaisa laga", "Neutral"),
        ("सुना है नई movie आ rahi hai next week", "Neutral"),
        ("theek hai bhai, na bahut acha na bahut bura", "Neutral"),
        ("normal experience tha, kuch special nahi", "Neutral"),
    ]

    df = pd.DataFrame(samples, columns=["text", "label"])
    df["label_id"] = df["label"].map(LABEL2ID)

    # Small splits — just for testing
    train_df, temp_df = train_test_split(df, test_size=0.33, stratify=df["label"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

    return DatasetDict({
        "train":      Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
        "test":       Dataset.from_pandas(test_df.reset_index(drop=True)),
    })


if __name__ == "__main__":
    print("Creating demo dataset for pipeline testing...")
    dsd = create_demo_dataset()
    print(dsd)
    print("\nSample training examples:")
    for ex in dsd["train"].select(range(3)):
        print(f"  [{ex['label']}] {ex['text']}")