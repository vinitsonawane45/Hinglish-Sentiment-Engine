"""
scripts/preprocess.py

Run data cleaning + build HuggingFace DatasetDict from raw files.

Run: python scripts/preprocess.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Paths
from src.preprocessing.cleaner import HinglishCleaner
from src.preprocessing.dataset_builder import HinglishDatasetBuilder


def main():
    Paths.make_all()

    print("=== Hinglish Sentiment Engine — Preprocessing ===\n")

    # Check data files exist
    for split, path in [("train", Paths.SEMEVAL_TRAIN),
                        ("val",   Paths.SEMEVAL_VAL),
                        ("test",  Paths.SEMEVAL_TEST)]:
        if not os.path.exists(path):
            print(f"Missing: {path}")
            print("Run: python scripts/download_data.py first.")
            sys.exit(1)

    # Build cleaner
    cleaner = HinglishCleaner(
        remove_mentions=True,
        expand_hashtags=True,
        remove_urls=True,
        handle_emoji="text",      # Convert emoji to text description
        normalize_slang=True,
        normalize_repeats=True,
        preserve_devanagari=True,
        lowercase=True,
    )

    # Build dataset
    builder = HinglishDatasetBuilder(cleaner=cleaner, val_size=0.1, random_state=42)

    # Check for any extra annotated data
    extra = []
    if os.path.exists(os.path.join(Paths.DATA_ANN, "my_data.csv")):
        extra.append(os.path.join(Paths.DATA_ANN, "my_data.csv"))
        print(f"Found extra annotated data: {extra}")

    save_path = os.path.join(Paths.DATA_PROC, "dataset")

    # dataset_dict = builder.build(
    #     train_file=Paths.SEMEVAL_TRAIN,
    #     test_file=Paths.SEMEVAL_TEST,
    #     extra_files=extra if extra else None,
    #     save_path=save_path,
    # )
    dataset_dict = builder.build(
    train_file=Paths.SEMEVAL_TRAIN,
    test_file=Paths.SEMEVAL_TEST,
    extra_files=extra if extra else None,
    save_path=save_path,
)

    # Save CSV versions for evaluation script
    train_df = dataset_dict["train"].to_pandas()
    val_df = dataset_dict["validation"].to_pandas()
    test_df = dataset_dict["test"].to_pandas()

    train_df.to_csv(os.path.join(Paths.DATA_PROC, "train.csv"), index=False)
    val_df.to_csv(os.path.join(Paths.DATA_PROC, "val.csv"), index=False)
    test_df.to_csv(os.path.join(Paths.DATA_PROC, "test.csv"), index=False)

    print("CSV files saved:")
    print(f"  - {os.path.join(Paths.DATA_PROC, 'train.csv')}")
    print(f"  - {os.path.join(Paths.DATA_PROC, 'val.csv')}")
    print(f"  - {os.path.join(Paths.DATA_PROC, 'test.csv')}")

    # Compute and save class weights
    weights = builder.get_class_weights(dataset_dict)
    import json
    weights_path = os.path.join(Paths.DATA_PROC, "class_weights.json")
    with open(weights_path, "w") as f:
        json.dump({"weights": weights, "labels": ["Negative", "Neutral", "Positive"]}, f)
    print(f"Class weights saved to: {weights_path}")

    print("\n=== Preprocessing complete ===")
    print(f"Dataset saved to: {save_path}")
    print("\nNext step: python scripts/train.py")


if __name__ == "__main__":
    main()