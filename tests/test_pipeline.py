"""
tests/test_pipeline.py

Unit tests for preprocessing, dataset building, and inference pipeline.

Run: pytest tests/ -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing.cleaner import HinglishCleaner, detect_script_ratio, is_hinglish
from src.preprocessing.dataset_builder import create_demo_dataset
from src.config import LABEL2ID, ID2LABEL


# ─── Cleaner tests ─────────────────────────────────────────────────────────────

class TestHinglishCleaner:

    def setup_method(self):
        self.cleaner = HinglishCleaner()

    def test_removes_urls(self):
        result = self.cleaner.clean("check this https://t.co/abc out")
        assert "https" not in result
        assert "t.co" not in result

    def test_removes_mentions(self):
        result = self.cleaner.clean("@RCB yaar match accha tha")
        assert "@RCB" not in result
        assert "yaar" in result

    def test_expands_hashtags(self):
        result = self.cleaner.clean("great movie #Bollywood")
        assert "#" not in result
        assert "Bollywood" in result.lower() or "bollywood" in result

    def test_normalizes_repeated_chars(self):
        result = self.cleaner.clean("soooooo boreddddd")
        assert "sooooo" not in result  # Should be reduced to max 2
        assert "so" in result

    def test_preserves_devanagari(self):
        result = self.cleaner.clean("यह bahut accha tha")
        assert "यह" in result

    def test_normalizes_slang(self):
        result = self.cleaner.clean("nhi yar ye bakwas hai")
        assert "nahi" in result or "nhi" not in result

    def test_empty_string(self):
        result = self.cleaner.clean("")
        assert result == ""

    def test_none_like_input(self):
        result = self.cleaner.clean("   ")
        assert result == ""

    def test_clean_batch(self):
        texts = ["@user hello!", "yaar accha tha", "  "]
        results = self.cleaner.clean_batch(texts)
        assert len(results) == 3
        assert results[2] == ""


class TestScriptDetection:

    def test_pure_latin(self):
        ratio = detect_script_ratio("hello world this is english")
        assert ratio["latin"] > 0.8
        assert ratio["devanagari"] == 0.0

    def test_pure_devanagari(self):
        ratio = detect_script_ratio("यह एक हिन्दी वाक्य है")
        assert ratio["devanagari"] > 0.5

    def test_mixed(self):
        ratio = detect_script_ratio("यह bahut accha tha")
        assert ratio["devanagari"] > 0
        assert ratio["latin"] > 0

    def test_empty(self):
        ratio = detect_script_ratio("")
        assert ratio == {"devanagari": 0.0, "latin": 0.0, "other": 0.0}

    def test_is_hinglish_roman(self):
        assert is_hinglish("yaar ye movie bahut acchi thi")

    def test_is_hinglish_short(self):
        # Too short / pure English should still pass the 30% Latin threshold
        assert is_hinglish("hello")


# ─── Dataset builder tests ─────────────────────────────────────────────────────

class TestDatasetBuilder:

    def test_demo_dataset_structure(self):
        dsd = create_demo_dataset()
        assert "train" in dsd
        assert "validation" in dsd
        assert "test" in dsd

    def test_demo_dataset_columns(self):
        dsd = create_demo_dataset()
        cols = set(dsd["train"].column_names)
        assert "text" in cols
        assert "label" in cols
        assert "label_id" in cols

    def test_demo_dataset_labels_valid(self):
        dsd = create_demo_dataset()
        for split in ["train", "validation", "test"]:
            labels = set(dsd[split]["label"])
            assert labels.issubset({"Positive", "Negative", "Neutral"})

    def test_demo_dataset_label_ids_valid(self):
        dsd = create_demo_dataset()
        for split in ["train", "validation", "test"]:
            ids = set(dsd[split]["label_id"])
            assert ids.issubset({0, 1, 2})

    def test_demo_dataset_nonempty(self):
        dsd = create_demo_dataset()
        assert len(dsd["train"]) > 0
        assert len(dsd["validation"]) > 0
        assert len(dsd["test"]) > 0


# ─── Config tests ──────────────────────────────────────────────────────────────

class TestConfig:

    def test_label_maps_consistent(self):
        for label, idx in LABEL2ID.items():
            assert ID2LABEL[idx] == label

    def test_all_labels_present(self):
        for label in ["Positive", "Negative", "Neutral"]:
            assert label in LABEL2ID

    def test_label_indices_contiguous(self):
        assert set(LABEL2ID.values()) == {0, 1, 2}


# ─── Integration: cleaner → dataset pipeline ──────────────────────────────────

class TestPipeline:

    def test_clean_then_label(self):
        cleaner = HinglishCleaner()
        raw_texts = [
            "@user yaar ye bahut accha tha!! #great",
            "https://t.co/xyz bekar product waste hai",
            "theek thi movie, average experience",
        ]
        cleaned = cleaner.clean_batch(raw_texts)
        assert len(cleaned) == 3
        assert all(isinstance(t, str) for t in cleaned)
        assert all("@" not in t for t in cleaned)
        assert all("https" not in t for t in cleaned)

    def test_demo_e2e(self):
        """Full end-to-end: build demo dataset → check integrity."""
        dsd = create_demo_dataset()
        train = dsd["train"]
        for example in train:
            assert isinstance(example["text"], str)
            assert example["label"] in {"Positive", "Negative", "Neutral"}
            assert example["label_id"] == LABEL2ID[example["label"]]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])