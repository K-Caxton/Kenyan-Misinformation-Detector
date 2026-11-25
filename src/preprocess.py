import os
import re
from pathlib import Path
import pandas as pd
from datasets import load_dataset

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"


def clean_tweet_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@user\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_row(text: str) -> bool:
    if not text or text.lower() == "#error!":
        return False
    words = [w for w in text.split() if any(c.isalpha() for c in w)]
    return len(words) >= 2


def prepare_datasets():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset from Hugging Face...")
    ds = load_dataset("lusamaki/Fake_News_Detection_System_29.5k")
    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()

    initial_train_len = len(train_df)
    initial_test_len = len(test_df)

    for df in (train_df, test_df):
        if "unnamed: 0" in df.columns:
            df.drop(columns=["unnamed: 0"], inplace=True)

    # 1. Remove train-test leakage
    overlap_tweets = set(train_df["tweet"].dropna()) & set(test_df["tweet"].dropna())
    train_df = train_df[~train_df["tweet"].isin(overlap_tweets)].copy()

    # 2. Remove all conflicting duplicate tweets in training split
    dup_mask = train_df.duplicated(subset=["tweet"], keep=False)
    removed_dups = int(dup_mask.sum())
    train_df = train_df[~dup_mask].copy()

    # 3. Lightly clean raw tweets (preserving Kiswahili/Sheng structure)
    train_df["clean_text"] = train_df["tweet"].apply(clean_tweet_text)
    test_df["clean_text"] = test_df["tweet"].apply(clean_tweet_text)

    # 4. Filter out corrupted / mention-only / empty rows
    valid_train = train_df["clean_text"].apply(is_valid_row)
    valid_test = test_df["clean_text"].apply(is_valid_row)

    train_clean = train_df[valid_train][["tweet", "clean_text", "label"]].reset_index(drop=True)
    test_clean = test_df[valid_test][["tweet", "clean_text", "label"]].reset_index(drop=True)

    label2id = {"neutral": 0, "fact": 1, "fake": 2}
    train_clean["label_id"] = train_clean["label"].map(label2id)
    test_clean["label_id"] = test_clean["label"].map(label2id)

    train_path = DATA_DIR / "train_clean.csv"
    test_path = DATA_DIR / "test_clean.csv"
    train_clean.to_csv(train_path, index=False)
    test_clean.to_csv(test_path, index=False)

    print("\nDATA PREPARATION REPORT")
    print("=" * 45)
    print(f"Train-test overlapping tweets removed: {len(overlap_tweets)}")
    print(f"Conflicting duplicate rows removed:    {removed_dups}")
    print(f"Corrupted/short train rows removed:    {(~valid_train).sum()}")
    print(f"Corrupted/short test rows removed:     {(~valid_test).sum()}")
    print(f"Final Train Size: {initial_train_len:,} -> {len(train_clean):,}")
    print(f"Final Test Size:  {initial_test_len:,} -> {len(test_clean):,}")
    print(f"\nSaved files:\n- {train_path}\n- {test_path}")


if __name__ == "__main__":
    prepare_datasets()