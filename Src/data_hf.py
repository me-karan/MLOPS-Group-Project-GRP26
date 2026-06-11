from datasets import load_dataset
from pathlib import Path
import pandas as pd

from data_local import PROCESSED_DIR

RAW_DATA_DIR = Path("Datasets/HFData/Raw")
TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"


def load_data():

    train_dataset = load_dataset(
        "stanfordnlp/imdb",
        split="train"
    )

    test_dataset = load_dataset(
        "stanfordnlp/imdb",
        split="test"
    )

    return train_dataset, test_dataset


def convert_to_dataframe(train_dataset, test_dataset):
    train_df = train_dataset.to_pandas()
    test_df = test_dataset.to_pandas()
    return train_df, test_df


def save_data(train_df, test_df):
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print(f"Saved train data to: {TRAIN_PATH}")
    print(f"Saved test data to: {TEST_PATH}")

def main():
    train_dataset, test_dataset = load_data()
    train_df, test_df = convert_to_dataframe(train_dataset, test_dataset)

    save_data(train_df, test_df)

    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")

if __name__ == "__main__":
    main()
