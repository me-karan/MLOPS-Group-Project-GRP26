# src/preprocess.py

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )


def load_data():
    train_df = pd.read_csv("Datasets/HFData/Processed/processed_train.csv")
    test_df = pd.read_csv("Datasets/HFData/Processed/processed_test.csv")

    return train_df, test_df


def create_datasets(train_df, test_df):
    train_dataset = Dataset.from_pandas(
        train_df,
        preserve_index=False
    )

    test_dataset = Dataset.from_pandas(
        test_df,
        preserve_index=False
    )

    return train_dataset, test_dataset


def preprocess_data():
    
    train_df, test_df = load_data()

    train_dataset, test_dataset = create_datasets(
        train_df,
        test_df
    )

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True
    )

    test_dataset = test_dataset.map(
        tokenize_function,
        batched=True
    )

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = preprocess_data()

    print(train_dataset)
    print(test_dataset)