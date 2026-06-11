import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

#Just for loading local data, not used in the main code
RAW_DATA_PATH = Path("dataset/imdb_top_1000.csv")

PROCESSED_DIR = Path("dataset/hf_processed")
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"

def load_data(csv_path=RAW_DATA_PATH):
    df = pd.read_csv(csv_path)

    print(f"Dataset shape: {df.shape}")
    print(df.head())

    return df


def split_data(df):
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
    )

    return train_df, test_df

def save_data(train_df, test_df):
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)


if __name__ == "__main__":
    df = load_data()

    train_df, test_df = split_data(df)
    save_data(train_df, test_df)

    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")