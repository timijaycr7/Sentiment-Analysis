import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split


def load_data(file_path):
    df = pd.read_csv(file_path)
    df["label"] = df["sentiment"].apply(lambda value: 1 if value == "positive" else 0)
    return df


def create_hf_datasets(df, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)
    return train_dataset, test_dataset