"""Split the table into X (features the model sees) and y (the answer)."""
import pandas as pd

from load_data import load

TARGET = "target"   # the column holding the class we want to predict


def split_xy(df: pd.DataFrame):
    # X = everything EXCEPT the answer; y = ONLY the answer.
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


if __name__ == "__main__":
    df = load()
    X, y = split_xy(df)
    print("X shape:", X.shape, "| y shape:", y.shape)
    print("classes:", sorted(y.unique()))
