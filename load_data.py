"""Load the wine dataset into a Pandas DataFrame — our starting point.

Every ML project begins here: get the data into a table you can inspect.
"""
import pandas as pd
from sklearn.datasets import load_wine


def load() -> pd.DataFrame:
    data = load_wine(as_frame=True)
    df = data.frame            # features + a 'target' column, all in one table
    return df


if __name__ == "__main__":
    df = load()
    print("shape:", df.shape)          # (rows, columns)
    print("columns:", list(df.columns))
    print(df.head())                   # the first 5 rows
