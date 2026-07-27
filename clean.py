"""Check data quality: missing values and types. Garbage in, garbage out."""
import pandas as pd

from load_data import load


def quality_report(df: pd.DataFrame) -> dict:
    return {
        "missing_per_column": df.isna().sum().to_dict(),
        "total_missing": int(df.isna().sum().sum()),
        "non_numeric_columns": [
            c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
        ],
    }


def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    # Simplest honest policy for a small clean dataset: drop incomplete rows.
    before = len(df)
    df = df.dropna()
    print(f"dropped {before - len(df)} rows with missing values")
    return df


if __name__ == "__main__":
    df = load()
    print(quality_report(df))
    df = drop_missing(df)
