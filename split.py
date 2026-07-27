"""Split into train and test sets — the test set is a locked vault."""
from sklearn.model_selection import train_test_split

from load_data import load
from prepare import split_xy


def make_split(test_size: float = 0.2, seed: int = 42):
    X, y = split_xy(load())
    # stratify=y keeps the class proportions identical in both splits.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = make_split()
    print("train:", X_train.shape, "| test:", X_test.shape)
