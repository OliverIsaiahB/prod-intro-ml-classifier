"""Fit the best pipeline on ALL data and save it for reuse."""
import joblib

from improve import build, tune
from split import make_split


def main() -> None:
    C, _ = tune()                 # find the best hyperparameter
    X_train, X_test, y_train, y_test = make_split()
    model = build(C)
    model.fit(X_train, y_train)   # fit the final model
    joblib.dump(model, "wine_model.joblib")
    print(f"saved wine_model.joblib (C={C})")


if __name__ == "__main__":
    main()
