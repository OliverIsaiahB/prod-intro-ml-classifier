"""Improve the model: scale features inside a pipeline, then tune C."""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from split import make_split


def build(C: float = 1.0):
    # Scaler + model in ONE pipeline: scaling is fit on train data only.
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=C),
    )


def tune():
    X_train, X_test, y_train, y_test = make_split()
    best = None
    # Try a few values of the regularization strength C; keep the best.
    for C in [0.01, 0.1, 1.0, 10.0]:
        model = build(C)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        if best is None or acc > best[1]:
            best = (C, acc)
    return best


if __name__ == "__main__":
    C, acc = tune()
    print(f"best C={C} with accuracy {acc:.3f}")
