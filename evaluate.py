"""Measure the model honestly: accuracy, precision, recall, confusion matrix."""
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from model import train_model


def evaluate():
    clf, X_test, y_test = train_model()
    preds = clf.predict(X_test)
    # 'macro' averages each class equally — fair when classes matter equally.
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="macro"),
        "recall": recall_score(y_test, preds, average="macro"),
        "confusion": confusion_matrix(y_test, preds).tolist(),
    }


if __name__ == "__main__":
    for k, v in evaluate().items():
        print(f"{k}: {v}")
