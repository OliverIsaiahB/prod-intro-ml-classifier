"""Visualize the data with Matplotlib before modeling — see, don't guess."""
import matplotlib.pyplot as plt

from load_data import load
from prepare import split_xy


def plot_class_balance(y) -> None:
    # How many examples of each class? Imbalance changes everything later.
    counts = y.value_counts().sort_index()
    counts.plot(kind="bar")
    plt.title("Examples per class")
    plt.xlabel("class")
    plt.ylabel("count")
    plt.savefig("class_balance.png")
    plt.close()


def plot_feature_by_class(X, y, feature: str) -> None:
    # Does this feature separate the classes? A good feature does.
    for cls in sorted(y.unique()):
        plt.hist(X[y == cls][feature], alpha=0.5, label=f"class {cls}")
    plt.title(f"{feature} by class")
    plt.legend()
    plt.savefig(f"{feature}_by_class.png")
    plt.close()


if __name__ == "__main__":
    df = load()
    X, y = split_xy(df)
    plot_class_balance(y)
    plot_feature_by_class(X, y, "alcohol")
