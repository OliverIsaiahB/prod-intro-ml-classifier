"""Train a classifier with scikit-learn's fit/predict API."""
from sklearn.linear_model import LogisticRegression

from split import make_split


def train_model():
    X_train, X_test, y_train, y_test = make_split()
    # max_iter raised so the solver fully converges on this data.
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)        # LEARN the pattern from training data
    return clf, X_test, y_test


if __name__ == "__main__":
    clf, X_test, y_test = train_model()
    preds = clf.predict(X_test)      # APPLY the learned pattern to new data
    print("first 10 predictions:", list(preds[:10]))
    print("first 10 actual:    ", list(y_test[:10]))
