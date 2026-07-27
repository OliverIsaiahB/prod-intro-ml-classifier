"""Tiny CLI: load the saved model and classify one wine from the command line.

Usage:
    python predict_cli.py 13.2 1.78 2.14 11.2 100 2.65 2.76 0.26 1.28 4.38 \
        1.05 3.40 1050
"""
import sys
import joblib
import numpy as np

CLASS_NAMES = {0: "class_0", 1: "class_1", 2: "class_2"}


def main() -> None:
    values = [float(x) for x in sys.argv[1:]]   # 13 features from the CLI
    if len(values) != 13:
        print(f"expected 13 features, got {len(values)}")
        sys.exit(1)
    model = joblib.load("wine_model.joblib")
    row = np.array(values).reshape(1, -1)
    pred = int(model.predict(row)[0])
    print("predicted:", CLASS_NAMES[pred])


if __name__ == "__main__":
    main()
