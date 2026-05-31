import time
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def compute_accuracy(y_true, y_pred) -> float:
    return accuracy_score(y_true, y_pred)


def compute_precision(y_true, y_pred) -> float:
    return precision_score(y_true, y_pred, average="weighted", zero_division=0)


def compute_recall(y_true, y_pred) -> float:
    return recall_score(y_true, y_pred, average="weighted", zero_division=0)


def compute_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="weighted", zero_division=0)


def measure_latency(model, X_test, n_runs: int = 5) -> float:
    timings = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(X_test)
        end = time.perf_counter()
        timings.append((end - start) * 1000)
    return float(np.mean(timings))
