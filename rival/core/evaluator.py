from rival.metrics.metrics import (
    compute_accuracy,
    compute_precision,
    compute_recall,
    compute_f1,
    measure_latency,
)


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": compute_accuracy(y_test, predictions),
        "precision": compute_precision(y_test, predictions),
        "recall": compute_recall(y_test, predictions),
        "f1": compute_f1(y_test, predictions),
        "latency_ms": measure_latency(model, X_test),
    }
    return metrics
