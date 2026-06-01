from rival.metrics.metrics import (
    compute_accuracy,
    compute_precision,
    compute_recall,
    compute_f1,
    measure_latency,
)

METRIC_FUNCTIONS = {
    "accuracy": compute_accuracy,
    "precision": compute_precision,
    "recall": compute_recall,
    "f1": compute_f1,
}


def evaluate(model, X_test, y_test, metric_names=None):
    predictions = model.predict(X_test)

    if metric_names is None:
        metric_names = list(METRIC_FUNCTIONS.keys())

    metrics = {}
    for name in metric_names:
        if name in METRIC_FUNCTIONS:
            metrics[name] = METRIC_FUNCTIONS[name](y_test, predictions)

    metrics["latency_ms"] = measure_latency(model, X_test)
    return metrics
