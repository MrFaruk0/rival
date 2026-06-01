import json
from typing import Optional

import typer

from rival.core.benchmark import Benchmark
from rival.core.config import load_config
from rival.core.experiment import Experiment

app = typer.Typer(
    name="rival",
    help="Lightweight ML benchmarking CLI tool. Compare ML models easily.",
    invoke_without_command=True,
)


@app.callback()
def main():
    """
    Lightweight ML benchmarking CLI tool. Compare ML models easily.
    """


METRIC_LABELS = {
    "accuracy": "ACCURACY",
    "precision": "PRECISION",
    "recall": "RECALL",
    "f1": "F1",
}


def _format_results_table(results, metric_names=None) -> str:
    if metric_names is None:
        metric_names = ["accuracy", "precision", "recall", "f1"]

    labels = [METRIC_LABELS.get(m, m.upper()) for m in metric_names]
    col_width = 10
    model_width = max(10, max((len(r.model_name.upper()) for r in results), default=5) + 1)
    header = f"{'MODEL':<{model_width}}" + "".join(f"{l:<{col_width}}" for l in labels) + f"{'LATENCY':<10}"
    sep = "-" * (model_width + col_width * len(labels) + 10)

    lines = [sep, header, sep]

    for r in results:
        name = r.model_name.upper()
        values = []
        for m in metric_names:
            val = getattr(r, m, 0)
            values.append(f"{val:<{col_width}.4f}")
        lines.append(
            f"{name:<{model_width}}{''.join(values)}{r.latency_ms}ms"
        )

    lines.append(sep)
    return "\n".join(lines)


def _export_json(results, output: str):
    export_data = []
    for r in results:
        export_data.append({
            "model": r.model_name,
            "accuracy": r.accuracy,
            "precision": r.precision,
            "recall": r.recall,
            "f1": r.f1,
            "latency_ms": r.latency_ms,
        })
    with open(output, "w") as f:
        json.dump(export_data, f, indent=2)
    typer.echo(f"\nResults exported to {output}")


@app.command()
def run(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to YAML experiment config"),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Path to CSV dataset"),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Target column name"),
    models: str = typer.Option("lr,rf", "--models", "-m", help="Comma-separated models: lr,rf,xgb"),
    missing: str = typer.Option("fill_mean", "--missing", help="Missing value strategy: fill_mean or drop"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed for reproducibility"),
    test_size: float = typer.Option(0.2, "--test-size", help="Test split ratio"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Export results as JSON"),
):
    """
    Run ML model benchmarks on a CSV dataset.
    """
    if config is not None:
        experiment_config = load_config(config)
        experiment = Experiment(experiment_config)
        results = experiment.run()
        metric_names = experiment_config.metrics
    elif dataset is not None and target is not None:
        model_names = [m.strip().lower() for m in models.split(",")]
        benchmark = Benchmark(
            dataset_path=dataset,
            target=target,
            model_names=model_names,
            test_size=test_size,
            random_state=seed,
            missing_strategy=missing,
        )
        results = benchmark.run()
        metric_names = ["accuracy", "precision", "recall", "f1"]
    else:
        raise typer.BadParameter(
            "either --config or both --dataset and --target must be provided"
        )

    typer.echo(_format_results_table(results, metric_names=metric_names))

    if output:
        _export_json(results, output)


if __name__ == "__main__":
    app()
