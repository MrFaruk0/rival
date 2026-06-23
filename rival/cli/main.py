import json
from typing import Optional

import typer

from rival.core.benchmark import Benchmark
from rival.core.config import load_config
from rival.core.experiment import Experiment
from rival.history.history_manager import HistoryManager
from rival.history.leaderboard import compute_leaderboard

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
    models: str = typer.Option("logistic_regression,random_forest", "--models", "-m", help="Comma-separated models: logistic_regression,random_forest,xgboost (short aliases lr,rf,xgb supported)"),
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


@app.command()
def history(
    latest: bool = typer.Option(False, "--latest", "-l", help="Show only the latest run"),
):
    """
    Show experiment run history.
    """
    if latest:
        run = HistoryManager.get_latest_run()
        if run is None:
            typer.echo("No runs found.")
            return
        _show_run_detail(run)
        return

    runs = HistoryManager.list_runs()
    if not runs:
        typer.echo("No runs found.")
        return

    typer.echo(f"{'RUN ID':<25}{'DATE':<28}{'DATASET':<20}{'WINNER'}")
    typer.echo("-" * 90)
    for run in runs:
        ts = run["timestamp"][:19] if run["timestamp"] else ""
        ds = run.get("dataset", "")[-20:]
        winner = run.get("winner", "")
        typer.echo(f"{run['run_id']:<25}{ts:<28}{ds:<20}{winner}")


def _show_run_detail(run):
    typer.echo(f"Run ID:         {run['run_id']}")
    typer.echo(f"Date:           {run['timestamp'][:19] if run['timestamp'] else ''}")
    typer.echo(f"Rival Version:  {run.get('rival_version', '')}")
    typer.echo(f"Dataset:        {run.get('dataset', '')}")
    typer.echo(f"Target:         {run.get('target', '')}")
    typer.echo(f"Primary Metric: {run.get('primary_metric', '')}")
    typer.echo(f"Winner:         {run.get('winner', '')}")
    typer.echo(f"Models:         {', '.join(run.get('models', []))}")
    typer.echo()
    typer.echo("Metrics:")

    col_width = 12
    metrics = run.get("metrics", {})
    if metrics:
        header = f"{'MODEL':<22}" + f"{'ACC':<{col_width}}{'PREC':<{col_width}}{'REC':<{col_width}}{'F1':<{col_width}}{'LATENCY'}"
        sep = "-" * (22 + col_width * 4 + 10)
        typer.echo(sep)
        typer.echo(header)
        typer.echo(sep)
        for model_name, m in metrics.items():
            typer.echo(
                f"{model_name:<22}"
                f"{m['accuracy']:<{col_width}.4f}"
                f"{m['precision']:<{col_width}.4f}"
                f"{m['recall']:<{col_width}.4f}"
                f"{m['f1']:<{col_width}.4f}"
                f"{m['latency_ms']}ms"
            )
        typer.echo(sep)


@app.command()
def leaderboard(
    metric: str = typer.Option("f1", "--metric", "-m", help="Metric to rank by: accuracy, precision, recall, f1"),
):
    """
    Show leaderboard of best model scores from run history.
    """
    ranked = compute_leaderboard(metric=metric)
    if not ranked:
        typer.echo("No runs found.")
        return

    metric_label = METRIC_LABELS.get(metric, metric.upper())
    typer.echo(f"{'MODEL':<25}{'BEST ' + metric_label}")
    typer.echo("-" * 45)
    for model_name, score in ranked:
        typer.echo(f"{model_name:<25}{score:.4f}")


if __name__ == "__main__":
    app()
