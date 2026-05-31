import json
from typing import Optional

import typer

from rival.core.benchmark import Benchmark

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


def _format_results_table(results) -> str:
    header = f"{'MODEL':<10} {'ACCURACY':<10} {'PRECISION':<10} {'RECALL':<10} {'F1':<10} {'LATENCY':<10}"
    sep = "-" * 60

    lines = [sep, header, sep]

    for r in results:
        lines.append(
            f"{r.model_name.upper():<10} {r.accuracy:<10.4f} {r.precision:<10.4f} "
            f"{r.recall:<10.4f} {r.f1:<10.4f} {r.latency_ms}ms"
        )

    lines.append(sep)
    return "\n".join(lines)


@app.command()
def run(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Path to CSV dataset"),
    target: str = typer.Option(..., "--target", "-t", help="Target column name"),
    models: str = typer.Option("lr,rf", "--models", "-m", help="Comma-separated models: lr,rf,xgb"),
    missing: str = typer.Option("fill_mean", "--missing", help="Missing value strategy: fill_mean or drop"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed for reproducibility"),
    test_size: float = typer.Option(0.2, "--test-size", help="Test split ratio"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Export results as JSON"),
):
    """
    Run ML model benchmarks on a CSV dataset.
    """
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

    typer.echo(_format_results_table(results))

    if output:
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


if __name__ == "__main__":
    app()
