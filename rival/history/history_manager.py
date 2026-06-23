import json
from datetime import datetime, timezone
from pathlib import Path

from rival import __version__

HISTORY_DIR = Path.cwd() / ".rival" / "runs"


class HistoryManager:
    @staticmethod
    def save(dataset_path, target, model_names, results, primary_metric="f1"):
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).isoformat()
        run_id = f"run_{int(datetime.now(timezone.utc).timestamp())}"

        metrics = {}
        for r in results:
            metrics[r.model_name] = {
                "accuracy": r.accuracy,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "latency_ms": r.latency_ms,
            }

        winner = max(metrics, key=lambda m: metrics[m][primary_metric], default=None) if metrics else None

        run_data = {
            "run_id": run_id,
            "timestamp": timestamp,
            "rival_version": __version__,
            "dataset": dataset_path,
            "target": target,
            "models": model_names,
            "metrics": metrics,
            "winner": winner,
            "primary_metric": primary_metric,
        }

        run_file = HISTORY_DIR / f"{run_id}.json"
        with open(run_file, "w") as f:
            json.dump(run_data, f, indent=2)

        return run_id

    @staticmethod
    def list_runs():
        if not HISTORY_DIR.exists():
            return []

        runs = []
        for file in sorted(HISTORY_DIR.glob("run_*.json")):
            with open(file) as f:
                data = json.load(f)
            runs.append({
                "run_id": data["run_id"],
                "timestamp": data["timestamp"],
                "dataset": data.get("dataset", ""),
                "models": data.get("models", []),
                "winner": data.get("winner", ""),
            })
        return runs

    @staticmethod
    def get_latest_run():
        runs = HistoryManager.list_runs()
        if not runs:
            return None
        return HistoryManager.get_run(runs[-1]["run_id"])

    @staticmethod
    def get_run(run_id):
        run_file = HISTORY_DIR / f"{run_id}.json"
        if not run_file.exists():
            raise FileNotFoundError(f"Run '{run_id}' not found.")
        with open(run_file) as f:
            return json.load(f)
