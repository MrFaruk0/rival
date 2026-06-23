import json

from rival.history.history_manager import HISTORY_DIR


def compute_leaderboard(metric="f1"):
    best_scores = {}

    if HISTORY_DIR.exists():
        for run_file in sorted(HISTORY_DIR.glob("run_*.json")):
            with open(run_file) as f:
                run = json.load(f)
            for model_name, model_metrics in run.get("metrics", {}).items():
                score = model_metrics.get(metric)
                if score is not None:
                    if model_name not in best_scores or score > best_scores[model_name]:
                        best_scores[model_name] = score

    ranked = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked
