import warnings
from dataclasses import dataclass, field

from rival.core.dataset import Dataset
from rival.core.evaluator import evaluate
from rival.core.registry import get_model
from rival.core.trainer import train
from rival.utils.seed import set_global_seed


@dataclass
class BenchmarkResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    latency_ms: float


@dataclass
class Benchmark:
    dataset_path: str
    target: str
    model_names: list[str] = field(default_factory=lambda: ["logistic_regression", "random_forest"])
    test_size: float = 0.2
    random_state: int = 42
    missing_strategy: str = "fill_mean"
    metric_names: list[str] = field(default_factory=list)

    def run(self) -> list[BenchmarkResult]:
        set_global_seed(self.random_state)
        warnings.filterwarnings("ignore")

        dataset = Dataset(
            filepath=self.dataset_path,
            target=self.target,
            test_size=self.test_size,
            random_state=self.random_state,
            missing_strategy=self.missing_strategy,
        )
        X_train, X_test, y_train, y_test = dataset.load()

        results: list[BenchmarkResult] = []

        names = self.metric_names if self.metric_names else None

        for name in self.model_names:
            try:
                model = get_model(name, random_state=self.random_state)

                if hasattr(model, "_available") and not model._available:
                    continue

                train(model, X_train, y_train)
                metrics = evaluate(model, X_test, y_test, metric_names=names)

                results.append(BenchmarkResult(
                    model_name=model.name,
                    accuracy=round(metrics.get("accuracy", 0), 4),
                    precision=round(metrics.get("precision", 0), 4),
                    recall=round(metrics.get("recall", 0), 4),
                    f1=round(metrics.get("f1", 0), 4),
                    latency_ms=round(metrics["latency_ms"], 1),
                ))
            except Exception:
                continue

        if results:
            try:
                from rival.history.history_manager import HistoryManager
                primary_metric = self.metric_names[0] if self.metric_names else "f1"
                HistoryManager.save(
                    dataset_path=self.dataset_path,
                    target=self.target,
                    model_names=self.model_names,
                    results=results,
                    primary_metric=primary_metric,
                )
            except Exception:
                pass

        return results
