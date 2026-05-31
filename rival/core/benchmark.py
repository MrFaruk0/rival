import warnings
from dataclasses import dataclass, field

from rival.core.dataset import Dataset
from rival.core.evaluator import evaluate
from rival.core.trainer import train
from rival.models.sklearn_models import get_model
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
    model_names: list[str] = field(default_factory=lambda: ["lr", "rf"])
    test_size: float = 0.2
    random_state: int = 42
    missing_strategy: str = "fill_mean"

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

        for name in self.model_names:
            try:
                model = get_model(name, random_state=self.random_state)

                if hasattr(model, "_available") and not model._available:
                    continue

                train(model, X_train, y_train)
                metrics = evaluate(model, X_test, y_test)

                results.append(BenchmarkResult(
                    model_name=name,
                    accuracy=round(metrics["accuracy"], 4),
                    precision=round(metrics["precision"], 4),
                    recall=round(metrics["recall"], 4),
                    f1=round(metrics["f1"], 4),
                    latency_ms=round(metrics["latency_ms"], 1),
                ))
            except Exception:
                continue

        return results
