from rival.core.benchmark import Benchmark, BenchmarkResult
from rival.core.config import ExperimentConfig


class Experiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run(self) -> list[BenchmarkResult]:
        benchmark = Benchmark(
            dataset_path=self.config.dataset.path,
            target=self.config.dataset.target,
            model_names=self.config.models,
            test_size=self.config.training.test_size,
            random_state=self.config.training.random_state,
            missing_strategy=self.config.dataset.missing_strategy,
            metric_names=self.config.metrics,
        )
        return benchmark.run()
