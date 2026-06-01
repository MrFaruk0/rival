from dataclasses import dataclass, field

import yaml


@dataclass
class DatasetConfig:
    path: str
    target: str
    missing_strategy: str = "fill_mean"

    def __post_init__(self):
        if self.missing_strategy not in ("fill_mean", "drop"):
            raise ValueError(
                f"invalid missing_strategy '{self.missing_strategy}'. "
                "use 'fill_mean' or 'drop'"
            )


@dataclass
class TrainingConfig:
    test_size: float = 0.2
    random_state: int = 42

    def __post_init__(self):
        if not (0 < self.test_size < 1):
            raise ValueError("test_size must be between 0 and 1")


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig
    models: list[str]
    metrics: list[str]
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        if not self.models:
            raise ValueError("at least one model must be specified")


def load_config(filepath: str) -> ExperimentConfig:
    with open(filepath, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError("config file is empty")

    _validate_top_level(raw)

    training_raw = raw.get("training", {})
    seed = training_raw.get("random_seed", training_raw.get("random_state", 42))

    training = TrainingConfig(
        test_size=training_raw.get("test_size", 0.2),
        random_state=seed,
    )

    dataset = DatasetConfig(
        path=raw["dataset"]["path"],
        target=raw["dataset"]["target"],
        missing_strategy=raw["dataset"].get("missing_strategy", "fill_mean"),
    )

    models = raw.get("models", [])
    metrics = raw.get("metrics", [])

    return ExperimentConfig(
        dataset=dataset,
        models=models,
        metrics=metrics,
        training=training,
    )


def _validate_top_level(raw: dict) -> None:
    if "dataset" not in raw:
        raise ValueError("config must contain 'dataset' section")
    ds = raw["dataset"]
    if "path" not in ds:
        raise ValueError("dataset must contain 'path'")
    if "target" not in ds:
        raise ValueError("dataset must contain 'target'")
    if "models" not in raw:
        raise ValueError("config must contain 'models' list")
    if "metrics" not in raw:
        raise ValueError("config must contain 'metrics' list")
