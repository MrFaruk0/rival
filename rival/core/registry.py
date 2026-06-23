_registry: dict[str, type] = {}

_LEGACY_ALIASES = {
    "lr": "logistic_regression",
    "rf": "random_forest",
    "xgb": "xgboost",
}


def register_model(name: str):
    def decorator(cls):
        key = name.lower().strip()
        if key in _registry:
            raise ValueError(f"Model '{key}' is already registered.")
        _registry[key] = cls
        return cls
    return decorator


def get_model(name: str, random_state: int = 42):
    key = name.lower().strip()
    key = _LEGACY_ALIASES.get(key, key)
    if key not in _registry:
        available = ", ".join(sorted(_registry.keys()))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return _registry[key](random_state=random_state)


def list_registered() -> list[str]:
    return sorted(_registry.keys())
