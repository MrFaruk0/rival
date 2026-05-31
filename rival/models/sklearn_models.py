from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


class LogisticRegressionAdapter:
    def __init__(self, random_state: int = 42):
        self.model = LogisticRegression(
            max_iter=2000,
            random_state=random_state,
        )
        self.name = "lr"

    def fit(self, X_train, y_train):
        self._le = LabelEncoder()
        y_encoded = self._le.fit_transform(y_train)
        self.model.fit(X_train, y_encoded)
        return self

    def predict(self, X_test):
        preds = self.model.predict(X_test)
        return self._le.inverse_transform(preds)


class RandomForestAdapter:
    def __init__(self, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
        )
        self.name = "rf"

    def fit(self, X_train, y_train):
        self._le = LabelEncoder()
        y_encoded = self._le.fit_transform(y_train)
        self.model.fit(X_train, y_encoded)
        return self

    def predict(self, X_test):
        preds = self.model.predict(X_test)
        return self._le.inverse_transform(preds)


class XGBoostAdapter:
    def __init__(self, random_state: int = 42):
        self.model = None
        self.name = "xgb"
        self.random_state = random_state
        self._available = False
        self._error = None

        try:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=100,
                random_state=random_state,
                verbosity=0,
                use_label_encoder=False,
            )
            self._available = True
        except ImportError as e:
            self._error = f"xgboost not installed: {e}"

    def fit(self, X_train, y_train):
        if not self._available:
            return self
        self._le = LabelEncoder()
        y_encoded = self._le.fit_transform(y_train)
        self.model.fit(X_train, y_encoded)
        return self

    def predict(self, X_test):
        if not self._available:
            return None
        preds = self.model.predict(X_test)
        return self._le.inverse_transform(preds)


def get_model(name: str, random_state: int = 42):
    registry = {
        "lr": LogisticRegressionAdapter,
        "rf": RandomForestAdapter,
        "xgb": XGBoostAdapter,
    }
    name = name.lower().strip()
    if name not in registry:
        raise ValueError(
            f"unknown model '{name}'. available: {', '.join(registry.keys())}"
        )
    return registry[name](random_state=random_state)
