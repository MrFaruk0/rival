from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from rival.core.registry import register_model, get_model


@register_model("logistic_regression")
class LogisticRegressionAdapter:
    def __init__(self, random_state: int = 42):
        self.model = LogisticRegression(
            max_iter=2000,
            random_state=random_state,
        )
        self.name = "logistic_regression"

    def fit(self, X_train, y_train):
        self._le = LabelEncoder()
        y_encoded = self._le.fit_transform(y_train)
        self.model.fit(X_train, y_encoded)
        return self

    def predict(self, X_test):
        preds = self.model.predict(X_test)
        return self._le.inverse_transform(preds)


@register_model("random_forest")
class RandomForestAdapter:
    def __init__(self, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
        )
        self.name = "random_forest"

    def fit(self, X_train, y_train):
        self._le = LabelEncoder()
        y_encoded = self._le.fit_transform(y_train)
        self.model.fit(X_train, y_encoded)
        return self

    def predict(self, X_test):
        preds = self.model.predict(X_test)
        return self._le.inverse_transform(preds)


@register_model("xgboost")
class XGBoostAdapter:
    def __init__(self, random_state: int = 42):
        self.model = None
        self.name = "xgboost"
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


MODEL_ALIASES = {
    "logistic_regression": "logistic_regression",
    "random_forest": "random_forest",
    "xgboost": "xgboost",
    "lr": "logistic_regression",
    "rf": "random_forest",
    "xgb": "xgboost",
}
