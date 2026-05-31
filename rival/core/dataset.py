import pandas as pd
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(
        self,
        filepath: str,
        target: str,
        test_size: float = 0.2,
        random_state: int = 42,
        missing_strategy: str = "fill_mean",
    ):
        self.filepath = filepath
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.missing_strategy = missing_strategy

    def load(self):
        if not self.filepath.endswith(".csv"):
            raise ValueError("only CSV files are supported")

        df = pd.read_csv(self.filepath)

        if self.target not in df.columns:
            raise ValueError(
                f"target column '{self.target}' not found in dataset"
            )

        df = self._handle_missing(df)

        X = df.drop(columns=[self.target])
        y = df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        return X_train, X_test, y_train, y_test

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.missing_strategy == "drop":
            return df.dropna()
        elif self.missing_strategy == "fill_mean":
            numeric_cols = df.select_dtypes(include="number").columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            return df
        else:
            raise ValueError(
                f"unknown missing strategy '{self.missing_strategy}'. "
                "use 'drop' or 'fill_mean'"
            )
