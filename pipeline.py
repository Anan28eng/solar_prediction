import os
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib


def load_data(csv_path: str = "solar_data_cleaned.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Date-Hour(NMT)"] = pd.to_datetime(df["Date-Hour(NMT)"])
    df.set_index("Date-Hour(NMT)", inplace=True)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Minimal cleaning — assumes `week-1.ipynb` produced a cleaned CSV
    df = df.copy()
    # Ensure non-negative radiation
    if "Radiation" in df.columns:
        df["Radiation"] = df["Radiation"].apply(lambda x: max(x, 0))
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # lag features
    for i in range(1, 5):
        df[f"lag_{i}"] = df["SystemProduction"].shift(i).fillna(0)

    # time features
    df["hour"] = df.index.hour
    df["weekday"] = df.index.weekday

    # rolling features
    df["rolling_6"] = df["SystemProduction"].rolling(window=6, min_periods=1).mean().fillna(0)
    df["rolling_24"] = df["SystemProduction"].rolling(window=24, min_periods=1).mean().fillna(0)

    return df


def get_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    features = [
        "WindSpeed",
        "Sunshine",
        "AirPressure",
        "Radiation",
        "AirTemperature",
        "RelativeAirHumidity",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_4",
        "hour",
        "weekday",
        "rolling_6",
        "rolling_24",
    ]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")

    X = df[features].copy()
    y = df["SystemProduction"].copy()
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    return train_test_split(X, y, test_size=test_size, shuffle=False)


def train_xgb(X_train: pd.DataFrame, y_train: pd.Series, params: dict = None) -> xgb.XGBRegressor:
    if params is None:
        params = {
            "objective": "reg:squarederror",
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "verbosity": 0,
        }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def save_model(model, path: str = "models/xgb_model.joblib"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str = "models/xgb_model.joblib"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def build_and_train(csv_path: str = "solar_data_cleaned.csv", save_path: str = "models/xgb_model.joblib") -> dict:
    df = load_data(csv_path)
    df = clean_data(df)
    df = add_features(df)
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    model = train_xgb(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)

    save_model(model, save_path)
    return {"model": model, "metrics": metrics, "X_test": X_test, "y_test": y_test, "y_pred": y_pred}


if __name__ == "__main__":
    # Run a training pipeline when executed directly
    result = build_and_train()
    print("Training finished. Metrics:")
    for k, v in result["metrics"].items():
        print(f"{k}: {v}")
