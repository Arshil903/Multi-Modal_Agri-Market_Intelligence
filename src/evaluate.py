"""Regression and classification metrics plus comparative reporting."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.config import BENCHMARK_CSV, ensure_directories


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(np.asarray(arr, dtype=float))
    return mask


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = _finite_mask(y_true, y_pred)
    if mask.sum() == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "R2": np.nan}
    yt, yp = y_true[mask], y_pred[mask]
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def classification_metrics(
    y_true: Iterable[float],
    y_prob: Iterable[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    mask = _finite_mask(y_true, y_prob)
    if mask.sum() == 0:
        return {
            "Precision": np.nan,
            "Recall": np.nan,
            "Macro_F1": np.nan,
            "ROC_AUC": np.nan,
            "PR_AUC": np.nan,
            "Accuracy": np.nan,
        }
    yt = y_true[mask].astype(int)
    yp = y_prob[mask]
    yhat = (yp >= threshold).astype(int)
    metrics = {
        "Precision": float(precision_score(yt, yhat, average="binary", zero_division=0)),
        "Recall": float(recall_score(yt, yhat, average="binary", zero_division=0)),
        "Macro_F1": float(f1_score(yt, yhat, average="macro", zero_division=0)),
        "Accuracy": float(accuracy_score(yt, yhat)),
        "ROC_AUC": np.nan,
        "PR_AUC": np.nan,
    }
    if len(np.unique(yt)) > 1:
        metrics["ROC_AUC"] = float(roc_auc_score(yt, yp))
        metrics["PR_AUC"] = float(average_precision_score(yt, yp))
    return metrics


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def markdown_table(df: pd.DataFrame, float_cols: list[str] | None = None) -> str:
    float_cols = float_cols or [c for c in df.columns if df[c].dtype.kind == "f"]
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for col in headers:
            val = row[col]
            if col in float_cols and not isinstance(val, str):
                cells.append(_fmt(float(val)))
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def print_regression_comparison(results: pd.DataFrame) -> None:
    task_a = results.loc[results["Task"] == "A_Price_Forecast"].copy()
    if task_a.empty:
        print("No Task A results to display.")
        return
    print("\n## Task A — Next-day Price Forecasting (Market-Only vs Multimodal)\n")
    show = task_a[["Commodity", "Regime", "Model", "RMSE", "MAE", "R2"]].sort_values(
        ["Commodity", "Model", "Regime"]
    )
    print(markdown_table(show, ["RMSE", "MAE", "R2"]))


def print_classification_comparison(results: pd.DataFrame) -> None:
    task_b = results.loc[results["Task"] == "B_Shock_Detection"].copy()
    if task_b.empty:
        print("No Task B results to display.")
        return
    print("\n## Task B — Market Shock Detection\n")
    show = task_b[
        ["Commodity", "Regime", "Model", "Precision", "Recall", "Macro_F1", "ROC_AUC", "PR_AUC"]
    ].sort_values(["Commodity", "Model", "Regime"])
    print(markdown_table(show, ["Precision", "Recall", "Macro_F1", "ROC_AUC", "PR_AUC"]))


def save_benchmark(results: pd.DataFrame, path=None) -> pd.DataFrame:
    ensure_directories()
    out_path = path or BENCHMARK_CSV
    results.to_csv(out_path, index=False)
    return results
