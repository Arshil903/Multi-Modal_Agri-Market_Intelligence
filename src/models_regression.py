"""Ridge, XGBoost, and bidirectional LSTM models for next-day price forecasting."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from src.config import LSTM_PARAMS, MODELS_DIR, RANDOM_SEED, RIDGE_ALPHA, XGB_PARAMS, get_device, seed_everything


class BiLSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.lstm(x)
        last_step = encoded[:, -1, :]
        return self.head(last_step).squeeze(-1)


def create_sequences(
    features: np.ndarray,
    target: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(features) < lookback:
        return np.empty((0, lookback, features.shape[1])), np.empty((0,))
    xs, ys = [], []
    for end in range(lookback - 1, len(features)):
        start = end - lookback + 1
        xs.append(features[start : end + 1])
        ys.append(target[end])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def chronological_split(frame: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values("Date").reset_index(drop=True)
    split_at = int(len(ordered) * train_fraction)
    split_at = max(split_at, 1)
    split_at = min(split_at, len(ordered) - 1)
    return ordered.iloc[:split_at].copy(), ordered.iloc[split_at:].copy()


def train_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    save_path: Path | None = None,
) -> Pipeline:
    seed_everything()
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )
    model.fit(x_train, y_train)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)
    return model


def train_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    save_path: Path | None = None,
) -> XGBRegressor:
    seed_everything()
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(x_train, y_train)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)
    return model


def _make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor_x = torch.tensor(x, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_bilstm_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    save_path: Path | None = None,
) -> tuple[BiLSTMRegressor, StandardScaler]:
    seed_everything()
    device = get_device()
    lookback = LSTM_PARAMS["lookback"]
    scaler = StandardScaler()
    y_scaler = StandardScaler()
    n_features = x_train.shape[1]
    x_train_scaled = scaler.fit_transform(x_train)
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel().astype(np.float32)
    seq_x, seq_y = create_sequences(x_train_scaled, y_train_scaled, lookback)
    if len(seq_x) == 0:
        raise ValueError("Not enough rows to build LSTM sequences for regression.")

    val_loader = None
    if x_val is not None and y_val is not None and len(x_val) > 0:
        x_val_scaled = scaler.transform(x_val)
        y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).ravel().astype(np.float32)
        context_x = np.vstack([x_train_scaled[-(lookback - 1) :], x_val_scaled]) if lookback > 1 else x_val_scaled
        context_y = np.concatenate([y_train_scaled[-(lookback - 1) :], y_val_scaled]) if lookback > 1 else y_val_scaled
        seq_vx, seq_vy = create_sequences(context_x, context_y, lookback)
        seq_vx = seq_vx[-len(x_val) :]
        seq_vy = seq_vy[-len(x_val) :]
        if len(seq_vx) > 0:
            val_loader = _make_loader(seq_vx, seq_vy, LSTM_PARAMS["batch_size"], shuffle=False)

    model = BiLSTMRegressor(
        input_size=n_features,
        hidden_size=LSTM_PARAMS["hidden_size"],
        num_layers=LSTM_PARAMS["num_layers"],
        dropout=LSTM_PARAMS["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LSTM_PARAMS["learning_rate"],
        weight_decay=LSTM_PARAMS["weight_decay"],
    )
    loss_fn = nn.MSELoss()
    train_loader = _make_loader(seq_x, seq_y, LSTM_PARAMS["batch_size"], shuffle=True)

    best_state = None
    best_val = float("inf")
    patience_left = LSTM_PARAMS["patience"]
    for _epoch in range(LSTM_PARAMS["epochs"]):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if val_loader is None:
            continue
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                val_losses.append(loss_fn(model(batch_x), batch_y).item())
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = LSTM_PARAMS["patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    model.y_scaler = y_scaler

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_size": n_features,
                "params": LSTM_PARAMS,
                "scaler": scaler,
                "y_scaler": y_scaler,
            },
            save_path,
        )
        joblib.dump({"feature_scaler": scaler, "y_scaler": y_scaler}, MODELS_DIR / f"{save_path.stem}_scaler.joblib")
    return model, scaler


def predict_lstm_regressor(
    model: BiLSTMRegressor,
    scaler: StandardScaler,
    x_hist: np.ndarray,
    y_hist: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
) -> np.ndarray:
    device = get_device()
    lookback = LSTM_PARAMS["lookback"]
    x_hist_s = scaler.transform(x_hist)
    x_eval_s = scaler.transform(x_eval)
    if lookback > 1 and len(x_hist_s) > 0:
        context_x = np.vstack([x_hist_s[-(lookback - 1) :], x_eval_s])
        context_y = np.concatenate([y_hist.astype(np.float32)[-(lookback - 1) :], y_eval.astype(np.float32)])
    else:
        context_x = x_eval_s
        context_y = y_eval.astype(np.float32)
    seq_x, _ = create_sequences(context_x, context_y, lookback)
    seq_x = seq_x[-len(x_eval) :]
    if len(seq_x) == 0:
        return np.full(len(x_eval), np.nan, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(seq_x, dtype=torch.float32).to(device)).cpu().numpy()
    y_scaler = getattr(model, "y_scaler", None)
    if y_scaler is not None:
        preds = y_scaler.inverse_transform(np.asarray(preds).reshape(-1, 1)).ravel()
    if len(preds) < len(x_eval):
        pad = np.full(len(x_eval) - len(preds), np.nan, dtype=np.float32)
        preds = np.concatenate([pad, preds])
    return preds.astype(np.float32)


def load_bilstm_regressor(path: Path) -> tuple[BiLSTMRegressor, StandardScaler]:
    try:
        checkpoint = torch.load(path, map_location=get_device(), weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=get_device())
    model = BiLSTMRegressor(
        input_size=int(checkpoint["input_size"]),
        hidden_size=LSTM_PARAMS["hidden_size"],
        num_layers=LSTM_PARAMS["num_layers"],
        dropout=LSTM_PARAMS["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(get_device())
    model.eval()
    scaler = checkpoint.get("scaler")
    y_scaler = checkpoint.get("y_scaler")
    if scaler is None:
        bundle = joblib.load(MODELS_DIR / f"{path.stem}_scaler.joblib")
        if isinstance(bundle, dict):
            scaler = bundle["feature_scaler"]
            y_scaler = bundle.get("y_scaler")
        else:
            scaler = bundle
    model.y_scaler = y_scaler
    return model, scaler
