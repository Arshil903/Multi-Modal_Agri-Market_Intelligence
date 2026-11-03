"""Balanced logistic regression, SMOTE + Random Forest, and Bi-LSTM shock detectors."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    LOGREG_PARAMS,
    LSTM_PARAMS,
    MODELS_DIR,
    RANDOM_SEED,
    RF_PARAMS,
    get_device,
    seed_everything,
)
from src.models_regression import create_sequences


class BiLSTMClassifier(nn.Module):
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
        logits = self.head(encoded[:, -1, :]).squeeze(-1)
        return torch.sigmoid(logits)


def train_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    save_path: Path | None = None,
) -> Pipeline:
    seed_everything()
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(**LOGREG_PARAMS)),
        ]
    )
    model.fit(x_train, y_train.astype(int))
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)
    return model


def _smote_resample(x_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_int = y_train.astype(int)
    counts = np.bincount(y_int)
    if len(counts) < 2 or counts.min() < 2:
        return x_train, y_int
    k_neighbors = min(5, int(counts.min()) - 1)
    sampler = SMOTE(random_state=RANDOM_SEED, k_neighbors=k_neighbors)
    return sampler.fit_resample(x_train, y_int)


def train_random_forest_smote(
    x_train: np.ndarray,
    y_train: np.ndarray,
    save_path: Path | None = None,
) -> RandomForestClassifier:
    seed_everything()
    x_res, y_res = _smote_resample(x_train, y_train)
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(x_res, y_res)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)
    return model


def _make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor_x = torch.tensor(x, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(tensor_x, tensor_y), batch_size=batch_size, shuffle=shuffle)


def train_bilstm_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    save_path: Path | None = None,
) -> tuple[BiLSTMClassifier, StandardScaler]:
    seed_everything()
    device = get_device()
    lookback = LSTM_PARAMS["lookback"]
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    y_train_f = y_train.astype(np.float32)
    seq_x, seq_y = create_sequences(x_train_scaled, y_train_f, lookback)
    if len(seq_x) == 0:
        raise ValueError("Not enough rows to build LSTM sequences for classification.")

    pos = float(seq_y.mean()) if len(seq_y) else 0.0
    pos_weight = None
    if 0.0 < pos < 1.0:
        pos_weight = (1.0 - pos) / pos

    val_loader = None
    if x_val is not None and y_val is not None and len(x_val) > 0:
        x_val_scaled = scaler.transform(x_val)
        if lookback > 1:
            context_x = np.vstack([x_train_scaled[-(lookback - 1) :], x_val_scaled])
            context_y = np.concatenate([y_train_f[-(lookback - 1) :], y_val.astype(np.float32)])
        else:
            context_x = x_val_scaled
            context_y = y_val.astype(np.float32)
        seq_vx, seq_vy = create_sequences(context_x, context_y, lookback)
        seq_vx = seq_vx[-len(x_val) :]
        seq_vy = seq_vy[-len(x_val) :]
        if len(seq_vx) > 0:
            val_loader = _make_loader(seq_vx, seq_vy, LSTM_PARAMS["batch_size"], shuffle=False)

    model = BiLSTMClassifier(
        input_size=x_train.shape[1],
        hidden_size=LSTM_PARAMS["hidden_size"],
        num_layers=LSTM_PARAMS["num_layers"],
        dropout=LSTM_PARAMS["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LSTM_PARAMS["learning_rate"],
        weight_decay=LSTM_PARAMS["weight_decay"],
    )
    loss_fn = nn.BCELoss(reduction="none")
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
            probs = model(batch_x).clamp(1e-6, 1 - 1e-6)
            raw_loss = loss_fn(probs, batch_y)
            if pos_weight is not None:
                weights = torch.where(batch_y > 0.5, torch.full_like(batch_y, pos_weight), torch.ones_like(batch_y))
                loss = (raw_loss * weights).mean()
            else:
                loss = raw_loss.mean()
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
                probs = model(batch_x).clamp(1e-6, 1 - 1e-6)
                val_losses.append(nn.BCELoss()(probs, batch_y).item())
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
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_size": int(x_train.shape[1]),
                "params": LSTM_PARAMS,
                "scaler": scaler,
            },
            save_path,
        )
        joblib.dump(scaler, MODELS_DIR / f"{save_path.stem}_scaler.joblib")
    return model, scaler


def predict_lstm_classifier(
    model: BiLSTMClassifier,
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
        probs = model(torch.tensor(seq_x, dtype=torch.float32).to(device)).cpu().numpy()
    if len(probs) < len(x_eval):
        pad = np.full(len(x_eval) - len(probs), np.nan, dtype=np.float32)
        probs = np.concatenate([pad, probs])
    return probs.astype(np.float32)


def load_bilstm_classifier(path: Path) -> tuple[BiLSTMClassifier, StandardScaler]:
    try:
        checkpoint = torch.load(path, map_location=get_device(), weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=get_device())
    model = BiLSTMClassifier(
        input_size=int(checkpoint["input_size"]),
        hidden_size=LSTM_PARAMS["hidden_size"],
        num_layers=LSTM_PARAMS["num_layers"],
        dropout=LSTM_PARAMS["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(get_device())
    model.eval()
    scaler = checkpoint.get("scaler")
    if scaler is None:
        scaler = joblib.load(MODELS_DIR / f"{path.stem}_scaler.joblib")
    return model, scaler
