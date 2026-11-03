"""CLI entry point: build data, run the ablation study, and save artifacts."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    COMMODITIES,
    FEATURE_REGIMES,
    MODELS_DIR,
    PREDICTIONS_CSV,
    TRAIN_FRACTION,
    ensure_directories,
    seed_everything,
)
from src.evaluate import (  # noqa: E402
    classification_metrics,
    print_classification_comparison,
    print_regression_comparison,
    regression_metrics,
    save_benchmark,
)
from src.feature_engineering import load_or_build_master  # noqa: E402
from src.models_classification import (  # noqa: E402
    predict_lstm_classifier,
    train_bilstm_classifier,
    train_logistic_regression,
    train_random_forest_smote,
)
from src.models_regression import (  # noqa: E402
    chronological_split,
    predict_lstm_regressor,
    train_bilstm_regressor,
    train_ridge,
    train_xgboost,
)

LOGGER = logging.getLogger(__name__)


def _model_path(commodity: str, task: str, model_name: str, regime: str, suffix: str) -> Path:
    safe_crop = commodity.lower()
    return MODELS_DIR / f"{safe_crop}_{task}_{model_name}_{regime}.{suffix}"


def _prepare_xy(frame: pd.DataFrame, features: list[str], target: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    work = frame.dropna(subset=features + [target]).copy()
    x = work[features].to_numpy(dtype=np.float32)
    y = work[target].to_numpy(dtype=np.float32)
    return work, x, y


def train_commodity(master: pd.DataFrame, commodity: str) -> tuple[list[dict], pd.DataFrame]:
    crop = master.loc[master["Commodity"] == commodity].sort_values("Date").reset_index(drop=True)
    rows: list[dict] = []
    pred_frame = crop[["Date", "Commodity", "Modal_Price", "Target_Price", "Target_Shock"]].copy()
    pred_frame["split"] = "train"
    split_at = int(len(crop) * TRAIN_FRACTION)
    pred_frame.loc[split_at:, "split"] = "test"

    for regime, features in FEATURE_REGIMES.items():
        missing = [c for c in features if c not in crop.columns]
        if missing:
            raise KeyError(f"{commodity} is missing features {missing}")

        reg_df, _, _ = _prepare_xy(crop, features, "Target_Price")
        train_reg, test_reg = chronological_split(reg_df, TRAIN_FRACTION)
        x_tr = train_reg[features].to_numpy(dtype=np.float32)
        y_tr = train_reg["Target_Price"].to_numpy(dtype=np.float32)
        x_te = test_reg[features].to_numpy(dtype=np.float32)
        y_te = test_reg["Target_Price"].to_numpy(dtype=np.float32)

        ridge_path = _model_path(commodity, "reg", "ridge", regime, "joblib")
        xgb_path = _model_path(commodity, "reg", "xgboost", regime, "joblib")
        lstm_reg_path = _model_path(commodity, "reg", "bilstm", regime, "pt")

        LOGGER.info("%s | %s | Ridge", commodity, regime)
        ridge = train_ridge(x_tr, y_tr, save_path=ridge_path)
        ridge_hat = ridge.predict(x_te)

        LOGGER.info("%s | %s | XGBoost", commodity, regime)
        xgb = train_xgboost(x_tr, y_tr, save_path=xgb_path)
        xgb_hat = xgb.predict(x_te)

        LOGGER.info("%s | %s | BiLSTM regressor", commodity, regime)
        inner_cut = max(int(len(x_tr) * 0.85), 40)
        if len(x_tr) - inner_cut >= 10:
            lstm_reg, lstm_reg_scaler = train_bilstm_regressor(
                x_tr[:inner_cut],
                y_tr[:inner_cut],
                x_val=x_tr[inner_cut:],
                y_val=y_tr[inner_cut:],
                save_path=lstm_reg_path,
            )
        else:
            lstm_reg, lstm_reg_scaler = train_bilstm_regressor(x_tr, y_tr, save_path=lstm_reg_path)
        lstm_hat = predict_lstm_regressor(lstm_reg, lstm_reg_scaler, x_tr, y_tr, x_te, y_te)

        for model_name, y_hat in (("Ridge", ridge_hat), ("XGBoost", xgb_hat), ("BiLSTM", lstm_hat)):
            metrics = regression_metrics(y_te, y_hat)
            rows.append(
                {
                    "Commodity": commodity,
                    "Task": "A_Price_Forecast",
                    "Regime": regime,
                    "Model": model_name,
                    **metrics,
                    "Precision": np.nan,
                    "Recall": np.nan,
                    "Macro_F1": np.nan,
                    "ROC_AUC": np.nan,
                    "PR_AUC": np.nan,
                    "Accuracy": np.nan,
                    "n_train": len(train_reg),
                    "n_test": len(test_reg),
                }
            )

        pred_frame = pred_frame.merge(
            test_reg[["Date"]].assign(
                **{
                    f"pred_price_ridge_{regime}": ridge_hat,
                    f"pred_price_xgboost_{regime}": xgb_hat,
                    f"pred_price_bilstm_{regime}": lstm_hat,
                }
            ),
            on="Date",
            how="left",
        )

        clf_df, _, _ = _prepare_xy(crop, features, "Target_Shock")
        train_clf, test_clf = chronological_split(clf_df, TRAIN_FRACTION)
        xc_tr = train_clf[features].to_numpy(dtype=np.float32)
        yc_tr = train_clf["Target_Shock"].to_numpy(dtype=np.float32)
        xc_te = test_clf[features].to_numpy(dtype=np.float32)
        yc_te = test_clf["Target_Shock"].to_numpy(dtype=np.float32)

        logreg_path = _model_path(commodity, "clf", "logreg", regime, "joblib")
        rf_path = _model_path(commodity, "clf", "random_forest", regime, "joblib")
        lstm_clf_path = _model_path(commodity, "clf", "bilstm", regime, "pt")

        LOGGER.info("%s | %s | LogisticRegression", commodity, regime)
        logreg = train_logistic_regression(xc_tr, yc_tr, save_path=logreg_path)
        logreg_prob = logreg.predict_proba(xc_te)[:, 1]

        LOGGER.info("%s | %s | RandomForest + SMOTE", commodity, regime)
        rf = train_random_forest_smote(xc_tr, yc_tr, save_path=rf_path)
        rf_prob = rf.predict_proba(xc_te)[:, 1]

        LOGGER.info("%s | %s | BiLSTM classifier", commodity, regime)
        inner_cut_c = max(int(len(xc_tr) * 0.85), 40)
        if len(xc_tr) - inner_cut_c >= 10:
            lstm_clf, lstm_clf_scaler = train_bilstm_classifier(
                xc_tr[:inner_cut_c],
                yc_tr[:inner_cut_c],
                x_val=xc_tr[inner_cut_c:],
                y_val=yc_tr[inner_cut_c:],
                save_path=lstm_clf_path,
            )
        else:
            lstm_clf, lstm_clf_scaler = train_bilstm_classifier(xc_tr, yc_tr, save_path=lstm_clf_path)
        lstm_prob = predict_lstm_classifier(lstm_clf, lstm_clf_scaler, xc_tr, yc_tr, xc_te, yc_te)

        for model_name, y_prob in (
            ("LogisticRegression", logreg_prob),
            ("RandomForest_SMOTE", rf_prob),
            ("BiLSTM", lstm_prob),
        ):
            metrics = classification_metrics(yc_te, y_prob)
            rows.append(
                {
                    "Commodity": commodity,
                    "Task": "B_Shock_Detection",
                    "Regime": regime,
                    "Model": model_name,
                    "RMSE": np.nan,
                    "MAE": np.nan,
                    "R2": np.nan,
                    **metrics,
                    "n_train": len(train_clf),
                    "n_test": len(test_clf),
                }
            )

        pred_frame = pred_frame.merge(
            test_clf[["Date"]].assign(
                **{
                    f"pred_shock_logreg_{regime}": logreg_prob,
                    f"pred_shock_rf_{regime}": rf_prob,
                    f"pred_shock_bilstm_{regime}": lstm_prob,
                }
            ),
            on="Date",
            how="left",
        )

        joblib.dump(features, MODELS_DIR / f"{commodity.lower()}_{regime}_features.joblib")

    return rows, pred_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Gujarat TOP price-forecast and shock-detection models."
    )
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="Rebuild the processed multimodal master CSV even if it already exists.",
    )
    parser.add_argument(
        "--commodities",
        nargs="+",
        default=COMMODITIES,
        help="Subset of crops to train.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    ensure_directories()
    seed_everything()

    LOGGER.info("Building / loading the multimodal master dataset")
    master = load_or_build_master(rebuild=args.rebuild_data)
    LOGGER.info("Master dataset: %s rows, columns=%s", len(master), list(master.columns))

    all_rows: list[dict] = []
    pred_frames: list[pd.DataFrame] = []
    for commodity in args.commodities:
        if commodity not in set(master["Commodity"].unique()):
            LOGGER.warning("Skipping %s — not present in the master dataset", commodity)
            continue
        LOGGER.info("===== Training %s =====", commodity)
        rows, preds = train_commodity(master, commodity)
        all_rows.extend(rows)
        pred_frames.append(preds)

    results = pd.DataFrame(all_rows)
    save_benchmark(results)
    print_regression_comparison(results)
    print_classification_comparison(results)
    print("\nSaved benchmark CSV to data/processed/model_benchmark_results.csv")

    if pred_frames:
        predictions = pd.concat(pred_frames, ignore_index=True)
        predictions.to_csv(PREDICTIONS_CSV, index=False)
        LOGGER.info("Wrote prediction overlay to %s", PREDICTIONS_CSV)

    LOGGER.info("Artifacts stored in %s", MODELS_DIR)


if __name__ == "__main__":
    main()
