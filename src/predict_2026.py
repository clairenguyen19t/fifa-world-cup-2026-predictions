"""Stage 4 - predict 2026 group-stage fixture probabilities.

Loads the Stage 3 best-model bundle and the Stage 2 fixture features, applies the
*exact* preprocessing recorded in the bundle (same feature list, same Elo
missingness indicators, same train medians, same class order), and emits a
3-way outcome probability for every 2026 fixture.

This stage only produces fixture-level probabilities. It does not train a model,
does not use any actual 2026 scores, and does not simulate the tournament.
"""
from __future__ import annotations

import joblib
import pandas as pd

from . import config

ID_COLS = ["date", "home_team", "away_team", "tournament"]
PROBA_COLS = ["P_home_win", "P_draw", "P_away_win"]
# Maps stored class labels -> output probability column names.
CLASS_TO_PROBA = {"home_win": "P_home_win", "draw": "P_draw", "away_win": "P_away_win"}


def load_bundle() -> dict:
    """Load the trained best-model bundle saved in Stage 3."""
    return joblib.load(config.OUTPUT_FILES["best_model"])


def apply_preprocessing(fixtures: pd.DataFrame, impute_state: dict) -> pd.DataFrame:
    """Recreate the Stage 3 feature matrix for the fixtures.

    Uses the stored feature list, Elo missingness indicators, and train medians
    so inference matches training exactly.
    """
    features = impute_state["features"]
    indicator_cols = impute_state["indicator_cols"]
    feature_names = impute_state["feature_names"]
    medians = impute_state["medians"]

    X = fixtures[features].copy()

    # Rebuild missingness indicators (computed BEFORE imputation, as in training).
    for ind in indicator_cols:
        base = ind[: -len("_missing")]
        X[ind] = X[base].isna().astype("int64")

    # Impute numeric features with the TRAIN medians captured at fit time.
    X[features] = X[features].fillna(value=medians)

    return X[feature_names]


def predict(fixtures: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Return fixtures with P_home_win / P_draw / P_away_win columns."""
    model = bundle["model"]
    X = apply_preprocessing(fixtures, bundle["impute_state"])

    proba = model.predict_proba(X)
    proba_df = pd.DataFrame(proba, columns=list(model.classes_), index=fixtures.index)

    out = fixtures[ID_COLS].copy()
    for cls, col in CLASS_TO_PROBA.items():
        out[col] = proba_df[cls].values

    # Most likely outcome + its confidence for convenient downstream use.
    label_map = {"P_home_win": "home_win", "P_draw": "draw", "P_away_win": "away_win"}
    out["predicted_outcome"] = out[PROBA_COLS].idxmax(axis=1).map(label_map)
    out["confidence"] = out[PROBA_COLS].max(axis=1)
    return out


def print_report(preds: pd.DataFrame) -> None:
    """Print the required summary views of the predictions."""
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    show = ID_COLS[:3] + PROBA_COLS + ["predicted_outcome", "confidence"]

    print("=" * 80)
    print("2026 GROUP-STAGE FIXTURE PREDICTIONS")
    print("=" * 80)
    print(f"fixtures predicted: {len(preds)}")

    print("\n--- First 10 predictions ---")
    print(preds[show].head(10).round(3).to_string(index=False))

    print("\n--- Highest-confidence matches (top 10) ---")
    top = preds.sort_values("confidence", ascending=False).head(10)
    print(top[show].round(3).to_string(index=False))

    print("\n--- Most balanced matches (lowest confidence, top 10) ---")
    bal = preds.sort_values("confidence", ascending=True).head(10)
    print(bal[show].round(3).to_string(index=False))

    print(f"\n--- Average predicted draw probability: {preds['P_draw'].mean():.4f} ---")


def run() -> pd.DataFrame:
    """Run Stage 4: load, predict, save, and report."""
    config.ensure_dirs()

    bundle = load_bundle()
    fixtures = pd.read_parquet(config.PROCESSED_FILES["fixtures_2026_features"])
    fixtures["date"] = pd.to_datetime(fixtures["date"])

    print(f"Loaded model: {bundle['model_name']}")
    print(f"Fixtures: {len(fixtures)} rows\n")

    preds = predict(fixtures, bundle)
    preds.to_csv(config.OUTPUT_FILES["fixtures_2026_predictions"], index=False)

    print_report(preds)
    print(f"\nSaved -> {config.OUTPUT_FILES['fixtures_2026_predictions']}")
    return preds


if __name__ == "__main__":
    run()
