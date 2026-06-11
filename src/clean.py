"""Stage 1 - cleaning and dataset preparation.

Scope (deliberately limited):
  * split ``results.csv`` into played history vs. unplayed 2026 fixtures
  * clean ``eloratings.csv`` (drop null / zero ratings, sort by team+date)
  * basic de-duplication and light type fixes on the remaining tables
  * build a short summary table describing every dataset
  * persist cleaned outputs to ``data/interim/``

No feature engineering, model training, or simulation happens here. Goalscorers
are cleaned for completeness only and are NOT used as predictive features yet.
"""
from __future__ import annotations

import pandas as pd

from . import config, ingest


# --------------------------------------------------------------------------- #
# results.csv: split played vs. unplayed fixtures
# --------------------------------------------------------------------------- #
def split_results(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split results into played matches and unplayed 2026 fixtures.

    A row is an *unplayed fixture* when either score is missing. We additionally
    require the date to be on/after ``FIXTURE_CUTOFF`` so that any genuinely
    missing historical scores are not silently treated as future fixtures
    (they would be flagged separately for review instead).

    Returns
    -------
    (played, fixtures)
        ``played``   - matches with both scores present (training history).
        ``fixtures`` - future matches with missing scores (prediction targets).
    """
    df = results.copy()
    df = df.drop_duplicates()

    missing_score = df["home_score"].isna() | df["away_score"].isna()
    is_future = df["date"] >= pd.Timestamp(config.FIXTURE_CUTOFF)

    fixtures = df[missing_score & is_future].copy()
    played = df[~missing_score].copy()

    # Guard: any pre-cutoff rows missing scores are data problems, not fixtures.
    orphan_missing = df[missing_score & ~is_future]
    if len(orphan_missing):
        print(
            f"[warn] {len(orphan_missing)} pre-{config.FIXTURE_CUTOFF} rows have "
            "missing scores and were excluded from both played and fixtures."
        )

    # Played scores are safe to store as integers now that NaNs are gone.
    played["home_score"] = played["home_score"].astype("int64")
    played["away_score"] = played["away_score"].astype("int64")

    # Drop score columns from fixtures (they are the unknown target).
    fixtures = fixtures.drop(columns=["home_score", "away_score"])

    played = played.sort_values("date").reset_index(drop=True)
    fixtures = fixtures.sort_values("date").reset_index(drop=True)
    return played, fixtures


# --------------------------------------------------------------------------- #
# eloratings.csv: drop invalid ratings, sort
# --------------------------------------------------------------------------- #
def clean_eloratings(elo: pd.DataFrame) -> pd.DataFrame:
    """Clean Elo ratings: drop unparseable dates, null/zero ratings, sort."""
    df = elo.copy()
    df = df.drop_duplicates()

    before = len(df)

    # Dates that failed both ISO and US parsing in ingest are unusable.
    df = df[df["date"].notna()]

    # Drop null ratings and invalid (<= 0) ratings.
    df = df[df["rating"].notna()]
    df = df[df["rating"] > config.MIN_VALID_ELO]

    df["rating"] = df["rating"].astype("float64")

    df = df.sort_values(["team", "date"]).reset_index(drop=True)

    print(f"[elo] kept {len(df)}/{before} rows after dropping invalid records")
    return df


# --------------------------------------------------------------------------- #
# goalscorers.csv / shootouts.csv / former_names.csv: light cleaning
# --------------------------------------------------------------------------- #
def clean_goalscorers(goals: pd.DataFrame) -> pd.DataFrame:
    """De-duplicate and type-fix goalscorers (kept for completeness, not a feature)."""
    df = goals.copy()
    df = df.drop_duplicates()
    # minute is a discrete value; nullable integer keeps NaNs without floats.
    df["minute"] = df["minute"].astype("Int64")
    return df.sort_values("date").reset_index(drop=True)


def clean_shootouts(shootouts: pd.DataFrame) -> pd.DataFrame:
    df = shootouts.copy()
    df = df.drop_duplicates()
    return df.sort_values("date").reset_index(drop=True)


def clean_former_names(former: pd.DataFrame) -> pd.DataFrame:
    df = former.copy()
    df = df.drop_duplicates()
    return df.sort_values(["current", "start_date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Summary table
# --------------------------------------------------------------------------- #
def summarize(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a one-row-per-dataset summary: shape, nulls, dups, date range."""
    rows = []
    for name, df in frames.items():
        date_min = date_max = None
        if "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["date"]):
            date_min = df["date"].min()
            date_max = df["date"].max()
        rows.append(
            {
                "dataset": name,
                "rows": len(df),
                "columns": df.shape[1],
                "total_nulls": int(df.isna().sum().sum()),
                "duplicate_rows": int(df.duplicated().sum()),
                "date_min": date_min,
                "date_max": date_max,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run() -> dict[str, pd.DataFrame]:
    """Run the full Stage 1 cleaning pipeline and write interim artifacts."""
    config.ensure_dirs()

    raw = ingest.load_all()

    played, fixtures = split_results(raw["results"])
    elo_clean = clean_eloratings(raw["eloratings"])
    goals_clean = clean_goalscorers(raw["goalscorers"])
    shootouts_clean = clean_shootouts(raw["shootouts"])
    former_clean = clean_former_names(raw["former_names"])

    outputs = {
        "results_played": played,
        "fixtures_2026": fixtures,
        "eloratings_clean": elo_clean,
        "goalscorers_clean": goals_clean,
        "shootouts_clean": shootouts_clean,
        "former_names_clean": former_clean,
    }

    for key, df in outputs.items():
        df.to_parquet(config.INTERIM_FILES[key], index=False)

    summary = summarize(outputs)
    summary.to_csv(config.INTERIM_FILES["summary"], index=False)

    print("\n=== Stage 1 dataset summary ===")
    print(summary.to_string(index=False))
    print(f"\nInterim files written to: {config.INTERIM_DIR}")

    outputs["summary"] = summary
    return outputs


if __name__ == "__main__":
    run()
