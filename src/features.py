"""Stage 2 - feature engineering.

Builds a leakage-safe feature table for played matches (training) and for the
unplayed 2026 World Cup fixtures (prediction inputs), using only the cleaned
interim artifacts from Stage 1.

Leakage guarantees (by construction):
  * Targets (home_win / draw / away_win) come only from the *current* match
    score and are never reused as inputs.
  * Elo features use a temporal as-of join with ``allow_exact_matches=False``,
    so only ratings dated *strictly before* the match are attached. The Elo
    ``change`` column (which encodes the match result) is never used.
  * Rolling "form" features for a match use only that team's matches dated
    strictly before the current match. For played rows this is enforced with a
    within-team shift; for fixtures with a strictly-backward as-of join.
  * Goals from the current match are never used as features.

No model training or tournament simulation happens here.
"""
from __future__ import annotations

import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #
def add_targets(played: pd.DataFrame) -> pd.DataFrame:
    """Add categorical and one-hot match-outcome targets from the final score."""
    df = played.copy()
    diff = df["home_score"] - df["away_score"]
    df["result"] = pd.Series(
        pd.cut(
            diff, bins=[-float("inf"), -1, 0, float("inf")],
            labels=["away_win", "draw", "home_win"],
        )
    ).astype("object")
    df["home_win"] = (diff > 0).astype("int64")
    df["draw"] = (diff == 0).astype("int64")
    df["away_win"] = (diff < 0).astype("int64")
    return df


# --------------------------------------------------------------------------- #
# Tournament weight
# --------------------------------------------------------------------------- #
def tournament_weight(name: str) -> float:
    """Map a tournament name to an importance weight.

    Exact-name match first, then substring rules so the long tail of
    qualification / continental competitions gets sensible values.
    """
    if name in config.TOURNAMENT_WEIGHTS:
        return config.TOURNAMENT_WEIGHTS[name]

    n = name.lower()
    if "world cup" in n and "qualif" in n:
        return 0.85
    if "qualif" in n:                       # continental qualifiers
        return 0.65
    if "nations league" in n:
        return 0.75
    # Continental finals / major confederation cups not listed explicitly.
    continental_kw = (
        "euro", "copa", "asian cup", "cup of nations", "gold cup",
        "gulf cup", "championship", "nations cup",
    )
    if any(k in n for k in continental_kw):
        return 0.75
    return config.DEFAULT_TOURNAMENT_WEIGHT


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add context features: neutral flag (int) and tournament_weight."""
    out = df.copy()
    out["neutral"] = out["neutral"].astype("int64")
    out["tournament_weight"] = out["tournament"].map(tournament_weight).astype("float64")
    return out


# --------------------------------------------------------------------------- #
# Elo: temporal as-of join (strictly before match date)
# --------------------------------------------------------------------------- #
def _asof_elo(matches: pd.DataFrame, elo: pd.DataFrame, team_col: str, out_col: str) -> pd.Series:
    """Latest Elo rating for ``team_col`` strictly before each match date.

    Returns a Series aligned to ``matches`` index named ``out_col``.
    """
    base = (
        matches[["__rid__", "date", team_col]]
        .dropna(subset=["date"])
        .sort_values("date")
    )
    right = (
        elo[["date", "team", "rating"]]
        .dropna(subset=["date"])
        .sort_values("date")
    )
    merged = pd.merge_asof(
        base,
        right,
        on="date",
        left_by=team_col,
        right_by="team",
        direction="backward",
        allow_exact_matches=False,  # strictly before
    )
    return merged.set_index("__rid__")["rating"].rename(out_col)


def add_elo_features(
    played: pd.DataFrame, fixtures: pd.DataFrame, elo: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach home_elo, away_elo, elo_diff to both played and fixture frames."""
    out = {}
    for name, df in (("played", played), ("fixtures", fixtures)):
        d = df.copy()
        d["__rid__"] = range(len(d))
        home = _asof_elo(d, elo, "home_team", "home_elo")
        away = _asof_elo(d, elo, "away_team", "away_elo")
        d["home_elo"] = d["__rid__"].map(home)
        d["away_elo"] = d["__rid__"].map(away)
        d["elo_diff"] = d["home_elo"] - d["away_elo"]
        d = d.drop(columns="__rid__")
        out[name] = d
    return out["played"], out["fixtures"]


# --------------------------------------------------------------------------- #
# Rolling form features (strictly past matches only)
# --------------------------------------------------------------------------- #
def _build_team_match_long(played: pd.DataFrame) -> pd.DataFrame:
    """Explode played matches into one row per team-appearance (long format).

    Each played match yields two rows (home and away perspective) carrying that
    team's goals-for/against and a win flag. Used to compute rolling form.
    """
    home = pd.DataFrame(
        {
            "match_id": played["match_id"].values,
            "date": played["date"].values,
            "team": played["home_team"].values,
            "gf": played["home_score"].values,
            "ga": played["away_score"].values,
            "win": (played["home_score"] > played["away_score"]).astype("int64").values,
            "side": "home",
        }
    )
    away = pd.DataFrame(
        {
            "match_id": played["match_id"].values,
            "date": played["date"].values,
            "team": played["away_team"].values,
            "gf": played["away_score"].values,
            "ga": played["home_score"].values,
            "win": (played["away_score"] > played["home_score"]).astype("int64").values,
            "side": "away",
        }
    )
    long = pd.concat([home, away], ignore_index=True)
    return long.sort_values(["team", "date", "match_id"]).reset_index(drop=True)


def _add_state_columns(long: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add trailing rolling means (inclusive of current row) per team.

    ``state_*`` columns summarize the team's last ``window`` matches up to and
    INCLUDING the current row. Form going *into* a match is derived from these
    by shifting (played) or by an as-of join (fixtures).
    """
    df = long.copy()
    g = df.groupby("team", sort=False)
    for src, dst in (("win", "state_win"), ("gf", "state_gf"), ("ga", "state_ga")):
        df[dst] = g[src].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
    return df


def add_form_features(
    played: pd.DataFrame, fixtures: pd.DataFrame, window: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach last-N rolling form features to played and fixture frames.

    Adds, for each of home/away:
      *_winrate_last5, *_goals_for_last5, *_goals_against_last5, *_goal_diff_last5
    """
    played = played.copy()
    played["match_id"] = range(len(played))

    long = _build_team_match_long(played)
    long = _add_state_columns(long, window)

    # --- Played rows: form = team's state at the PREVIOUS match (shift) ---
    g = long.groupby("team", sort=False)
    long["f_winrate"] = g["state_win"].shift(1)
    long["f_gf"] = g["state_gf"].shift(1)
    long["f_ga"] = g["state_ga"].shift(1)

    def _merge_side(matches: pd.DataFrame, side: str) -> pd.DataFrame:
        cols = ["match_id", "f_winrate", "f_gf", "f_ga"]
        side_long = long.loc[long["side"] == side, cols]
        prefix = "home" if side == "home" else "away"
        side_long = side_long.rename(
            columns={
                "f_winrate": f"{prefix}_winrate_last5",
                "f_gf": f"{prefix}_goals_for_last5",
                "f_ga": f"{prefix}_goals_against_last5",
            }
        )
        return matches.merge(side_long, on="match_id", how="left")

    played = _merge_side(played, "home")
    played = _merge_side(played, "away")
    played["home_goal_diff_last5"] = (
        played["home_goals_for_last5"] - played["home_goals_against_last5"]
    )
    played["away_goal_diff_last5"] = (
        played["away_goals_for_last5"] - played["away_goals_against_last5"]
    )
    played = played.drop(columns="match_id")

    # --- Fixtures: form = latest team state strictly before the fixture date ---
    state = (
        long[["team", "date", "state_win", "state_gf", "state_ga"]]
        .sort_values("date")
        .reset_index(drop=True)
    )

    def _asof_form(fx: pd.DataFrame, team_col: str, prefix: str) -> pd.DataFrame:
        fx = fx.copy()
        fx["__rid__"] = range(len(fx))
        base = fx[["__rid__", "date", team_col]].sort_values("date")
        right = state.rename(
            columns={
                "team": team_col,
                "state_win": f"{prefix}_winrate_last5",
                "state_gf": f"{prefix}_goals_for_last5",
                "state_ga": f"{prefix}_goals_against_last5",
            }
        )
        merged = pd.merge_asof(
            base,
            right,
            on="date",
            by=team_col,
            direction="backward",
            allow_exact_matches=False,  # strictly before fixture date
        ).set_index("__rid__")
        for c in (
            f"{prefix}_winrate_last5",
            f"{prefix}_goals_for_last5",
            f"{prefix}_goals_against_last5",
        ):
            fx[c] = fx["__rid__"].map(merged[c])
        return fx.drop(columns="__rid__")

    fixtures = _asof_form(fixtures, "home_team", "home")
    fixtures = _asof_form(fixtures, "away_team", "away")
    fixtures["home_goal_diff_last5"] = (
        fixtures["home_goals_for_last5"] - fixtures["home_goals_against_last5"]
    )
    fixtures["away_goal_diff_last5"] = (
        fixtures["away_goals_for_last5"] - fixtures["away_goals_against_last5"]
    )
    return played, fixtures


# --------------------------------------------------------------------------- #
# Column selection
# --------------------------------------------------------------------------- #
ID_COLS = ["date", "home_team", "away_team", "tournament"]

FEATURE_COLS = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_winrate_last5",
    "away_winrate_last5",
    "home_goals_for_last5",
    "away_goals_for_last5",
    "home_goals_against_last5",
    "away_goals_against_last5",
    "home_goal_diff_last5",
    "away_goal_diff_last5",
    "neutral",
    "tournament_weight",
]

TARGET_COLS = ["result", "home_win", "draw", "away_win"]


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full Stage 2 pipeline; return (train_features, fixtures_features)."""
    config.ensure_dirs()

    played = pd.read_parquet(config.INTERIM_FILES["results_played"])
    fixtures = pd.read_parquet(config.INTERIM_FILES["fixtures_2026"])
    elo = pd.read_parquet(config.INTERIM_FILES["eloratings_clean"])

    played = add_targets(played)

    played, fixtures = add_elo_features(played, fixtures, elo)
    played, fixtures = add_form_features(played, fixtures, config.FORM_WINDOW)
    played = add_context_features(played)
    fixtures = add_context_features(fixtures)

    train = played[ID_COLS + FEATURE_COLS + TARGET_COLS].copy()
    fix = fixtures[ID_COLS + FEATURE_COLS].copy()

    train.to_parquet(config.PROCESSED_FILES["train_features"], index=False)
    fix.to_parquet(config.PROCESSED_FILES["fixtures_2026_features"], index=False)

    return train, fix


def print_summary(train: pd.DataFrame, fix: pd.DataFrame) -> None:
    """Print row counts, missing values, and the final feature columns."""
    print("=" * 70)
    print("STAGE 2 FEATURE SUMMARY")
    print("=" * 70)
    print(f"train_features rows         : {len(train):,}")
    print(f"fixtures_2026_features rows : {len(fix):,}")

    print("\n--- target distribution (train) ---")
    print(train["result"].value_counts(dropna=False).to_string())

    print("\n--- missing values: train_features ---")
    print(train.isna().sum().to_string())
    print("\n--- missing values: fixtures_2026_features ---")
    print(fix.isna().sum().to_string())

    print("\n--- final feature columns ---")
    print("id      :", ID_COLS)
    print("features:", FEATURE_COLS)
    print("targets :", TARGET_COLS)

    print(f"\nSaved -> {config.PROCESSED_FILES['train_features']}")
    print(f"Saved -> {config.PROCESSED_FILES['fixtures_2026_features']}")


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    train, fix = build_features()
    print_summary(train, fix)
    return train, fix


if __name__ == "__main__":
    run()
