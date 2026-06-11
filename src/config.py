"""Central configuration: paths and shared constants for the project.

Stage 1 only uses the path definitions and a few cleaning constants. Keeping
everything here means downstream stages (features, models, simulation) can import
the same canonical paths instead of hard-coding strings.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Project root = parent of the `src/` directory that contains this file.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"

NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
REPORT_DIR: Path = PROJECT_ROOT / "report"
LINKEDIN_DIR: Path = OUTPUTS_DIR / "linkedin_visuals"
# Stage 10 final public-facing visuals (kept separate from Stage 8 outputs).
FINAL_LINKEDIN_DIR: Path = OUTPUTS_DIR / "final_linkedin_visuals"

# --------------------------------------------------------------------------- #
# Raw input files
# --------------------------------------------------------------------------- #
RAW_FILES: dict[str, Path] = {
    "results": RAW_DIR / "results.csv",
    "goalscorers": RAW_DIR / "goalscorers.csv",
    "shootouts": RAW_DIR / "shootouts.csv",
    "eloratings": RAW_DIR / "eloratings.csv",
    "former_names": RAW_DIR / "former_names.csv",
}

# --------------------------------------------------------------------------- #
# Interim (cleaned) output files
# --------------------------------------------------------------------------- #
INTERIM_FILES: dict[str, Path] = {
    # results.csv split into played history vs. unplayed 2026 fixtures
    "results_played": INTERIM_DIR / "results_played.parquet",
    "fixtures_2026": INTERIM_DIR / "fixtures_2026.parquet",
    "eloratings_clean": INTERIM_DIR / "eloratings_clean.parquet",
    "goalscorers_clean": INTERIM_DIR / "goalscorers_clean.parquet",
    "shootouts_clean": INTERIM_DIR / "shootouts_clean.parquet",
    "former_names_clean": INTERIM_DIR / "former_names_clean.parquet",
    # human-readable summary of every dataset
    "summary": INTERIM_DIR / "dataset_summary.csv",
}

# --------------------------------------------------------------------------- #
# Processed (feature) output files
# --------------------------------------------------------------------------- #
PROCESSED_FILES: dict[str, Path] = {
    "train_features": PROCESSED_DIR / "train_features.parquet",
    "fixtures_2026_features": PROCESSED_DIR / "fixtures_2026_features.parquet",
}

# --------------------------------------------------------------------------- #
# Output (model artifacts & reports) files
# --------------------------------------------------------------------------- #
OUTPUT_FILES: dict[str, Path] = {
    "model_results": OUTPUTS_DIR / "model_results.csv",
    "confusion_matrix": OUTPUTS_DIR / "confusion_matrix.png",
    "best_model": OUTPUTS_DIR / "best_match_model.pkl",
    "fixtures_2026_predictions": OUTPUTS_DIR / "fixtures_2026_predictions.csv",
    "champion_probabilities": OUTPUTS_DIR / "champion_probabilities.csv",
    "advancement_probabilities": OUTPUTS_DIR / "advancement_probabilities.csv",
    # Stage 9 model-improvement artifacts (kept separate from Stage 3)
    "model_results_stage9": OUTPUTS_DIR / "model_results_stage9.csv",
    "best_match_model_stage9": OUTPUTS_DIR / "best_match_model_stage9.pkl",
    "confusion_matrix_stage9": OUTPUTS_DIR / "confusion_matrix_stage9.png",
    # Stage 10 final public-facing deployment artifacts ("final_" convention)
    "final_model": OUTPUTS_DIR / "final_match_model.pkl",
    "final_model_metrics": OUTPUTS_DIR / "final_model_metrics.csv",
    "final_confusion_matrix": OUTPUTS_DIR / "final_confusion_matrix.png",
    "final_fixtures_predictions": OUTPUTS_DIR / "final_fixtures_2026_predictions.csv",
    "final_champion_probabilities": OUTPUTS_DIR / "final_champion_probabilities.csv",
    "final_advancement_probabilities": OUTPUTS_DIR / "final_advancement_probabilities.csv",
}

# --------------------------------------------------------------------------- #
# Simulation constants (Stage 5)
# --------------------------------------------------------------------------- #
# Number of Monte Carlo tournament simulations.
N_SIMULATIONS: int = 10_000

# Group-stage points.
POINTS_WIN: int = 3
POINTS_DRAW: int = 1
POINTS_LOSS: int = 0

# Number of best third-placed teams that advance (WC 2026 format).
N_BEST_THIRDS: int = 8

# Official FIFA 2026 group letters, keyed by each group's Pot 1 seed team. Groups
# are reconstructed from fixture pairings; this maps each cluster to its real
# A-L letter (every group contains exactly one Pot 1 seed). Falls back to
# alphabetical labelling if a seed is missing.
OFFICIAL_GROUP_SEEDS: dict[str, str] = {
    "Mexico": "A",
    "Canada": "B",
    "Brazil": "C",
    "United States": "D",
    "Germany": "E",
    "Netherlands": "F",
    "Belgium": "G",
    "Spain": "H",
    "France": "I",
    "Argentina": "J",
    "Portugal": "K",
    "England": "L",
}

# --------------------------------------------------------------------------- #
# Feature-engineering constants
# --------------------------------------------------------------------------- #
# Window size (number of previous matches) for rolling "form" features.
FORM_WINDOW: int = 5

# Leakage-safe model input features (Stage 3).
MODEL_FEATURES: list[str] = [
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

# Elo columns that may be missing and get an explicit missingness indicator.
ELO_FEATURES: list[str] = ["home_elo", "away_elo", "elo_diff"]

# Chronological split boundaries for Stage 3 training/evaluation.
TRAIN_START: str = "2010-01-01"
TRAIN_END: str = "2022-12-31"
TEST_START: str = "2023-01-01"
TEST_END: str = "2025-12-31"

# Relative importance weight per tournament type. Looked up by exact name first,
# then by substring rules in ``features.tournament_weight``. Higher = more
# competitive / informative.
TOURNAMENT_WEIGHTS: dict[str, float] = {
    "FIFA World Cup": 1.0,
    "Confederations Cup": 0.85,
    "UEFA Euro": 0.80,
    "Copa América": 0.80,
    "African Cup of Nations": 0.80,
    "AFC Asian Cup": 0.80,
    "Gold Cup": 0.80,
    "UEFA Nations League": 0.75,
    "CONCACAF Nations League": 0.75,
    "Friendly": 0.20,
}
DEFAULT_TOURNAMENT_WEIGHT: float = 0.50

# --------------------------------------------------------------------------- #
# Cleaning constants
# --------------------------------------------------------------------------- #
# Matches with a date on/after this cutoff that still have no score are treated
# as unplayed fixtures (prediction targets), not training rows.
FIXTURE_CUTOFF: str = "2026-01-01"

# Minimum plausible Elo rating; values <= this (incl. 0) are considered invalid.
MIN_VALID_ELO: float = 0.0

# Team-name alias map -> canonical name (the spelling used in results.csv, the
# spine dataset). Applied after whitespace normalization to every team column so
# joins across datasets line up. Keys are the variant spellings, values the
# canonical names. See Stage 7 (non-breaking-space / naming bug fix).
TEAM_ALIASES: dict[str, str] = {
    "Czechia": "Czech Republic",
    "Democratic Republic of Congo": "DR Congo",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
    "USA": "United States",
}

# Random seed reused across the project for reproducibility.
RANDOM_SEED: int = 42


def ensure_dirs() -> None:
    """Create all project directories if they do not already exist."""
    for d in (
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        NOTEBOOKS_DIR,
        OUTPUTS_DIR,
        REPORT_DIR,
        LINKEDIN_DIR,
        FINAL_LINKEDIN_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
