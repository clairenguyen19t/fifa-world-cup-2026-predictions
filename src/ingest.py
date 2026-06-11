"""Stage 1 - ingestion.

Loads every raw CSV into a typed pandas DataFrame and parses date columns
correctly. The only non-trivial case is ``eloratings.csv``, whose ``date``
column mixes ISO (``YYYY-MM-DD``) and US (``M/D/YYYY``) formats; a naive
``pd.to_datetime`` silently fails on ~99% of those rows, so we handle it
explicitly here.

This module performs *loading and date parsing only*. All filtering, splitting
and value cleaning lives in ``clean.py``.
"""
from __future__ import annotations

import re

import pandas as pd

from . import config

# Matches any run of whitespace (incl. the non-breaking space U+00A0, which the
# Elo dataset uses inside multi-word team names and which broke the joins).
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_name(value: object) -> object:
    """Normalize a single team name: NBSP -> space, collapse/trim, then alias.

    Returns the value unchanged if it is missing (NaN/None).
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    if pd.isna(value):
        return value
    text = str(value).replace("\u00a0", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return config.TEAM_ALIASES.get(text, text)


def normalize_team_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Apply ``normalize_name`` to the given team-name columns of ``df``."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map(normalize_name)
    return df


def parse_mixed_dates(series: pd.Series) -> pd.Series:
    """Parse a date column that may mix ISO and US (M/D/Y) formats.

    Strategy: first try strict ISO parsing; for whatever fails, fall back to
    US ``month/day/year`` parsing. This avoids day/month ambiguity that a
    single ``format='mixed'`` call can introduce, and surfaces any leftover
    unparseable values as ``NaT`` rather than guessing.
    """
    s = series.astype("string").str.strip()

    # Pass 1: ISO yyyy-mm-dd
    parsed = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")

    # Pass 2: US m/d/yyyy for the rows ISO could not handle
    remaining = parsed.isna() & s.notna()
    if remaining.any():
        parsed_us = pd.to_datetime(s[remaining], format="%m/%d/%Y", errors="coerce")
        parsed.loc[remaining] = parsed_us

    return parsed


def load_results() -> pd.DataFrame:
    df = pd.read_csv(config.RAW_FILES["results"])
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    return normalize_team_columns(df, ["home_team", "away_team"])


def load_goalscorers() -> pd.DataFrame:
    df = pd.read_csv(config.RAW_FILES["goalscorers"])
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    return normalize_team_columns(df, ["home_team", "away_team", "team"])


def load_shootouts() -> pd.DataFrame:
    df = pd.read_csv(config.RAW_FILES["shootouts"])
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    return normalize_team_columns(
        df, ["home_team", "away_team", "winner", "first_shooter"]
    )


def load_eloratings() -> pd.DataFrame:
    """Load Elo ratings, repairing the mixed-format ``date`` and team names."""
    df = pd.read_csv(config.RAW_FILES["eloratings"])
    df["date"] = parse_mixed_dates(df["date"])
    return normalize_team_columns(df, ["team"])


def load_former_names() -> pd.DataFrame:
    df = pd.read_csv(config.RAW_FILES["former_names"])
    for col in ("start_date", "end_date"):
        df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")
    return normalize_team_columns(df, ["current", "former"])


def load_all() -> dict[str, pd.DataFrame]:
    """Load every raw dataset into a name -> DataFrame mapping."""
    return {
        "results": load_results(),
        "goalscorers": load_goalscorers(),
        "shootouts": load_shootouts(),
        "eloratings": load_eloratings(),
        "former_names": load_former_names(),
    }
