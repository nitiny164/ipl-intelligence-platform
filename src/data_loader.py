"""
data_loader.py — the single source of truth for READING the processed Parquet layer.

Every module (src/module_*.py) and every Streamlit page imports its data from here.
No other file should call pd.read_parquet directly. This gives us:
  * one place that controls all file paths,
  * one place to apply Streamlit caching (so the app does not re-read disk on every click),
  * functions that still work when called from a plain script or a test (no Streamlit needed).

Design note on caching:
  Streamlit re-runs the whole script on every UI interaction. We wrap loaders in
  st.cache_data so repeated reads are instant. But we want these functions to ALSO work
  outside Streamlit (e.g. in the Module 0 pipeline verification, or a unit test). So we
  define a `cache` decorator that becomes st.cache_data *if Streamlit is available and
  running*, and otherwise is a no-op pass-through.
"""
from __future__ import annotations

from pathlib import Path
import functools
import pandas as pd

# ---------------------------------------------------------------------------
# Paths — every path in the project is derived from PROJECT_ROOT, so the code
# works no matter what directory you launch it from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"


# ---------------------------------------------------------------------------
# Optional-caching decorator.
# ---------------------------------------------------------------------------
def _make_cache():
    """Return st.cache_data if Streamlit is importable, else an identity decorator."""
    try:
        import streamlit as st

        return st.cache_data
    except Exception:
        def identity(func=None, **_kwargs):
            # Support both @cache and @cache(...) usage.
            if func is None:
                return lambda f: f
            return func

        return identity


cache = _make_cache()


def _read(name: str) -> pd.DataFrame:
    """Read a Parquet file from the processed layer by its base name (no extension)."""
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed file '{path.name}' not found. "
            f"Run the Module 0 pipeline first: python -m src.module_0_foundation"
        )
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Public loaders — one per processed table. Add more as later modules create them.
# ---------------------------------------------------------------------------
@cache
def load_matches() -> pd.DataFrame:
    """Cleaned, enriched match records (one row per match)."""
    return _read("matches")


@cache
def load_deliveries() -> pd.DataFrame:
    """Cleaned, enriched ball-by-ball records (one row per legal/illegal delivery)."""
    return _read("deliveries")


@cache
def load_players() -> pd.DataFrame:
    """Player master keyed by player_id, with name-resolution bridge to ball-by-ball."""
    return _read("players")


@cache
def load_teams() -> pd.DataFrame:
    """Canonical team records, including franchise-lineage grouping."""
    return _read("teams")


def processed_exists() -> bool:
    """True if the core processed files have been generated."""
    return all(
        (PROCESSED_DIR / f"{n}.parquet").exists()
        for n in ("matches", "deliveries", "players", "teams")
    )
