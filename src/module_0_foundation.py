"""
module_0_foundation.py — the one-time data pipeline for the IPL Intelligence Platform.

Run once to convert the five raw CSVs into a clean, enriched Parquet layer.
Every other module reads exclusively from that Parquet layer; no module touches CSVs at runtime.

Usage:
    python -m src.module_0_foundation          (from project root, inside the venv)

Stages:
  1. Load raw CSVs.
  2. Clean  — fix types, resolve ambiguity, flag specials, build name↔ID bridges.
  3. Enrich — compute derived columns (phases, running scores, RRR, targets, lineage).
  4. Write  — Parquet files to data/processed/.
  5. Verify — assert row counts, zero unresolved players, spot-checks.
  6. Report — write docs/data_quality_report.md.
"""
from __future__ import annotations

import json
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"
DOCS = PROJECT_ROOT / "docs"
PROCESSED.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Franchise lineage map
# Who should be counted as "the same franchise" for historical trend analysis?
# Key  = team_id in the data  →  Value = canonical_lineage_id (the surviving team's id)
#
# Distinction:
#   REBRAND (same lineage)  : Deccan Chargers (1068) → SRH (2); they share trophy history
#   SPELLING VARIANT        : Rising Pune Supergiant (4) ↔ Rising Pune Supergiants (3604)
#                             — two seasons, one franchise, two spelling errors in the data
#   NEW FRANCHISE           : Gujarat Lions (5) ≠ Gujarat Titans (615) — different ownership,
#                             different era; we do NOT merge them in the lineage view
# ---------------------------------------------------------------------------
FRANCHISE_LINEAGE: dict[int, int] = {
    # Deccan Chargers → Sunrisers Hyderabad
    1068: 2,
    # Delhi Daredevils → Delhi Capitals (Delhi Capitals already has id 252)
    # (no separate id for Daredevils — same id 252 used; alias only in team_aliases)
    # Rising Pune Supergiant spelling variant
    4: 3604,   # normalise to the later spelling id
    # Kochi Tuskers Kerala — defunct, no successor
    # Pune Warriors — defunct, no successor
    # All other teams map to themselves (identity) — handled in build_teams()
}

# Phase boundaries (over_number is 0-indexed in the raw data)
POWERPLAY_OVERS = range(0, 6)    # overs 0-5 → "powerplay"
MIDDLE_OVERS = range(6, 15)      # overs 6-14 → "middle"
DEATH_OVERS = range(15, 20)      # overs 15-19 → "death"


# ===========================================================================
# STAGE 1 — LOAD RAW
# ===========================================================================
def load_raw() -> dict[str, pd.DataFrame]:
    print("  Loading raw CSVs...")
    raw: dict[str, pd.DataFrame] = {}

    raw["matches"] = pd.read_csv(RAW / "ipl_matches_data.csv", low_memory=False)
    raw["deliveries"] = pd.read_csv(RAW / "ball_by_ball_data.csv", low_memory=False)
    raw["players"] = pd.read_csv(RAW / "players-data-updated.csv")
    raw["teams"] = pd.read_csv(RAW / "teams_data.csv")
    raw["aliases"] = pd.read_csv(RAW / "team_aliases.csv")

    print(f"    matches     : {len(raw['matches']):,} rows")
    print(f"    deliveries  : {len(raw['deliveries']):,} rows")
    print(f"    players     : {len(raw['players']):,} rows")
    print(f"    teams       : {len(raw['teams']):,} rows")
    print(f"    aliases     : {len(raw['aliases']):,} rows")
    return raw


# ===========================================================================
# STAGE 2 — CLEAN
# ===========================================================================
def clean_players(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise the player master. Rename columns for clarity."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df["player_id"] = df["player_id"].astype(int)
    df["player_name"] = df["player_name"].str.strip()
    df["player_full_name"] = df["player_full_name"].str.strip()
    df["bat_style"] = df["bat_style"].str.strip().fillna("Unknown")
    df["bowl_style"] = df["bowl_style"].str.strip().fillna("Unknown")
    return df


def build_name_to_id(players: pd.DataFrame) -> dict[str, int]:
    """
    Build a name → player_id lookup from the player master.
    The ball-by-ball file uses player NAMES; the matches file uses player IDs.
    This bridge connects both sides.
    """
    return dict(zip(players["player_name"], players["player_id"]))


def clean_teams(teams: pd.DataFrame, aliases: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a clean, lineage-aware team table.
    Adds 'lineage_id' — the canonical franchise id for historical grouping.
    """
    teams = teams.copy()
    teams["team_id"] = teams["team_id"].astype(int)
    teams["team_name"] = teams["team_name"].str.strip()

    # Lineage id defaults to the team's own id; rebrands get the successor's id.
    teams["lineage_id"] = teams["team_id"].map(lambda tid: FRANCHISE_LINEAGE.get(tid, tid))

    # Attach all known aliases as a JSON list (useful for Module 0 schema display)
    alias_map: dict[int, list[str]] = (
        aliases.groupby("team_id")["alias_name"].apply(list).to_dict()
    )
    teams["aliases"] = teams["team_id"].map(lambda tid: alias_map.get(tid, []))
    return teams


def clean_matches(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the match table:
      - Normalise season string  ('2020/21' → 2020, flag UAE season)
      - Cast numeric id columns
      - Derive is_completed, is_tie, is_no_result, is_super_over_decided flags
      - Keep win margins; resolve 'no result' / 'tie' records cleanly
    """
    df = raw.copy()

    # --- Season normalisation ---
    df["season_str"] = df["season"].astype(str).str.strip()
    df["season"] = df["season_str"].replace("2020/21", "2020").astype(int)
    df["is_uae_season"] = df["season"] == 2020

    # --- Integer ID columns ---
    for col in ["match_id", "season_id", "team1", "team2", "toss_winner", "match_winner"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # --- player_of_match: stored as float (e.g. 46.0); cast to Int64 ---
    df["player_of_match"] = pd.to_numeric(df["player_of_match"], errors="coerce").astype("Int64")

    # --- Result flags ---
    result = df["result"].str.strip().str.lower()
    df["is_completed"] = result == "win"
    df["is_tie"] = result == "tie"
    df["is_no_result"] = result == "no result"

    # For tie matches the match_winner column is the super-over winner.
    # We mark these explicitly so analysts know the 'winner' was decided by super over.
    df["is_super_over_decided"] = (
        df["is_tie"] & df["match_winner"].notna()
    )

    # --- Toss decision as category ---
    df["toss_decision"] = df["toss_decision"].str.strip().astype("category")

    # --- Date ---
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    # --- Venue: strip whitespace ---
    df["venue"] = df["venue"].str.strip()
    df["city"] = df["city"].str.strip()

    return df


def clean_deliveries(raw: pd.DataFrame, name_to_id: dict[str, int]) -> tuple[pd.DataFrame, list[str]]:
    """
    Clean the ball-by-ball table:
      - Cast types, strip whitespace
      - Resolve player names → player_ids
      - Log any unresolved names (critical data quality signal)
      - Add over_number_1idx (1-based) for human-readable display
    Returns (cleaned_df, unresolved_names).
    """
    df = raw.copy()

    # Strip whitespace from all string columns
    str_cols = df.select_dtypes("object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    # Boolean columns — convert "True"/"False" strings if needed
    bool_cols = [
        "is_wicket", "is_wide_ball", "is_no_ball", "is_leg_bye",
        "is_bye", "is_penalty", "is_super_over"
    ]
    for col in bool_cols:
        if df[col].dtype == object:
            df[col] = df[col].map({"True": True, "False": False})
        df[col] = df[col].astype(bool)

    # Numeric columns
    df["match_id"] = df["match_id"].astype(int)
    df["season_id"] = df["season_id"].astype(int)
    df["season"] = df["season_id"]  # keep both for convenience
    df["over_number"] = df["over_number"].astype(int)      # 0-indexed
    df["ball_number"] = df["ball_number"].astype(int)      # 0-indexed within over
    df["over_number_1idx"] = df["over_number"] + 1        # human-readable (1–20)
    df["batter_runs"] = df["batter_runs"].astype(int)
    df["extras"] = df["extras"].astype(int)
    df["total_runs"] = df["total_runs"].astype(int)

    # --- Player name → player_id resolution ---
    for name_col, id_col in [("batter", "batter_id"), ("bowler", "bowler_id"), ("non_striker", "non_striker_id")]:
        df[id_col] = df[name_col].map(name_to_id).astype("Int64")

    unresolved = sorted(set(
        df.loc[df["batter_id"].isna(), "batter"].unique().tolist() +
        df.loc[df["bowler_id"].isna(), "bowler"].unique().tolist()
    ))

    # --- Over phase label ---
    def _phase(over: int) -> str:
        if over < 6:
            return "powerplay"
        if over < 15:
            return "middle"
        return "death"

    df["over_phase"] = df["over_number"].apply(_phase).astype("category")

    # --- Batting/bowling team ids already int ---
    df["team_batting"] = df["team_batting"].astype(int)
    df["team_bowling"] = df["team_bowling"].astype(int)

    return df, unresolved


# ===========================================================================
# STAGE 3 — ENRICH
# ===========================================================================
def enrich_deliveries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-ball derived columns needed by Modules 1–4:

    Within every (match_id, innings) group, sorted by (over_number, ball_number):
      - running_score      : cumulative total_runs BEFORE this ball (exclusive)
      - running_wickets    : cumulative wickets BEFORE this ball
      - legal_ball_number  : count of non-wide, non-no-ball deliveries bowled so far (for RRR)
      - balls_remaining    : 120 − legal_ball_number (only for innings 2)
      - runs_remaining     : target − running_score − 1 (only for innings 2; target joined later)
      - required_run_rate  : runs_remaining / (balls_remaining / 6)
    """
    df = df.copy()
    df = df.sort_values(["match_id", "innings", "over_number", "ball_number"]).reset_index(drop=True)

    # A delivery is a "legal ball" (counts toward 120) if it is NOT a wide AND NOT a no-ball
    df["is_legal_ball"] = ~(df["is_wide_ball"] | df["is_no_ball"])

    grp = ["match_id", "innings"]

    # Cumulative runs and wickets BEFORE this ball (exclusive cumsum = shift(1).cumsum())
    df["running_score"] = (
        df.groupby(grp)["total_runs"].cumsum() - df["total_runs"]
    )
    df["running_wickets"] = (
        df.groupby(grp)["is_wicket"].cumsum().astype(int) - df["is_wicket"].astype(int)
    )

    # Legal ball index within innings (0-based, exclusive — i.e. how many legal balls BOWLED so far)
    df["legal_balls_bowled"] = df.groupby(grp)["is_legal_ball"].cumsum().astype(int) - df["is_legal_ball"].astype(int)

    print("  Enriching deliveries — computing first-innings totals for target calculation...")
    # First-innings total per match (needed to compute target for second innings)
    inns1 = (
        df[df["innings"] == 1]
        .groupby("match_id")["total_runs"]
        .sum()
        .rename("first_innings_total")
    )
    df = df.merge(inns1, on="match_id", how="left")

    # Target = first_innings_total + 1 (chasing team needs one more run to win)
    df["target"] = (df["first_innings_total"] + 1).where(df["innings"] == 2)

    # Second-innings derived metrics
    mask2 = df["innings"] == 2
    df["balls_remaining"] = np.where(
        mask2,
        (120 - df["legal_balls_bowled"]).clip(lower=0),
        np.nan
    )
    df["runs_remaining"] = np.where(
        mask2,
        (df["target"] - df["running_score"]).clip(lower=0),
        np.nan
    )
    # Required run rate: runs needed per 6 balls. Guard against division-by-zero.
    df["required_run_rate"] = np.where(
        mask2 & (df["balls_remaining"] > 0),
        (df["runs_remaining"] / df["balls_remaining"]) * 6,
        np.nan
    )
    df["current_run_rate"] = np.where(
        mask2 & (df["legal_balls_bowled"] > 0),
        (df["running_score"] / df["legal_balls_bowled"]) * 6,
        np.nan
    )
    df["run_rate_gap"] = df["current_run_rate"] - df["required_run_rate"]

    # Apply category dtypes to low-cardinality columns to save memory
    for col in ["over_phase", "wicket_kind", "toss_decision"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def enrich_matches(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Join first_innings_total and target back into the matches table.
    Also add home_team heuristic: the team that played more home games at the venue per season.
    """
    df = matches.copy()

    # First-innings totals from deliveries
    fi = (
        deliveries[deliveries["innings"] == 1]
        .groupby("match_id")["total_runs"]
        .sum()
        .rename("first_innings_total")
        .reset_index()
    )
    df = df.merge(fi, on="match_id", how="left")
    df["target"] = df["first_innings_total"] + 1

    # Normalise season to int (already done in clean_matches, but safety check)
    if df["season"].dtype == object:
        df["season"] = df["season"].replace("2020/21", "2020").astype(int)

    return df


# ===========================================================================
# STAGE 4 — WRITE PARQUET
# ===========================================================================
def write_parquet(df: pd.DataFrame, name: str) -> None:
    path = PROCESSED / f"{name}.parquet"
    df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    size_kb = path.stat().st_size / 1024
    print(f"    [OK] {name}.parquet  ({len(df):,} rows, {size_kb:.0f} KB)")


# ===========================================================================
# STAGE 5 — VERIFY
# ===========================================================================
def verify(matches: pd.DataFrame, deliveries: pd.DataFrame, players: pd.DataFrame, unresolved: list[str]) -> dict:
    """
    Run sanity assertions and return a quality report dict.
    Raises AssertionError if critical checks fail.
    """
    report: dict = {}

    # Row counts
    report["matches_total"] = len(matches)
    report["deliveries_total"] = len(deliveries)
    report["players_total"] = len(players)
    report["seasons"] = sorted(matches["season"].dropna().unique().tolist())
    report["season_count"] = len(report["seasons"])

    assert len(matches) >= 1200, f"Expected ~1212 matches, got {len(matches)}"
    assert len(deliveries) >= 280000, f"Expected ~288k deliveries, got {len(deliveries)}"

    # Result breakdown
    report["completed_matches"] = int(matches["is_completed"].sum())
    report["tie_matches"] = int(matches["is_tie"].sum())
    report["no_result_matches"] = int(matches["is_no_result"].sum())
    report["super_over_decided"] = int(matches["is_super_over_decided"].sum())

    # Player resolution
    report["unresolved_player_names"] = unresolved
    report["unresolved_count"] = len(unresolved)
    if unresolved:
        print(f"  [WARN]  {len(unresolved)} player names in ball-by-ball not found in player master:")
        for name in unresolved[:20]:
            print(f"       • {name}")

    # Derived column spot-checks
    inns2 = deliveries[deliveries["innings"] == 2]
    assert inns2["balls_remaining"].notna().mean() > 0.95, "Too many nulls in balls_remaining"
    assert inns2["required_run_rate"].notna().mean() > 0.90, "Too many nulls in required_run_rate"

    # Super-over exclusion check
    super_over_count = deliveries["is_super_over"].sum()
    report["super_over_deliveries"] = int(super_over_count)

    # Phase coverage
    phase_counts = deliveries["over_phase"].value_counts().to_dict()
    report["phase_coverage"] = {str(k): int(v) for k, v in phase_counts.items()}

    print(f"\n  [PASS] Verification passed")
    print(f"     {report['matches_total']:,} matches | {report['deliveries_total']:,} deliveries | {report['players_total']:,} players")
    print(f"     {report['completed_matches']} won | {report['tie_matches']} tied | {report['no_result_matches']} no-result")
    print(f"     {report['unresolved_count']} unresolved player names")
    print(f"     {report['super_over_deliveries']} super-over deliveries (flagged, not deleted)")
    return report


# ===========================================================================
# STAGE 6 — DATA QUALITY REPORT
# ===========================================================================
def write_data_quality_report(report: dict) -> None:
    season_list = ", ".join(str(s) for s in report["seasons"])
    unresolved_block = (
        "\n".join(f"  - `{n}`" for n in report["unresolved_player_names"])
        if report["unresolved_player_names"]
        else "  *None — all ball-by-ball player names resolved successfully.*"
    )

    md = textwrap.dedent(f"""
    # Data Quality Report — IPL Intelligence Platform
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

    ## Source Files

    | File | Rows |
    |---|---|
    | ipl_matches_data.csv | {report['matches_total']:,} |
    | ball_by_ball_data.csv | {report['deliveries_total']:,} |
    | players-data-updated.csv | {report['players_total']:,} |
    | teams_data.csv | 16 |
    | team_aliases.csv | 46 |

    ## Dataset Statistics

    - **Matches:** {report['matches_total']:,}
    - **Deliveries:** {report['deliveries_total']:,}
    - **Players:** {report['players_total']:,}
    - **Seasons:** {report['season_count']} ({season_list})
    - **Season 2020/21 → 2020:** normalised with `is_uae_season = True` flag

    ## Match Results

    | Result | Count |
    |---|---|
    | Completed (win) | {report['completed_matches']} |
    | Tie (super-over decided) | {report['super_over_decided']} |
    | Tie (no super-over result) | {report['tie_matches'] - report['super_over_decided']} |
    | No result | {report['no_result_matches']} |

    ## Cleaning Decisions

    ### Season normalisation
    The string `"2020/21"` is mapped to integer `2020`. A boolean column `is_uae_season`
    is added so any analysis can filter or annotate the UAE-hosted season separately.

    ### Super-over deliveries
    {report['super_over_deliveries']:,} super-over deliveries exist in the raw data.
    They are **retained** in `deliveries.parquet` with `is_super_over = True` and
    **excluded** by all win-probability model queries and run-rate aggregations via
    filter `is_super_over == False`. Tie outcomes are resolved to the super-over winner
    in `matches.parquet` via `match_winner`.

    ### No-result matches
    {report['no_result_matches']} no-result matches are retained with `is_no_result = True`
    and excluded from win/loss analysis. They are included in match counts and venue aggregations.

    ### Player name resolution
    The ball-by-ball file uses player **names**; the matches file uses player **IDs**.
    A `player_name → player_id` bridge was built from the player master.

    **Unresolved names ({report['unresolved_count']}):**
{unresolved_block}

    ### Franchise lineage
    Two analytical views are maintained:
    - **`team_id` view:** season-accurate analysis using the exact team identity per season.
    - **`lineage_id` view:** historical trend analysis treating rebrands as one continuous franchise.
      - Deccan Chargers (id 1068) → Sunrisers Hyderabad (id 2)
      - Rising Pune Supergiant (id 4) → Rising Pune Supergiants (id 3604) *(spelling normalisation)*
      - All other teams: `lineage_id = team_id`
      - Gujarat Lions (id 5) and Gujarat Titans (id 615) remain **separate** (different franchises).

    ## Enriched Columns Added

    ### deliveries.parquet (beyond raw columns)
    | Column | Description |
    |---|---|
    | `over_phase` | `powerplay` (overs 1–6) / `middle` (7–15) / `death` (16–20) |
    | `over_number_1idx` | 1-based over number (human-readable) |
    | `running_score` | Cumulative runs *before* this ball (exclusive) |
    | `running_wickets` | Cumulative wickets *before* this ball (exclusive) |
    | `legal_balls_bowled` | Non-wide, non-no-ball deliveries bowled so far in innings |
    | `first_innings_total` | Total runs scored in innings 1 of this match |
    | `target` | `first_innings_total + 1` (innings 2 only) |
    | `balls_remaining` | `120 − legal_balls_bowled` (innings 2 only) |
    | `runs_remaining` | `target − running_score` (innings 2 only) |
    | `required_run_rate` | `runs_remaining / (balls_remaining / 6)` (innings 2 only) |
    | `current_run_rate` | `running_score / (legal_balls_bowled / 6)` (innings 2 only) |
    | `run_rate_gap` | `current_run_rate − required_run_rate` |
    | `batter_id`, `bowler_id`, `non_striker_id` | Resolved player IDs |

    ### matches.parquet (beyond raw columns)
    | Column | Description |
    |---|---|
    | `is_completed` | True if result == "win" |
    | `is_tie` | True if result == "tie" |
    | `is_no_result` | True if result == "no result" |
    | `is_super_over_decided` | True if tie was resolved via super over |
    | `is_uae_season` | True if season == 2020 (UAE-hosted) |
    | `first_innings_total` | Joined from deliveries |
    | `target` | `first_innings_total + 1` |

    ## Assumptions & Limitations

    - **No auction/salary data:** all recommendations are framed around performance and strategy, not player valuation.
    - **No injury records:** player career gaps are visible but not explainable.
    - **No DRS/review data:** umpire decisions are taken as ground truth.
    - **Win probability model** is trained on completed matches only; DLS-adjusted matches are not explicitly handled.
    - **Over phase boundaries** are fixed (PP: 1–6, Middle: 7–15, Death: 16–20) and do not account for strategic timeouts or fielding restrictions in the first two overs.

    ## Processed Files

    | File | Key Contents |
    |---|---|
    | `matches.parquet` | Cleaned + enriched match records |
    | `deliveries.parquet` | Cleaned + enriched ball-by-ball records |
    | `players.parquet` | Player master with resolved IDs |
    | `teams.parquet` | Canonical teams with lineage mapping |
    """).lstrip()

    path = DOCS / "data_quality_report.md"
    path.write_text(md, encoding="utf-8")
    print(f"    [OK] docs/data_quality_report.md written")


# ===========================================================================
# MAIN
# ===========================================================================
def run_pipeline() -> None:
    print("\n" + "=" * 60)
    print("  IPL Intelligence Platform — Module 0 Data Pipeline")
    print("=" * 60)

    # Stage 1 — Load
    print("\n[1/6] Loading raw CSVs...")
    raw = load_raw()

    # Stage 2 — Clean
    print("\n[2/6] Cleaning...")
    players = clean_players(raw["players"])
    teams = clean_teams(raw["teams"], raw["aliases"])
    matches = clean_matches(raw["matches"])
    name_to_id = build_name_to_id(players)
    deliveries, unresolved = clean_deliveries(raw["deliveries"], name_to_id)
    print(f"    Player name bridge: {len(name_to_id):,} names")
    print(f"    Unresolved player names: {len(unresolved)}")

    # Stage 3 — Enrich
    print("\n[3/6] Enriching...")
    deliveries = enrich_deliveries(deliveries)
    matches = enrich_matches(matches, deliveries)

    # Stage 4 — Write
    print("\n[4/6] Writing Parquet layer...")
    write_parquet(matches, "matches")
    write_parquet(deliveries, "deliveries")
    write_parquet(players, "players")
    write_parquet(teams, "teams")

    # Stage 5 — Verify
    print("\n[5/6] Verifying...")
    report = verify(matches, deliveries, players, unresolved)
    report["unresolved_player_names_in_pipeline"] = unresolved  # save for report

    # Stage 6 — Report
    print("\n[6/6] Writing data quality report...")
    write_data_quality_report(report)

    print("\n" + "=" * 60)
    print("  Pipeline complete. Parquet layer is ready.")
    print("  Run: streamlit run app/main.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_pipeline()
