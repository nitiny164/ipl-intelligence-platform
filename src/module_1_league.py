"""
module_1_league.py — League Pulse analytics functions.

Business question: "How has the IPL evolved as a product — scoring, strategy,
and competitiveness — over 18 seasons?"

All functions take DataFrames (from data_loader) and return DataFrames or dicts.
No Streamlit imports here — this module is framework-independent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Season-wise scoring evolution
# ---------------------------------------------------------------------------
def season_scoring_trends(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per season with:
      - avg_first_innings_score
      - avg_run_rate  (first innings, excl. super overs)
      - match_count
      - yoy_score_change  (% change from prior season)
      - boundary_pct      (% of legal balls that went to boundary 4 or 6)
      - six_pct           (% of legal balls hit for six)
      - dot_ball_pct      (% of legal balls that scored 0 runs to batter)
    """
    completed = matches[matches["is_completed"]].copy()

    season_scores = (
        completed.groupby("season")
        .agg(
            avg_first_innings_score=("first_innings_total", "mean"),
            match_count=("match_id", "count"),
        )
        .reset_index()
    )

    # Year-over-year scoring change
    season_scores = season_scores.sort_values("season").reset_index(drop=True)
    season_scores["yoy_score_change_pct"] = season_scores["avg_first_innings_score"].pct_change() * 100

    # Run rate and boundary stats from deliveries (first innings, legal balls, no super overs)
    d1 = deliveries[(deliveries["innings"] == 1) & (~deliveries["is_super_over"]) & (deliveries["is_legal_ball"])].copy()

    # Boundary flag: batter scored 4 or 6 off this ball
    # (batter_runs = 4 or 6; extras like overthrows are batter_runs too if awarded to batter)
    d1["is_four"] = d1["batter_runs"] == 4
    d1["is_six"] = d1["batter_runs"] == 6
    d1["is_boundary"] = d1["is_four"] | d1["is_six"]
    d1["is_dot"] = d1["batter_runs"] == 0

    delivery_stats = (
        d1.groupby("season_id")
        .agg(
            total_runs_d=("batter_runs", "sum"),
            total_balls=("batter_runs", "count"),
            boundaries=("is_boundary", "sum"),
            sixes=("is_six", "sum"),
            dots=("is_dot", "sum"),
        )
        .reset_index()
        .rename(columns={"season_id": "season"})
    )
    delivery_stats["avg_run_rate"] = (delivery_stats["total_runs_d"] / delivery_stats["total_balls"]) * 6
    delivery_stats["boundary_pct"] = (delivery_stats["boundaries"] / delivery_stats["total_balls"]) * 100
    delivery_stats["six_pct"] = (delivery_stats["sixes"] / delivery_stats["total_balls"]) * 100
    delivery_stats["dot_ball_pct"] = (delivery_stats["dots"] / delivery_stats["total_balls"]) * 100

    result = season_scores.merge(delivery_stats, on="season", how="left")
    return result


# ---------------------------------------------------------------------------
# 2. Toss decision trend
# ---------------------------------------------------------------------------
def toss_decision_trend(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per season with:
      - field_first_pct: % of tosses where captain chose to field
      - field_win_pct:   win % when fielding first (chasing)
      - bat_win_pct:     win % when batting first (defending)
      - chase_success_pct (same as field_win_pct, named for clarity)
    """
    completed = matches[matches["is_completed"]].copy()

    # Did the toss winner also win the match?
    completed["toss_winner_won"] = completed["toss_winner"] == completed["match_winner"]

    # Win % by decision
    by_dec = (
        completed.groupby(["season", "toss_decision"], observed=True)
        .agg(
            matches=("match_id", "count"),
            toss_winner_wins=("toss_winner_won", "sum"),
        )
        .reset_index()
    )
    by_dec["toss_decision_win_pct"] = (by_dec["toss_winner_wins"] / by_dec["matches"]) * 100

    field = by_dec[by_dec["toss_decision"] == "field"][["season", "toss_decision_win_pct", "matches"]].rename(
        columns={"toss_decision_win_pct": "field_win_pct", "matches": "field_matches"}
    )
    bat = by_dec[by_dec["toss_decision"] == "bat"][["season", "toss_decision_win_pct", "matches"]].rename(
        columns={"toss_decision_win_pct": "bat_win_pct", "matches": "bat_matches"}
    )

    # % choosing to field per season
    total = (
        completed.groupby("season")
        .agg(total=("match_id", "count"), field=("toss_decision", lambda x: (x == "field").sum()))
        .reset_index()
    )
    total["field_first_pct"] = (total["field"] / total["total"]) * 100

    result = total.merge(field, on="season", how="left").merge(bat, on="season", how="left")
    return result


# ---------------------------------------------------------------------------
# 3. Venue scoring heatmap
# ---------------------------------------------------------------------------
def venue_scoring_profile(matches: pd.DataFrame, min_matches: int = 10) -> pd.DataFrame:
    """
    Returns one row per venue (with at least min_matches completed matches) with:
      - avg_first_innings_score
      - avg_target
      - chase_success_pct
      - match_count
      - low_sample flag (< min_matches)
    """
    completed = matches[matches["is_completed"]].copy()

    # Chase success: toss_decision == 'field' and toss_winner == match_winner
    # OR: determine by which team batted 2nd. We use the win_by_wickets proxy:
    # if match was won by wickets, the chasing team won.
    completed["chase_won"] = completed["win_by_wickets"].notna() & (completed["win_by_wickets"] > 0)

    venue_stats = (
        completed.groupby("venue")
        .agg(
            match_count=("match_id", "count"),
            avg_first_innings_score=("first_innings_total", "mean"),
            avg_target=("target", "mean"),
            chase_wins=("chase_won", "sum"),
        )
        .reset_index()
    )
    venue_stats["chase_success_pct"] = (venue_stats["chase_wins"] / venue_stats["match_count"]) * 100
    venue_stats["low_sample"] = venue_stats["match_count"] < min_matches
    venue_stats = venue_stats.sort_values("avg_first_innings_score", ascending=False)
    return venue_stats


# ---------------------------------------------------------------------------
# 4. Competitiveness index
# ---------------------------------------------------------------------------
def competitiveness_trend(matches: pd.DataFrame) -> pd.DataFrame:
    """
    A match is 'close' if the margin was small.
    We compute a normalised closeness score per match:
      - Wins by runs: invert and cap at 30 runs = boundary. score = max(0, 1 - runs_margin/30)
      - Wins by wickets: invert. score = max(0, 1 - wickets_margin/10)
    Then average per season. Higher = more competitive season.
    Also returns 'close_match_pct': % of matches decided by ≤15 runs or ≤2 wickets.
    """
    completed = matches[matches["is_completed"]].copy()

    completed["run_margin"] = pd.to_numeric(completed["win_by_runs"], errors="coerce")
    completed["wicket_margin"] = pd.to_numeric(completed["win_by_wickets"], errors="coerce")

    def closeness_score(row):
        if pd.notna(row["run_margin"]):
            return max(0.0, 1 - row["run_margin"] / 50)
        if pd.notna(row["wicket_margin"]):
            return max(0.0, 1 - row["wicket_margin"] / 10)
        return np.nan

    completed["closeness"] = completed.apply(closeness_score, axis=1)

    # Close match: ≤15 runs or ≤2 wickets
    completed["is_close"] = (
        (completed["run_margin"].fillna(999) <= 15) |
        (completed["wicket_margin"].fillna(999) <= 2)
    )

    result = (
        completed.groupby("season")
        .agg(
            avg_closeness=("closeness", "mean"),
            close_match_count=("is_close", "sum"),
            total_matches=("match_id", "count"),
        )
        .reset_index()
    )
    result["close_match_pct"] = (result["close_match_count"] / result["total_matches"]) * 100
    return result


# ---------------------------------------------------------------------------
# 5. Powerplay vs Death overs scoring divergence
# ---------------------------------------------------------------------------
def phase_scoring_era(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Season-wise average run rate per phase (powerplay / middle / death).
    Only first innings, legal balls, no super overs.
    Reveals whether death-over scoring is inflating faster than powerplay.
    """
    d = deliveries[
        (deliveries["innings"] == 1) &
        (~deliveries["is_super_over"]) &
        (deliveries["is_legal_ball"])
    ].copy()

    phase_rr = (
        d.groupby(["season_id", "over_phase"])
        .agg(runs=("batter_runs", "sum"), balls=("batter_runs", "count"))
        .reset_index()
    )
    phase_rr["run_rate"] = (phase_rr["runs"] / phase_rr["balls"]) * 6
    phase_rr = phase_rr.rename(columns={"season_id": "season"})
    return phase_rr


# ---------------------------------------------------------------------------
# 6. Impact Player rule effect (2023+)
# ---------------------------------------------------------------------------
def impact_player_effect(deliveries: pd.DataFrame, matches: pd.DataFrame) -> dict:
    """
    Simple before/after analysis of the Impact Player rule introduced in 2023.
    Returns aggregated avg scores and avg run rates for pre-2023 vs 2023+.
    """
    completed = matches[matches["is_completed"]].copy()
    completed["era"] = completed["season"].apply(lambda s: "Post-IP Rule (2023+)" if s >= 2023 else "Pre-IP Rule (<2023)")

    summary = (
        completed.groupby("era")
        .agg(
            avg_first_innings_score=("first_innings_total", "mean"),
            match_count=("match_id", "count"),
        )
        .reset_index()
    )

    d1 = deliveries[(deliveries["innings"] == 1) & (~deliveries["is_super_over"]) & deliveries["is_legal_ball"]].copy()
    d1 = d1.merge(completed[["match_id", "era"]], on="match_id", how="inner")
    rr = (
        d1.groupby("era")
        .agg(runs=("batter_runs", "sum"), balls=("batter_runs", "count"))
        .reset_index()
    )
    rr["avg_run_rate"] = (rr["runs"] / rr["balls"]) * 6
    return summary.merge(rr[["era", "avg_run_rate"]], on="era").to_dict(orient="records")
