"""
module_3_players.py — Player Performance Lab analytics & original metric computation.

Business question: "How good is this player, really — beyond raw stats?"

Original metrics defined here:
  1. Impact Score  — per-match composite measuring performance above a context-adjusted baseline
  2. Clutch Differential — pressure-situation delta vs normal-situation performance

All metric formulas are documented in docs/impact_score_methodology.md and
docs/clutch_contribution_methodology.md (written by write_methodology_docs()).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_div(num, den, default=np.nan):
    return num / den if den > 0 else default


# ---------------------------------------------------------------------------
# 1. Career & season aggregate stats
# ---------------------------------------------------------------------------
def career_batting_stats(deliveries: pd.DataFrame, players: pd.DataFrame,
                          min_innings: int = 10) -> pd.DataFrame:
    """
    Per-player career batting stats (across all seasons).
    Excludes super-over deliveries.
    """
    d = deliveries[~deliveries["is_super_over"]].copy()
    d["is_boundary"] = d["batter_runs"].isin([4, 6])
    d["is_six"]      = d["batter_runs"] == 6
    d["is_four"]     = d["batter_runs"] == 4
    d["is_dot"]      = d["batter_runs"] == 0

    # Innings = unique (match_id, batter) combinations
    innings = d.groupby(["batter", "match_id"]).agg(
        innings_runs=("batter_runs", "sum"),
        balls_faced=("is_legal_ball", "sum"),
        dismissed=("is_wicket", lambda x: x.any()),  # dismissed in this innings?
    ).reset_index()

    # Dismissal flag: the player was the one out
    wkt_balls = d[d["is_wicket"]].copy()
    dismissed_innings = set(zip(wkt_balls[wkt_balls["batter"] == wkt_balls["batter"]]["match_id"],
                                wkt_balls["batter"]))

    career = d.groupby("batter").agg(
        total_runs=("batter_runs", "sum"),
        balls_faced=("is_legal_ball", "sum"),
        matches=("match_id", "nunique"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum"),
        boundaries=("is_boundary", "sum"),
        dots=("is_dot", "sum"),
    ).reset_index()

    # Innings count and dismissals
    inn_agg = innings.groupby("batter").agg(
        innings=("match_id", "count"),
        dismissals=("dismissed", "sum"),
    ).reset_index()

    career = career.merge(inn_agg, on="batter", how="left")
    career["strike_rate"] = (career["total_runs"] / career["balls_faced"] * 100).round(2)
    career["batting_avg"]  = (career["total_runs"] / career["dismissals"].clip(lower=1)).round(2)
    career["boundary_pct"] = (career["boundaries"] / career["balls_faced"] * 100).round(1)
    career["dot_pct"]      = (career["dots"] / career["balls_faced"] * 100).round(1)

    # Merge full name
    name_map = dict(zip(players["player_name"], players["player_full_name"]))
    id_map   = dict(zip(players["player_name"], players["player_id"]))
    career["player_full_name"] = career["batter"].map(name_map).fillna(career["batter"])
    career["player_id"]        = career["batter"].map(id_map)

    career = career[career["innings"] >= min_innings].sort_values("total_runs", ascending=False)
    return career.reset_index(drop=True)


def career_bowling_stats(deliveries: pd.DataFrame, players: pd.DataFrame,
                          min_overs: int = 10) -> pd.DataFrame:
    """Per-player career bowling stats."""
    d = deliveries[~deliveries["is_super_over"] & deliveries["is_legal_ball"]].copy()
    d["is_dot"] = d["batter_runs"] == 0

    career = d.groupby("bowler").agg(
        wickets=("is_wicket", "sum"),
        runs_conceded=("total_runs", "sum"),
        balls=("is_legal_ball", "sum"),
        matches=("match_id", "nunique"),
        dots=("is_dot", "sum"),
    ).reset_index()

    career["overs"]       = (career["balls"] / 6).round(1)
    career["economy"]     = (career["runs_conceded"] / career["balls"] * 6).round(2)
    career["bowling_avg"] = (career["runs_conceded"] / career["wickets"].clip(lower=1)).round(2)
    career["bowling_sr"]  = (career["balls"] / career["wickets"].clip(lower=1)).round(1)
    career["dot_pct"]     = (career["dots"] / career["balls"] * 100).round(1)

    name_map = dict(zip(players["player_name"], players["player_full_name"]))
    id_map   = dict(zip(players["player_name"], players["player_id"]))
    career["player_full_name"] = career["bowler"].map(name_map).fillna(career["bowler"])
    career["player_id"]        = career["bowler"].map(id_map)

    career = career[career["overs"] >= min_overs].sort_values("wickets", ascending=False)
    return career.reset_index(drop=True)


def season_batting_stats(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Per-player per-season batting stats for career trajectory charts."""
    d = deliveries[~deliveries["is_super_over"]].copy()

    season_stats = d.groupby(["season_id", "batter"]).agg(
        runs=("batter_runs", "sum"),
        balls=("is_legal_ball", "sum"),
        matches=("match_id", "nunique"),
    ).reset_index().rename(columns={"season_id": "season"})

    innings_per_season = (
        d.groupby(["season_id", "batter"])["match_id"].nunique()
        .reset_index().rename(columns={"season_id": "season", "match_id": "innings"})
    )

    wkt_per_season = (
        d[d["is_wicket"]].groupby(["season_id", "batter"])["is_wicket"].sum()
        .reset_index().rename(columns={"season_id": "season", "is_wicket": "dismissals"})
    )

    result = season_stats.merge(innings_per_season, on=["season", "batter"], how="left")
    result = result.merge(wkt_per_season, on=["season", "batter"], how="left")
    result["dismissals"] = result["dismissals"].fillna(0)
    result["strike_rate"] = (result["runs"] / result["balls"].clip(lower=1) * 100).round(2)
    result["avg"]         = (result["runs"] / result["dismissals"].clip(lower=1)).round(2)
    return result


# ---------------------------------------------------------------------------
# 2. Phase-wise role classification
# ---------------------------------------------------------------------------
# Thresholds: if % of batter's runs come from this phase, classify accordingly
ROLE_THRESHOLDS = {
    "powerplay": 0.40,   # >= 40% runs in PP → Powerplay Hitter
    "death":     0.38,   # >= 38% runs in death → Finisher
    # else → Anchor (middle-overs specialist batter)
}

BOWL_ROLE_THRESHOLDS = {
    "powerplay": 0.42,
    "death":     0.40,
}


def player_phase_profile(deliveries: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """
    Per-player phase-wise batting contribution breakdown.
    Assigns a role label and returns the underlying phase percentages.
    """
    d = deliveries[~deliveries["is_super_over"] & deliveries["is_legal_ball"]].copy()
    d["over_phase"] = d["over_phase"].astype(str)

    phase_runs = d.groupby(["batter", "over_phase"])["batter_runs"].sum().reset_index()
    total_runs = d.groupby("batter")["batter_runs"].sum().reset_index().rename(columns={"batter_runs": "total"})
    phase_runs = phase_runs.merge(total_runs, on="batter")
    phase_runs["phase_pct"] = phase_runs["batter_runs"] / phase_runs["total"].clip(lower=1)

    pivot = phase_runs.pivot(index="batter", columns="over_phase", values="phase_pct").fillna(0).reset_index()
    for col in ["powerplay", "middle", "death"]:
        if col not in pivot.columns:
            pivot[col] = 0.0

    def classify_batter(row):
        pp, mid, death = row.get("powerplay", 0), row.get("middle", 0), row.get("death", 0)
        if pp >= ROLE_THRESHOLDS["powerplay"]:
            return "Powerplay Hitter"
        if death >= ROLE_THRESHOLDS["death"]:
            return "Finisher"
        if mid >= 0.45:
            return "Anchor"
        return "Versatile"

    pivot["batting_role"] = pivot.apply(classify_batter, axis=1)

    # Bowling phase profile
    phase_wkts = d.groupby(["bowler", "over_phase"])["is_wicket"].sum().reset_index()
    total_wkts = d.groupby("bowler")["is_wicket"].sum().reset_index().rename(columns={"is_wicket": "total_wkts"})
    phase_wkts = phase_wkts.merge(total_wkts, on="bowler")
    phase_wkts["wkt_pct"] = phase_wkts["is_wicket"] / phase_wkts["total_wkts"].clip(lower=1)
    bowl_pivot = phase_wkts.pivot(index="bowler", columns="over_phase", values="wkt_pct").fillna(0).reset_index()
    for col in ["powerplay", "middle", "death"]:
        if col not in bowl_pivot.columns:
            bowl_pivot[col] = 0.0

    def classify_bowler(row):
        pp, death = row.get("powerplay", 0), row.get("death", 0)
        if pp >= BOWL_ROLE_THRESHOLDS["powerplay"]:
            return "New-Ball Bowler"
        if death >= BOWL_ROLE_THRESHOLDS["death"]:
            return "Death Specialist"
        return "Middle-Overs Specialist"

    bowl_pivot["bowling_role"] = bowl_pivot.apply(classify_bowler, axis=1)
    bowl_pivot = bowl_pivot.rename(columns={
        "powerplay": "bowl_pp_pct", "middle": "bowl_mid_pct", "death": "bowl_death_pct"
    })

    pivot = pivot.rename(columns={
        "powerplay": "bat_pp_pct", "middle": "bat_mid_pct", "death": "bat_death_pct"
    })

    return pivot, bowl_pivot


# ---------------------------------------------------------------------------
# 3. Impact Score  (original metric — Dream11-inspired event scoring)
# ---------------------------------------------------------------------------
# BATTING (per innings):
#   Runs: +0.5 pts each
#   Four: +1 bonus  |  Six: +2 bonus
#   Milestones: 25=+4, 50=+8, 75=+12, 100=+16
#   Strike rate (min 10 balls): ≥170=+6, ≥150=+4, ≥130=+2, <70=-6, <90=-4, <100=-2
#   Duck (out for 0): -4
#   Context × 1.5 if chasing RRR > 10  |  × 1.25 if chasing RRR 8–10
#
# BOWLING (per match):
#   Wicket: +25 base
#   Premium batter (career SR > 130): +8 bonus per wicket
#   Bowled/LBW: +8 bonus per wicket (skill dismissal)
#   3-wkt haul: +4  |  4-wkt: +8  |  5-wkt: +16
#   Maiden over: +12 each
#   Economy bonus (vs league phase avg): <5=+10, 5–6=+6, 6–7=+2, 8–9=-4, 9–10=-8, >10=-12
#   Context × 1.3 if defending low total (<160) or bowling death overs in high-RRR chase
#
# CAREER SCORE = mean_per_match × (n / (n + 25))   ← Bayesian shrinkage
#   This ensures 5-match wonders don't outscore 200-match legends
#   e.g. 5 matches gets 5/30 = 17% weight; 100 matches gets 100/125 = 80% weight
#
# SEASON SCORE = mean_per_match (no shrinkage — single season comparison is fair)
#
# MINIMUM THRESHOLDS (career): batting ≥ 20 innings  |  bowling ≥ 30 overs
# MINIMUM THRESHOLDS (season): batting ≥ 3 innings   |  bowling ≥ 4 overs
# ---------------------------------------------------------------------------

def compute_impact_score(deliveries: pd.DataFrame, career_bat: pd.DataFrame,
                          career_bowl: pd.DataFrame,
                          matches: pd.DataFrame = None,
                          career_mode: bool = True) -> pd.DataFrame:
    """
    Compute Impact Score using Dream11-inspired event-based scoring.

    Aggregation strategy (cricket expert rationale):
      career_mode=True  → per-match MEAN x Bayesian shrinkage(k=25)
                          "who is the most consistently excellent player all-time?"
                          Small-sample wonders get penalised vs 200-match legends.
      career_mode=False → TOTAL season points (sum, not average)
                          "who contributed the most this season?"
                          Kohli playing 16 matches and scoring 973 runs MUST outscore
                          Zampa playing 5 brilliant games — total contribution wins.

    Bowling scale calibrated to be comparable to batting:
      Batting: Kohli 60-run average innings ≈ 44 pts/innings
      Bowling: Bumrah 1.5-wicket average match ≈ 40-50 pts/match
      Wicket base = 16 (not 25) keeps these comparable; quality/type/haul bonuses
      reward exceptional bowling without creating a scale mismatch.
    """
    d = deliveries[~deliveries["is_super_over"] & deliveries["is_legal_ball"]].copy()
    d["over_phase"] = d["over_phase"].astype(str)

    min_inn  = 20 if career_mode else 3
    min_ovs  = 30 if career_mode else 4
    shrink_k = 25  # only used in career mode; season uses total contribution

    # ── BATTING IMPACT ────────────────────────────────────────────────────────
    inn_bat = d.groupby(["match_id", "batter"]).agg(
        runs  = ("batter_runs", "sum"),
        balls = ("is_legal_ball", "sum"),
        fours = ("batter_runs", lambda x: (x == 4).sum()),
        sixes = ("batter_runs", lambda x: (x == 6).sum()),
        out   = ("is_wicket", "max"),
    ).reset_index()
    inn_bat["sr"] = inn_bat["runs"] / inn_bat["balls"].clip(lower=1) * 100

    inn_bat["pts_runs"]      = inn_bat["runs"] * 0.5
    inn_bat["pts_4s"]        = inn_bat["fours"] * 1.0
    inn_bat["pts_6s"]        = inn_bat["sixes"] * 2.0
    inn_bat["pts_milestone"] = __import__("numpy").select(
        [inn_bat["runs"] >= 100, inn_bat["runs"] >= 75,
         inn_bat["runs"] >= 50,  inn_bat["runs"] >= 25],
        [16, 12, 8, 4], default=0
    )
    inn_bat["pts_sr"] = __import__("numpy").select(
        [
            inn_bat["balls"] < 10,
            inn_bat["sr"] >= 170, inn_bat["sr"] >= 150, inn_bat["sr"] >= 130,
            inn_bat["sr"] <  70,  inn_bat["sr"] <  90,  inn_bat["sr"] < 100,
        ],
        [0, 6, 4, 2, -6, -4, -2], default=0
    )
    inn_bat["pts_duck"] = ((inn_bat["runs"] == 0) & (inn_bat["out"] == 1)).astype(int) * -4
    inn_bat["bat_raw"]  = (
        inn_bat["pts_runs"] + inn_bat["pts_4s"] + inn_bat["pts_6s"] +
        inn_bat["pts_milestone"] + inn_bat["pts_sr"] + inn_bat["pts_duck"]
    )

    # Pressure chase multiplier
    chase_ctx = (
        d[d["innings"] == 2]
        .groupby(["match_id", "batter"])
        .agg(avg_rrr=("required_run_rate", "mean"))
        .reset_index()
    )
    chase_ctx["bat_ctx"] = __import__("numpy").where(
        chase_ctx["avg_rrr"] > 10, 1.5,
        __import__("numpy").where(chase_ctx["avg_rrr"] > 8, 1.25, 1.0)
    )
    inn_bat = inn_bat.merge(chase_ctx[["match_id","batter","bat_ctx"]], on=["match_id","batter"], how="left")
    inn_bat["bat_ctx"] = inn_bat["bat_ctx"].fillna(1.0)
    inn_bat["bat_pts"] = inn_bat["bat_raw"] * inn_bat["bat_ctx"]

    bat_agg = inn_bat.groupby("batter").agg(
        bat_total   = ("bat_pts", "sum"),
        bat_mean    = ("bat_pts", "mean"),
        bat_innings = ("bat_pts", "count"),
        bat_matches = ("match_id", "nunique"),
    ).reset_index()
    bat_agg = bat_agg[bat_agg["bat_innings"] >= min_inn]

    if career_mode:
        # Career: per-match mean × Bayesian shrinkage
        # Rewards consistent excellence — Kohli 275 matches >> Suryavanshi 23 matches
        bat_agg["batting_impact_score"] = (
            bat_agg["bat_mean"] *
            (bat_agg["bat_matches"] / (bat_agg["bat_matches"] + shrink_k))
        ).round(1)
    else:
        # Season: total contribution across all matches in that season
        # Rewards sustained performance — Kohli 16×60 runs beats Zampa 5 brilliant games
        bat_agg["batting_impact_score"] = bat_agg["bat_total"].round(1)

    # ── BOWLING IMPACT ────────────────────────────────────────────────────────
    batter_sr_map = dict(zip(career_bat["batter"], career_bat["strike_rate"]))
    median_sr     = career_bat["strike_rate"].median() if len(career_bat) > 0 else 120.0

    # Wicket base = 16 (calibrated so bowling is comparable scale to batting)
    # Quality bonus +8 for dismissing premium batter (SR > 130)
    # Type bonus +4 for Bowled/LBW (skill dismissal)
    wkt_balls = d[d["is_wicket"]].copy()
    wkt_balls["dismissed_sr"] = wkt_balls["batter"].map(batter_sr_map).fillna(median_sr)
    wkt_balls["wkt_base"]     = 16.0
    wkt_balls["wkt_quality"]  = (wkt_balls["dismissed_sr"] > 130).astype(float) * 8.0
    if "wicket_kind" in wkt_balls.columns:
        wkt_balls["wkt_type"] = wkt_balls["wicket_kind"].isin(["bowled","lbw"]).astype(float) * 4.0
    else:
        wkt_balls["wkt_type"] = 0.0
    wkt_balls["wkt_pts"] = wkt_balls["wkt_base"] + wkt_balls["wkt_quality"] + wkt_balls["wkt_type"]

    bowl_wkt_match = (
        wkt_balls.groupby(["bowler","match_id"])
        .agg(wkt_pts=("wkt_pts","sum"), wickets=("is_wicket","sum"))
        .reset_index()
    )
    bowl_wkt_match["haul_bonus"] = __import__("numpy").select(
        [bowl_wkt_match["wickets"] >= 5, bowl_wkt_match["wickets"] == 4,
         bowl_wkt_match["wickets"] == 3],
        [16, 8, 4], default=0
    )
    bowl_wkt_match["wkt_total"] = bowl_wkt_match["wkt_pts"] + bowl_wkt_match["haul_bonus"]

    # Maiden over bonus (+8 each — reduced from 12, rare in T20)
    if "over_number" in d.columns:
        over_totals = d.groupby(["match_id","bowler","over_number"])["total_runs"].sum().reset_index()
        maiden_df   = (
            over_totals[over_totals["total_runs"] == 0]
            .groupby(["match_id","bowler"]).size().reset_index(name="maidens")
        )
        maiden_df["maiden_pts"] = maiden_df["maidens"] * 8.0
    else:
        maiden_df = __import__("pandas").DataFrame(columns=["match_id","bowler","maiden_pts"])

    # Economy points: reward tight bowling, penalise expensive spells
    # Bands calibrated vs IPL league average ~8.5 RPO
    bowl_match = (
        d.groupby(["bowler","match_id"])
        .agg(runs_c=("total_runs","sum"), balls_b=("is_legal_ball","sum"))
        .reset_index()
    )
    bowl_match = bowl_match[bowl_match["balls_b"] >= 6]
    bowl_match["actual_rpo"] = bowl_match["runs_c"] / (bowl_match["balls_b"] / 6)
    bowl_match["eco_pts"] = __import__("numpy").select(
        [
            bowl_match["actual_rpo"] < 5,
            bowl_match["actual_rpo"] < 6,
            bowl_match["actual_rpo"] < 7,
            bowl_match["actual_rpo"] < 8,
            bowl_match["actual_rpo"] < 9,
            bowl_match["actual_rpo"] < 10,
        ],
        [10, 6, 2, 0, -4, -8], default=-12
    )

    # Context multiplier
    if matches is not None:
        low_total_set = set(
            d[d["innings"]==1].groupby("match_id")["batter_runs"].sum()
            .loc[lambda x: x < 160].index
        )
    else:
        low_total_set = set()

    death_chase_pairs = set(
        map(tuple,
            d[(d["innings"]==2) & (d["over_phase"]=="death") &
              (d["required_run_rate"].fillna(0) > 9)]
            .groupby(["match_id","bowler"]).size().reset_index()[["match_id","bowler"]].values.tolist()
        )
    )

    bowl_combined = (
        bowl_wkt_match[["bowler","match_id","wkt_total"]]
        .merge(maiden_df[["match_id","bowler","maiden_pts"]], on=["match_id","bowler"], how="outer")
        .merge(bowl_match[["match_id","bowler","eco_pts"]],  on=["match_id","bowler"], how="outer")
        .fillna({"wkt_total":0, "maiden_pts":0, "eco_pts":0})
    )
    bowl_combined["low_total"]   = bowl_combined["match_id"].isin(low_total_set)
    bowl_combined["death_chase"] = [
        (r["match_id"], r["bowler"]) in death_chase_pairs
        for _, r in bowl_combined[["match_id","bowler"]].iterrows()
    ]
    bowl_combined["ctx"]      = __import__("numpy").where(
        bowl_combined["low_total"] | bowl_combined["death_chase"], 1.3, 1.0
    )
    bowl_combined["bowl_pts"] = (
        (bowl_combined["wkt_total"] + bowl_combined["maiden_pts"] + bowl_combined["eco_pts"])
        * bowl_combined["ctx"]
    )

    bowler_overs = (
        d.groupby("bowler")["is_legal_ball"].sum().div(6).reset_index()
        .rename(columns={"is_legal_ball":"total_overs"})
    )
    bowl_agg = (
        bowl_combined.groupby("bowler").agg(
            bowl_total   = ("bowl_pts", "sum"),
            bowl_mean    = ("bowl_pts", "mean"),
            bowl_matches = ("match_id", "count"),
        ).reset_index()
        .merge(bowler_overs, on="bowler", how="left")
    )
    bowl_agg = bowl_agg[bowl_agg["total_overs"].fillna(0) >= min_ovs]

    if career_mode:
        bowl_agg["bowling_impact_score"] = (
            bowl_agg["bowl_mean"] *
            (bowl_agg["bowl_matches"] / (bowl_agg["bowl_matches"] + shrink_k))
        ).round(1)
    else:
        bowl_agg["bowling_impact_score"] = bowl_agg["bowl_total"].round(1)

    # ── COMBINE ───────────────────────────────────────────────────────────────
    result = (
        bat_agg[["batter","batting_impact_score","bat_matches"]]
        .merge(
            bowl_agg[["bowler","bowling_impact_score","bowl_matches"]],
            left_on="batter", right_on="bowler", how="outer"
        )
    )
    result["player"]                = result["batter"].fillna(result["bowler"])
    result["batting_impact_score"]  = result["batting_impact_score"].fillna(0).round(1)
    result["bowling_impact_score"]  = result["bowling_impact_score"].fillna(0).round(1)
    result["combined_impact_score"] = (
        result["batting_impact_score"] + result["bowling_impact_score"]
    ).round(1)
    result = result[result["player"].notna()]

    return result[["player","batting_impact_score","bowling_impact_score",
                   "combined_impact_score","bat_matches","bowl_matches"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Clutch Score  (original metric)
# ---------------------------------------------------------------------------
# Pressure = 2nd innings chasing a target >= 150 (genuine competitive T20 chase).
# Minimum 15 balls faced in that innings (meaningful individual contribution).
#
# Batting Clutch Score:
#   batting_avg   = total_pressure_runs / dismissals  (not-outs keep denominator low)
#   sr_component  = mean innings SR
#   win_contrib   = % innings where player scored >= 25 AND team won
#
#   raw = (bat_avg  / league_bat_avg  - 1) * 60
#       + (avg_sr   / league_avg_sr   - 1) * 25
#       + win_contrib_rate             * 15
#
#   clutch_score_bat = raw * shrinkage(innings, k=25)
#
# Bowling Clutch: death overs in close match (unchanged — individual stats).
# ---------------------------------------------------------------------------


def compute_clutch_differential(deliveries: pd.DataFrame,
                                 matches: pd.DataFrame,
                                 min_bat_normal: int = 200,
                                 min_bat_pressure: int = 60,
                                 min_bat_innings: int = 20,
                                 min_bowl_normal: int = 300,
                                 min_bowl_pressure: int = 120,
                                 min_bowl_innings: int = 5,
                                 min_target: int = 150,
                                 min_bat_balls_per_inn: int = 15,
                                 fixed_league_bat_avg: float = None,
                                 fixed_league_avg_sr: float = None) -> tuple:
    """
    Returns (bat_clutch, bowl_clutch) DataFrames.

    Batting Clutch -- Chase Batting Index:
      Who are the best clutch performers in competitive T20 chases?

      Metric = Batting Average (with not-out credit) + SR component + Win Contribution
      compared to the league average across the same qualifying pressure innings.

      Key insight: Batting AVERAGE (runs / dismissals) rather than runs per innings
      rewards players who stay not-out and finish chases. Kohli's 30 fifties across
      59 qualifying innings and ABD's 14 not-outs in 26 innings both get full credit.

    Bowling Clutch: unchanged -- eco/wicket-rate composite in death close matches.
    """
    d = deliveries[~deliveries["is_super_over"] & deliveries["is_legal_ball"]].copy()
    d["over_phase"] = d["over_phase"].astype(str)

    # Match-level target (1st innings total + 1)
    first_inn_runs = (
        d[d["innings"] == 1]
        .groupby("match_id")["total_runs"].sum()
    )
    target_map = (first_inn_runs + 1).to_dict()
    d["match_target"] = d["match_id"].map(target_map)

    # Close match (for bowling only)
    close_ids = matches[
        (matches["is_completed"]) & (
            (matches["win_by_runs"].fillna(999) <= 15) |
            (matches["win_by_wickets"].fillna(999) <= 3)
        )
    ]["match_id"].tolist()
    d["is_close_match"] = d["match_id"].isin(close_ids)

    # Bowling pressure: death overs in close match
    d["is_pressure_bowl"] = d["is_close_match"] & (d["over_phase"] == "death")

    # Match winner lookup
    match_winner = matches.set_index("match_id")["match_winner"].to_dict()

    shrink_k = 25

    # -- BATTING CLUTCH (Chase Batting Index) ---------------------------------
    # Pressure innings: 2nd innings, target >= min_target, batter faced >= min balls
    bat2 = d[(d["innings"] == 2) & (d["match_target"] >= min_target)].copy()

    inn = bat2.groupby(["match_id", "batter"]).agg(
        p_runs=("batter_runs", "sum"),
        p_balls=("is_legal_ball", "sum"),
        team=("team_batting", "first"),
        dismissed=("is_wicket", "any"),
    ).reset_index()

    # Only meaningful innings (min balls faced)
    inn = inn[inn["p_balls"] >= min_bat_balls_per_inn].copy()
    inn["p_sr"] = inn["p_runs"] / inn["p_balls"] * 100
    inn["won"] = inn.apply(
        lambda r: r["team"] == match_winner.get(r["match_id"]), axis=1
    )
    # Win contribution: individual score >= 25 AND team won
    inn["win_contrib"] = inn["won"] & (inn["p_runs"] >= 25)

    # League baselines — use fixed career-level values when provided (season view)
    # so scores stay on the same scale regardless of which season is selected
    if fixed_league_bat_avg is not None:
        league_bat_avg = fixed_league_bat_avg
        league_avg_sr  = fixed_league_avg_sr if fixed_league_avg_sr is not None else inn["p_sr"].mean()
    else:
        league_bat_avg = inn["p_runs"].sum() / max(inn["dismissed"].sum(), 1)
        league_avg_sr  = inn["p_sr"].mean()

    # Aggregate per batter
    bat_agg = inn.groupby("batter").agg(
        total_runs=("p_runs", "sum"),
        innings=("match_id", "nunique"),
        dismissals=("dismissed", "sum"),
        avg_sr=("p_sr", "mean"),
        win_innings=("won", "sum"),
        win_contrib_innings=("win_contrib", "sum"),
        fifties_pressure=("p_runs", lambda x: (x >= 50).sum()),
        thirties_pressure=("p_runs", lambda x: ((x >= 30) & (x < 50)).sum()),
    ).reset_index()

    bat_agg = bat_agg[bat_agg["innings"] >= min_bat_innings].reset_index(drop=True)

    bat_agg["bat_avg"]           = (bat_agg["total_runs"] / bat_agg["dismissals"].clip(lower=1)).round(1)
    bat_agg["avg_pressure_sr"]   = bat_agg["avg_sr"].round(1)
    bat_agg["pressure_win_rate"] = (bat_agg["win_innings"] / bat_agg["innings"]).round(3)
    bat_agg["not_out_rate"]      = ((bat_agg["innings"] - bat_agg["dismissals"]) / bat_agg["innings"]).round(3)
    bat_agg["win_contrib_rate"]  = (bat_agg["win_contrib_innings"] / bat_agg["innings"]).round(3)

    bat_agg["avg_diff"] = (bat_agg["bat_avg"]         / league_bat_avg - 1)
    bat_agg["sr_diff"]  = (bat_agg["avg_pressure_sr"] / league_avg_sr  - 1)
    bat_agg["raw_clutch"] = (
        bat_agg["avg_diff"] * 60 +
        bat_agg["sr_diff"]  * 25 +
        bat_agg["win_contrib_rate"] * 15
    )

    bat_agg["shrinkage"] = bat_agg["innings"] / (bat_agg["innings"] + shrink_k)
    bat_agg["clutch_score_bat"] = (bat_agg["raw_clutch"] * bat_agg["shrinkage"]).round(1)

    bat_clutch = bat_agg.rename(columns={"innings": "pressure_matches"})
    bat_clutch["league_bat_avg"] = round(league_bat_avg, 1)
    bat_clutch["league_avg_sr"]  = round(league_avg_sr, 1)

    # -- BOWLING CLUTCH (unchanged) -------------------------------------------
    def bowl_stats(mask, label):
        sub = d[mask]
        eco_grp = sub.groupby("bowler").agg(
            runs=("total_runs", "sum"),
            balls=("is_legal_ball", "sum"),
            innings=("match_id", "nunique"),
        ).reset_index()
        eco_grp[f"eco_{label}"] = (eco_grp["runs"] / eco_grp["balls"].clip(lower=1) * 6).round(2)
        eco_grp[f"balls_{label}"] = eco_grp["balls"]
        eco_grp[f"innings_{label}"] = eco_grp["innings"]
        wkt_grp = sub[sub["is_wicket"] == 1].groupby("bowler").size().reset_index(name=f"wkts_{label}")
        eco_grp = eco_grp.merge(wkt_grp, on="bowler", how="left")
        eco_grp[f"wkts_{label}"] = eco_grp[f"wkts_{label}"].fillna(0)
        eco_grp[f"wpo_{label}"] = (
            eco_grp[f"wkts_{label}"] / (eco_grp[f"balls_{label}"] / 6).clip(lower=0.01)
        ).round(3)
        return eco_grp[["bowler", f"eco_{label}", f"balls_{label}", f"innings_{label}", f"wpo_{label}"]]

    bowl_normal   = bowl_stats(~d["is_pressure_bowl"], "normal")
    bowl_pressure = bowl_stats(d["is_pressure_bowl"],  "pressure")

    bowl_clutch = bowl_normal.merge(bowl_pressure, on="bowler", how="inner")
    bowl_clutch["eco_diff"] = (bowl_clutch["eco_normal"]  - bowl_clutch["eco_pressure"]).round(2)
    bowl_clutch["wpo_diff"] = (bowl_clutch["wpo_pressure"] - bowl_clutch["wpo_normal"]).round(3)
    bowl_clutch["clutch_differential_bowl"] = (
        bowl_clutch["eco_diff"] * 0.7 + bowl_clutch["wpo_diff"] * 6 * 0.3
    ).round(2)
    bowl_clutch = bowl_clutch[
        (bowl_clutch["balls_normal"] >= min_bowl_normal) &
        (bowl_clutch["balls_pressure"] >= min_bowl_pressure) &
        (bowl_clutch["innings_pressure"] >= min_bowl_innings)
    ].reset_index(drop=True)

    pboM = (
        d[d["is_pressure_bowl"]]
        .groupby(["bowler", "match_id"])
        .agg(team=("team_bowling", "first"))
        .reset_index()
    )
    pboM["won"] = pboM.apply(
        lambda r: r["team"] == match_winner.get(r["match_id"]), axis=1
    )
    win_rate_bowl = pboM.groupby("bowler")["won"].mean().reset_index(name="pressure_win_rate")
    bowl_clutch = bowl_clutch.merge(win_rate_bowl, on="bowler", how="left")
    bowl_clutch["pressure_win_rate"] = bowl_clutch["pressure_win_rate"].fillna(0.5)

    bowl_clutch["shrinkage"]  = (
        bowl_clutch["innings_pressure"] / (bowl_clutch["innings_pressure"] + 10)
    )
    bowl_clutch["win_factor"] = 0.5 + bowl_clutch["pressure_win_rate"] * 0.5
    bowl_clutch["clutch_score_bowl"] = (
        bowl_clutch["clutch_differential_bowl"] *
        bowl_clutch["win_factor"] *
        bowl_clutch["shrinkage"]
    ).round(2)

    return bat_clutch, bowl_clutch


# ---------------------------------------------------------------------------
# 5. Player archetype quadrant (SR × Average)
# ---------------------------------------------------------------------------
def archetype_quadrant(career_bat: pd.DataFrame) -> pd.DataFrame:
    """
    Classify batters into 4 quadrants based on Strike Rate vs Average:
      High SR + High Avg  = Match-Winner
      High SR + Low Avg   = Hitter (explosive but inconsistent)
      Low SR  + High Avg  = Accumulator (safe but slow)
      Low SR  + Low Avg   = Passenger
    Uses median as the split point for each axis.
    """
    df = career_bat[career_bat["innings"] >= 15].copy()
    if df.empty:
        return df
    sr_median  = df["strike_rate"].median()
    avg_median = df["batting_avg"].median()

    def quadrant(row):
        hi_sr  = row["strike_rate"]  >= sr_median
        hi_avg = row["batting_avg"] >= avg_median
        if hi_sr and hi_avg:   return "Match-Winner"
        if hi_sr and not hi_avg: return "Hitter"
        if not hi_sr and hi_avg: return "Accumulator"
        return "Passenger"

    df["archetype"]  = df.apply(quadrant, axis=1)
    df["sr_median"]  = sr_median
    df["avg_median"] = avg_median
    return df


# ---------------------------------------------------------------------------
# 6. Finisher Index (death-overs batting)
# ---------------------------------------------------------------------------
def finisher_index(deliveries: pd.DataFrame, players: pd.DataFrame,
                    min_death_balls: int = 60) -> pd.DataFrame:
    """
    Finisher Index = (death-overs SR / league_death_SR) * (death_runs / total_runs contribution)
    Measures how much a batter elevates their game in the death compared to their own average.
    Higher = more effective death-overs finisher.
    """
    d = deliveries[~deliveries["is_super_over"] & deliveries["is_legal_ball"]].copy()
    d["over_phase"] = d["over_phase"].astype(str)

    league_death_sr = (
        d[d["over_phase"] == "death"]["batter_runs"].sum() /
        (d[d["over_phase"] == "death"]["is_legal_ball"].sum() + 1e-9) * 100
    )

    death = d[d["over_phase"] == "death"].groupby("batter").agg(
        death_runs=("batter_runs", "sum"),
        death_balls=("is_legal_ball", "sum"),
    ).reset_index()
    death["death_sr"] = (death["death_runs"] / death["death_balls"].clip(lower=1) * 100).round(2)

    total = d.groupby("batter")["batter_runs"].sum().reset_index().rename(columns={"batter_runs": "total_runs"})
    death = death.merge(total, on="batter")
    death["death_contribution_pct"] = (death["death_runs"] / death["total_runs"].clip(lower=1) * 100).round(1)
    death["finisher_index"] = (
        (death["death_sr"] / league_death_sr) * (death["death_contribution_pct"] / 100)
    ).round(3)

    name_map = dict(zip(players["player_name"], players["player_full_name"]))
    death["player_full_name"] = death["batter"].map(name_map).fillna(death["batter"])

    return (
        death[death["death_balls"] >= min_death_balls]
        .sort_values("finisher_index", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 7. Consistency score
# ---------------------------------------------------------------------------
def consistency_score(deliveries: pd.DataFrame, players: pd.DataFrame,
                       min_innings: int = 15) -> pd.DataFrame:
    """
    Consistency = 1 / CV(innings_runs)  where CV = std / mean.
    Higher = more consistent performer per innings.
    Also computes boundary-dependency ratio (% of runs from 4s and 6s).
    """
    d = deliveries[~deliveries["is_super_over"] & deliveries["is_legal_ball"]].copy()

    innings = d.groupby(["batter", "match_id"]).agg(
        innings_runs=("batter_runs", "sum"),
        boundaries=("batter_runs", lambda x: ((x == 4) | (x == 6)).sum()),
    ).reset_index()

    stats = innings.groupby("batter").agg(
        innings_count=("innings_runs", "count"),
        mean_runs=("innings_runs", "mean"),
        std_runs=("innings_runs", "std"),
        total_runs=("innings_runs", "sum"),
        total_boundaries=("boundaries", "sum"),
    ).reset_index()

    # Also compute dismissals for batting average
    wicket_balls = d[d["is_wicket"]].copy()
    dismissals = wicket_balls.groupby("batter")["is_wicket"].sum().reset_index()
    dismissals.columns = ["batter", "dismissals"]

    stats = stats[stats["innings_count"] >= min_innings]
    stats = stats.merge(dismissals, on="batter", how="left")
    stats["dismissals"] = stats["dismissals"].fillna(1)

    # Batting average = runs per dismissal (standard cricket definition)
    stats["batting_avg"] = (stats["total_runs"] / stats["dismissals"].clip(lower=1)).round(2)

    # Stability = 1/CV (how steady the scores are)
    stats["cv"] = (stats["std_runs"] / stats["mean_runs"].clip(lower=0.1)).round(3)
    stats["stability_score"] = (1 / stats["cv"].clip(lower=0.01)).round(2)

    # TRUE consistency = quality (avg) × stability — rewards only high-avg + low-variance players
    # Normalise batting_avg to same scale as stability (divide by 10 so ~30avg gives ~3.0)
    stats["consistency_score"] = ((stats["batting_avg"] / 10) * stats["stability_score"]).round(2)

    stats["boundary_dependency"] = (stats["total_boundaries"] * 4.5 / stats["total_runs"].clip(lower=1) * 100).round(1)

    name_map = dict(zip(players["player_name"], players["player_full_name"]))
    stats["player_full_name"] = stats["batter"].map(name_map).fillna(stats["batter"])

    return stats.sort_values("consistency_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 8. Write methodology documentation
# ---------------------------------------------------------------------------
def write_methodology_docs(docs_dir) -> None:
    from pathlib import Path
    docs_dir = Path(docs_dir)

    impact_md = """# Impact Score Methodology
## Purpose
Impact Score measures how much a player's performance in a given match *exceeded what an
average IPL player would have achieved in the same ball-by-ball context*. It rewards
performing in difficult situations and penalises ordinary returns under easy conditions.

## Formula

### Batting Impact (per ball)
```
batting_impact(ball) = (batter_runs - expected_runs_per_ball_in_phase) × context_multiplier
```
- **expected_runs_per_ball_in_phase**: league-wide average runs per legal ball in that phase
  (powerplay / middle / death) computed from first-innings, non-super-over deliveries
- **context_multiplier**: 1.25 if the ball was bowled in a second-innings chase with
  Required Run Rate ≥ 9; 1.0 otherwise. Rewards scoring in high-pressure chases.

Per-match batting impact = sum of ball-level batting_impact values for that player in that match.

### Bowling Impact (per match)
```
bowling_impact(match) = wicket_value + dot_pressure + economy_bonus
```
- **wicket_value** = (dismissed batter's career strike rate / 130) × 2.5
  → A higher-quality batter's wicket is worth more
- **dot_pressure** = 0.15 per dot ball bowled
  → Rewards building pressure regardless of wickets
- **economy_bonus** = (expected_runs_per_ball - actual_runs_per_ball) × 0.3 per ball
  → Rewards bowling below the phase average; penalises leaking runs

### Career Impact Score
```
Impact Score = mean(per-match impact across all matches, min 5 matches)
```
Using the **mean** (not sum) removes volume bias — a player with 5 exceptional innings
scores higher than a player who played 50 ordinary innings.

## Worked Example — Batting (MS Dhoni, hypothetical match)
Suppose Dhoni faced 20 balls in the death overs (overs 16–20) in a second-innings chase
where Required Run Rate = 11:
- League death-overs expected runs per ball = 1.45/6 ≈ 0.242
- Context multiplier = 1.25 (RRR > 9 in a chase)
- Dhoni scored 30 off those 20 balls
- Per-ball batting impact = (30/20 − 0.242) × 1.25 = (1.5 − 0.242) × 1.25 = 1.573
- Over 20 balls: batting_impact = 1.573 × 20 / 6 ≈ **+5.24** for that match

## Assumptions & Limitations
- No fielding, fitness, or captaincy contribution
- Context multiplier uses a binary RRR ≥ 9 threshold (a continuous function would be more precise)
- Bowlers who specialise in non-death phases may have lower impact scores than their quality warrants
- Career Impact Score weights each match equally (no recency weighting)
"""

    clutch_md = """# Clutch Differential Methodology
## Purpose
Clutch Differential answers: "Does this player raise their game under pressure, or do they
wilt?" It measures the *gap* between pressure-situation performance and normal-situation
performance.

## Pressure Situation Definition

### Batters
A batting delivery is "pressure" if:
- The match is a second-innings chase AND required_run_rate ≥ 9, OR
- The ball is in the final 24 legal balls of a second-innings chase (death-overs crunch)

### Bowlers
A bowling delivery is "pressure" if:
- The match was ultimately decided by ≤ 15 runs or ≤ 2 wickets (i.e., it was close) AND
- The ball was bowled in the death phase (overs 16–20)

These definitions are shown in-app alongside every Clutch Differential number.

## Formula

### Batting Clutch Differential
```
Clutch Differential (bat) = Pressure Strike Rate − Normal Strike Rate
```
Positive = batter performs BETTER under pressure than in normal situations.

### Bowling Clutch Differential
```
Clutch Differential (bowl) = Normal Economy − Pressure Economy
```
Positive = bowler is MORE economical under pressure than in normal situations.

## Minimum Sample Thresholds
- Batters: ≥ 50 normal balls AND ≥ 20 pressure balls
- Bowlers: ≥ 100 normal balls AND ≥ 30 pressure balls

Small samples would make the metric unreliable and easily gamed by one exceptional innings.

## Interpretation
| Clutch Differential | Interpretation |
|---|---|
| > +15 (bat) / > +1.5 (bowl) | Elite clutch performer |
| +5 to +15 (bat) / +0.5 to +1.5 (bowl) | Above-average under pressure |
| −5 to +5 (bat) / −0.5 to +0.5 (bowl) | Consistent regardless of context |
| < −5 (bat) / < −0.5 (bowl) | Performance drops under pressure |

## Limitations
- 'Pressure' is defined by match-state, not by the player's psychological experience
- A small number of very high-pressure matches can dominate the metric even with thresholds applied
- The metric does not account for opposition quality in pressure situations
"""

    (docs_dir / "impact_score_methodology.md").write_text(impact_md.lstrip(), encoding="utf-8")
    (docs_dir / "clutch_contribution_methodology.md").write_text(clutch_md.lstrip(), encoding="utf-8")
    print("    [OK] impact_score_methodology.md written")
    print("    [OK] clutch_contribution_methodology.md written")
