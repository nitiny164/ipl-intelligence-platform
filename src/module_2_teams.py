"""
module_2_teams.py — Team War Room analytics functions.

Business question: "For any team, what is their competitive identity — where are they
strong, where do they leak, and what should they do differently?"
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper: resolve team_id to name
# ---------------------------------------------------------------------------
def id_to_name(teams: pd.DataFrame) -> dict[int, str]:
    return dict(zip(teams["team_id"], teams["team_name"]))


# ---------------------------------------------------------------------------
# 1. Head-to-head matrix
# ---------------------------------------------------------------------------
def head_to_head_matrix(matches: pd.DataFrame, teams: pd.DataFrame,
                         season_min: int = 2008, season_max: int = 2026) -> pd.DataFrame:
    """
    Returns a pivot table: rows = team_a, cols = team_b, values = win % of team_a vs team_b.
    Only completed matches, filtered by season range. Uses lineage_id view via teams table.
    """
    id2name = id_to_name(teams)
    completed = matches[
        matches["is_completed"] &
        matches["season"].between(season_min, season_max)
    ].copy()

    completed["team1_name"] = completed["team1"].map(id2name)
    completed["team2_name"] = completed["team2"].map(id2name)
    completed["winner_name"] = completed["match_winner"].map(id2name)

    # Build match pairs (both orderings)
    rows = []
    for _, row in completed.iterrows():
        t1, t2, winner = row["team1_name"], row["team2_name"], row["winner_name"]
        if pd.isna(t1) or pd.isna(t2) or pd.isna(winner):
            continue
        rows.append({"team_a": t1, "team_b": t2, "team_a_won": winner == t1})
        rows.append({"team_a": t2, "team_b": t1, "team_a_won": winner == t2})

    pairs = pd.DataFrame(rows)
    h2h = (
        pairs.groupby(["team_a", "team_b"])
        .agg(matches=("team_a_won", "count"), wins=("team_a_won", "sum"))
        .reset_index()
    )
    h2h["win_pct"] = (h2h["wins"] / h2h["matches"] * 100).round(1)

    pivot_pct = h2h.pivot(index="team_a", columns="team_b", values="win_pct")
    pivot_matches = h2h.pivot(index="team_a", columns="team_b", values="matches")
    return pivot_pct, pivot_matches


# ---------------------------------------------------------------------------
# 2. Team home vs away win %
# ---------------------------------------------------------------------------
def home_away_record(matches: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per team with home_win_pct and away_win_pct.
    'Home' = team's most frequently used venue per season.
    """
    id2name = id_to_name(teams)
    completed = matches[matches["is_completed"]].copy()
    completed["team1_name"] = completed["team1"].map(id2name)
    completed["team2_name"] = completed["team2"].map(id2name)
    completed["winner_name"] = completed["match_winner"].map(id2name)

    # Build per-season home venue for each team (most frequent venue)
    venue_counts = (
        pd.concat([
            completed[["season", "team1_name", "venue"]].rename(columns={"team1_name": "team"}),
            completed[["season", "team2_name", "venue"]].rename(columns={"team2_name": "team"}),
        ])
        .groupby(["season", "team", "venue"])
        .size()
        .reset_index(name="count")
    )
    home_venues = (
        venue_counts.sort_values("count", ascending=False)
        .groupby(["season", "team"])
        .first()
        .reset_index()[["season", "team", "venue"]]
        .rename(columns={"venue": "home_venue"})
    )

    # Flatten all matches to (team, opponent, venue, won)
    rows = []
    for _, row in completed.iterrows():
        t1, t2, winner = row["team1_name"], row["team2_name"], row["winner_name"]
        if pd.isna(t1) or pd.isna(t2) or pd.isna(winner):
            continue
        rows.append({"team": t1, "season": row["season"], "venue": row["venue"], "won": winner == t1})
        rows.append({"team": t2, "season": row["season"], "venue": row["venue"], "won": winner == t2})

    all_matches = pd.DataFrame(rows).merge(home_venues, on=["season", "team"], how="left")
    all_matches["is_home"] = all_matches["venue"] == all_matches["home_venue"]

    result = (
        all_matches.groupby(["team", "is_home"])
        .agg(matches=("won", "count"), wins=("won", "sum"))
        .reset_index()
    )
    result["win_pct"] = (result["wins"] / result["matches"] * 100).round(1)

    home = result[result["is_home"]][["team", "win_pct", "matches"]].rename(
        columns={"win_pct": "home_win_pct", "matches": "home_matches"}
    )
    away = result[~result["is_home"]][["team", "win_pct", "matches"]].rename(
        columns={"win_pct": "away_win_pct", "matches": "away_matches"}
    )
    return home.merge(away, on="team").sort_values("home_win_pct", ascending=False)


# ---------------------------------------------------------------------------
# 3. Phase-wise batting & bowling strength
# ---------------------------------------------------------------------------
def team_phase_profile(deliveries: pd.DataFrame, teams: pd.DataFrame,
                        team_id: int, season_min: int = 2008, season_max: int = 2026) -> dict:
    """
    Returns batting and bowling metrics per phase for a single team.
    batting: run_rate, boundary_pct, dot_pct per phase
    bowling: economy, wicket_rate, dot_pct per phase
    """
    d = deliveries[
        (~deliveries["is_super_over"]) &
        (deliveries["is_legal_ball"]) &
        (deliveries["season_id"].between(season_min, season_max))
    ].copy()

    def phase_metrics(df: pd.DataFrame) -> pd.DataFrame:
        df["is_boundary"] = df["batter_runs"].isin([4, 6])
        df["is_dot"] = df["batter_runs"] == 0
        grp = df.groupby("over_phase").agg(
            balls=("batter_runs", "count"),
            runs=("batter_runs", "sum"),
            wickets=("is_wicket", "sum"),
            boundaries=("is_boundary", "sum"),
            dots=("is_dot", "sum"),
        ).reset_index()
        grp["run_rate"] = (grp["runs"] / grp["balls"] * 6).round(2)
        grp["economy"] = (grp["runs"] / grp["balls"] * 6).round(2)
        grp["wicket_rate"] = (grp["wickets"] / grp["balls"] * 6).round(3)
        grp["boundary_pct"] = (grp["boundaries"] / grp["balls"] * 100).round(1)
        grp["dot_pct"] = (grp["dots"] / grp["balls"] * 100).round(1)
        return grp

    batting_df = phase_metrics(d[d["team_batting"] == team_id])
    bowling_df = phase_metrics(d[d["team_bowling"] == team_id])

    return {"batting": batting_df, "bowling": bowling_df}


# ---------------------------------------------------------------------------
# 4. Rolling form (last-N-match win %)
# ---------------------------------------------------------------------------
def rolling_form(matches: pd.DataFrame, teams: pd.DataFrame,
                  team_id: int, window: int = 10,
                  season_min: int = None, season_max: int = None) -> pd.DataFrame:
    """
    Returns a chronological series of the team's rolling win % over the last `window` matches.
    """
    completed = matches[matches["is_completed"]].copy()
    if season_min is not None:
        completed = completed[completed["season"] >= season_min]
    if season_max is not None:
        completed = completed[completed["season"] <= season_max]
    team_matches = completed[
        (completed["team1"] == team_id) | (completed["team2"] == team_id)
    ].sort_values("match_date").copy()

    team_matches["won"] = team_matches["match_winner"] == team_id
    team_matches["rolling_win_pct"] = (
        team_matches["won"].rolling(window=window, min_periods=1).mean() * 100
    )
    id2name = id_to_name(teams)
    team_matches["opponent"] = team_matches.apply(
        lambda r: id2name.get(r["team2"] if r["team1"] == team_id else r["team1"], "Unknown"),
        axis=1
    )
    return team_matches[["match_date", "season", "opponent", "won", "rolling_win_pct"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Season standings reconstruction
# ---------------------------------------------------------------------------
def season_standings(matches: pd.DataFrame, teams: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Reconstructed IPL points table for a given season.
    Points: Win=2, Loss=0, Tie=1, No Result=1.
    NRR computed as (total runs scored / total overs faced) - (total runs conceded / total overs bowled).
    """
    id2name = id_to_name(teams)
    season_m = matches[matches["season"] == season].copy()

    standings: dict[int, dict] = {}

    def ensure(tid):
        if tid not in standings:
            standings[tid] = dict(team_id=tid, name=id2name.get(tid, str(tid)),
                                  played=0, won=0, lost=0, tied=0, nr=0, points=0,
                                  runs_for=0, overs_for=0.0, runs_against=0, overs_against=0.0)

    for _, row in season_m.iterrows():
        t1, t2 = int(row["team1"]), int(row["team2"])
        ensure(t1); ensure(t2)
        standings[t1]["played"] += 1
        standings[t2]["played"] += 1

        result = str(row.get("result", "")).lower().strip()
        if result == "win":
            winner = int(row["match_winner"]) if pd.notna(row["match_winner"]) else None
            loser = t2 if winner == t1 else t1
            if winner:
                standings[winner]["won"] += 1
                standings[winner]["points"] += 2
                standings[loser]["lost"] += 1
        elif result == "tie":
            standings[t1]["tied"] += 1; standings[t1]["points"] += 1
            standings[t2]["tied"] += 1; standings[t2]["points"] += 1
        elif result == "no result":
            standings[t1]["nr"] += 1; standings[t1]["points"] += 1
            standings[t2]["nr"] += 1; standings[t2]["points"] += 1

        # NRR: first-innings total from match
        fi = row.get("first_innings_total", np.nan)
        target = row.get("target", np.nan)
        if pd.notna(fi) and result == "win":
            winner = int(row["match_winner"]) if pd.notna(row["match_winner"]) else None
            # Simplified: assume 20 overs each (real NRR needs per-ball data; this is close enough)
            if winner:
                loser = t2 if winner == t1 else t1
                # Determine which team batted first
                # team1 always bats first in IPL (toss winner decides)
                # We'll use win_by_runs/wickets to infer
                if pd.notna(row.get("win_by_runs")) and row["win_by_runs"] > 0:
                    # team batting first won
                    first_bat = winner
                else:
                    first_bat = loser
                second_bat = t2 if first_bat == t1 else t1
                standings[first_bat]["runs_for"] += float(fi)
                standings[first_bat]["overs_for"] += 20.0
                standings[second_bat]["runs_against"] += float(fi)
                standings[second_bat]["overs_against"] += 20.0
                if pd.notna(target):
                    standings[second_bat]["runs_for"] += float(target) - 1
                    standings[second_bat]["overs_for"] += 20.0
                    standings[first_bat]["runs_against"] += float(target) - 1
                    standings[first_bat]["overs_against"] += 20.0

    df = pd.DataFrame(list(standings.values()))
    df["nrr"] = np.where(
        (df["overs_for"] > 0) & (df["overs_against"] > 0),
        df["runs_for"] / df["overs_for"] - df["runs_against"] / df["overs_against"],
        0.0
    )
    df = df.sort_values(["points", "nrr"], ascending=False).reset_index(drop=True)
    df.index += 1
    df["nrr"] = df["nrr"].round(3)
    return df[["name", "played", "won", "lost", "tied", "nr", "points", "nrr"]].rename(columns={"name": "Team"})


# ---------------------------------------------------------------------------
# 6. Venue toss recommender
# ---------------------------------------------------------------------------
def toss_recommender(matches: pd.DataFrame, venue: str, min_matches: int = 5) -> dict:
    """
    For a given venue, recommend bat or field based on historical chase success rate.
    Returns a dict with recommendation, confidence (sample size), and supporting stats.
    """
    v_matches = matches[(matches["venue"] == venue) & matches["is_completed"]].copy()
    n = len(v_matches)

    if n == 0:
        return {"venue": venue, "recommendation": "No data", "sample": 0, "low_sample": True}

    v_matches["chase_won"] = v_matches["win_by_wickets"].notna() & (v_matches["win_by_wickets"] > 0)
    chase_pct = v_matches["chase_won"].mean() * 100
    defend_pct = 100 - chase_pct

    recommendation = "Field First (Chase)" if chase_pct >= 50 else "Bat First (Defend)"
    confidence_label = "High" if n >= 20 else ("Medium" if n >= 10 else "Low")

    return {
        "venue": venue,
        "recommendation": recommendation,
        "chase_success_pct": round(chase_pct, 1),
        "defend_success_pct": round(defend_pct, 1),
        "avg_first_innings_score": round(v_matches["first_innings_total"].mean(), 1),
        "avg_target": round(v_matches["target"].mean(), 1),
        "sample": n,
        "low_sample": n < min_matches,
        "confidence": confidence_label,
    }


# ---------------------------------------------------------------------------
# 7. Team overall win % by season
# ---------------------------------------------------------------------------
def team_season_record(matches: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """All teams' win % per season — for multi-line form chart."""
    id2name = id_to_name(teams)
    completed = matches[matches["is_completed"]].copy()

    rows = []
    for _, row in completed.iterrows():
        t1, t2, winner = int(row["team1"]), int(row["team2"]), row["match_winner"]
        if pd.isna(winner):
            continue
        winner = int(winner)
        rows.append({"season": row["season"], "team_id": t1, "won": winner == t1})
        rows.append({"season": row["season"], "team_id": t2, "won": winner == t2})

    df = pd.DataFrame(rows)
    result = df.groupby(["season", "team_id"]).agg(
        played=("won", "count"), wins=("won", "sum")
    ).reset_index()
    result["win_pct"] = (result["wins"] / result["played"] * 100).round(1)
    result["team_name"] = result["team_id"].map(id2name)
    return result


# ---------------------------------------------------------------------------
# 8. Batting collapse propensity
# ---------------------------------------------------------------------------
def collapse_propensity(deliveries: pd.DataFrame, teams: pd.DataFrame,
                         season_min: int = 2008, season_max: int = 2026) -> pd.DataFrame:
    """
    Batting collapse = 3+ wickets falling within a 12-ball window in an innings.
    Returns per-team collapse rate (collapses per 100 innings).
    """
    id2name = id_to_name(teams)
    d = deliveries[
        (~deliveries["is_super_over"]) &
        (deliveries["season_id"].between(season_min, season_max))
    ].copy()

    d_wkt = d[d["is_wicket"]].copy()

    # For each innings, find groups of 3+ wickets within 12-ball window
    collapses = []
    for (match_id, team_id), grp in d_wkt.groupby(["match_id", "team_batting"]):
        ball_nums = grp["legal_balls_bowled"].values
        if len(ball_nums) < 3:
            continue
        # Sliding window: any 3 consecutive wickets within 12 balls
        for i in range(len(ball_nums) - 2):
            if ball_nums[i + 2] - ball_nums[i] <= 12:
                collapses.append(team_id)
                break  # count at most one collapse per innings

    collapse_counts = pd.Series(collapses).value_counts().reset_index()
    collapse_counts.columns = ["team_id", "collapses"]

    # Total batting innings per team = number of distinct matches where team batted
    # (each team bats exactly once per match in T20)
    innings_counts = (
        d.groupby(["match_id", "team_batting"])
        .size()
        .reset_index()[["team_batting"]]
        .value_counts()
        .reset_index()
    )
    innings_counts.columns = ["team_id", "innings_count"]

    result = innings_counts.merge(collapse_counts, on="team_id", how="left")
    result["collapses"] = result["collapses"].fillna(0)
    result["collapse_per_100_innings"] = (result["collapses"] / result["innings_count"] * 100).round(1)
    result["team_name"] = result["team_id"].map(id2name)
    return result.sort_values("collapse_per_100_innings", ascending=False).reset_index(drop=True)
