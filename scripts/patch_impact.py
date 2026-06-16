new_body = r'''    """
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
    shrink_k = 25 if career_mode else 0

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
        bat_total   = ("bat_pts", "sum"),    # total season points
        bat_mean    = ("bat_pts", "mean"),   # per-innings mean for career
        bat_innings = ("bat_pts", "count"),
        bat_matches = ("match_id", "nunique"),
    ).reset_index()
    bat_agg = bat_agg[bat_agg["bat_innings"] >= min_inn]

    if career_mode:
        # Career: per-match mean × Bayesian shrinkage (consistent excellence)
        bat_agg["batting_impact_score"] = (
            bat_agg["bat_mean"] *
            (bat_agg["bat_matches"] / (bat_agg["bat_matches"] + shrink_k))
        ).round(3)
    else:
        # Season: total contribution (sustained performers win, 5-game wonders don't)
        bat_agg["batting_impact_score"] = bat_agg["bat_total"].round(3)

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
        ).round(3)
    else:
        # Season: total contribution
        bowl_agg["bowling_impact_score"] = bowl_agg["bowl_total"].round(3)

    # ── COMBINE ───────────────────────────────────────────────────────────────
    result = (
        bat_agg[["batter","batting_impact_score","bat_matches"]]
        .merge(
            bowl_agg[["bowler","bowling_impact_score","bowl_matches"]],
            left_on="batter", right_on="bowler", how="outer"
        )
    )
    result["player"]                = result["batter"].fillna(result["bowler"])
    result["batting_impact_score"]  = result["batting_impact_score"].fillna(0).round(3)
    result["bowling_impact_score"]  = result["bowling_impact_score"].fillna(0).round(3)
    result["combined_impact_score"] = (
        result["batting_impact_score"] + result["bowling_impact_score"]
    ).round(3)
    result = result[result["player"].notna()]

    return result[["player","batting_impact_score","bowling_impact_score",
                   "combined_impact_score","bat_matches","bowl_matches"]].reset_index(drop=True)
'''

with open('src/module_3_players.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the function body start (docstring line) and end
start = None
end = None
for i, line in enumerate(lines):
    if 'def compute_impact_score' in line:
        # body starts after the last line of the def signature
        j = i + 1
        while j < len(lines) and lines[j].strip().startswith('career_mode') or (j < len(lines) and lines[j].strip() == ''):
            j += 1
        # find the triple-quote docstring start
        while j < len(lines) and '"""' not in lines[j]:
            j += 1
        start = j  # line with opening """
        break

# Find the closing return statement of the function
for i in range(start, len(lines)):
    if 'return result[["player","batting_impact_score"' in lines[i] or \
       'return result[["player", "batting_impact_score"' in lines[i]:
        end = i
        break

print(f"Replacing lines {start+1} to {end+1} (1-indexed)")
new_lines = lines[:start] + [new_body] + lines[end+1:]

with open('src/module_3_players.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"Done. Total lines: {len(new_lines)}")
