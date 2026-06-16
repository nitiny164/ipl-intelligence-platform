# Impact Score — Methodology & Worked Example

## What is Impact Score?

Impact Score measures **how much a player contributed to their team's chances of winning**, going far beyond raw stats. Unlike batting average or economy rate, it rewards:

- Scoring big in pressure chases
- Taking high-quality wickets (not just tail-enders)
- Sustaining excellence across a full season or career

It is inspired by **Dream11 fantasy points** — granular, event-driven, and context-aware.

---

## Batting Impact Score

### Points System (per innings)

| Event | Points |
|-------|--------|
| Every run scored | +0.5 |
| Every boundary (4) | +1 bonus |
| Every six | +2 bonus |
| 25-run milestone | +4 |
| 50-run milestone | +8 |
| 75-run milestone | +12 |
| Century (100+) | +16 |
| Duck (dismissed for 0) | −4 |

> Milestone tiers are exclusive — only the highest applies. A century gets +16, not +4+8+12+16.

### Strike Rate Adjustment (minimum 10 balls faced)

| Strike Rate | Adjustment |
|-------------|------------|
| ≥ 170 | +6 |
| 150–169 | +4 |
| 130–149 | +2 |
| 100–129 | 0 |
| 90–99 | −2 |
| 70–89 | −4 |
| < 70 | −6 |

### Pressure Context Multiplier

When a batter faces a high-pressure chase, all points are amplified:

| Situation | Multiplier |
|-----------|------------|
| Chasing, Required Run Rate > 10 | ×1.5 |
| Chasing, Required Run Rate 8–10 | ×1.25 |
| All other situations | ×1.0 |

---

## Bowling Impact Score

### Wicket Points

| Event | Points |
|-------|--------|
| Wicket (base value) | +16 |
| Dismissed a premium batter (career SR > 130) | +8 bonus |
| Bowled or LBW dismissal (skill dismissal) | +4 bonus |
| 3-wicket haul in a match | +4 bonus |
| 4-wicket haul in a match | +8 bonus |
| 5-wicket haul in a match | +16 bonus |
| Maiden over | +8 per maiden |

> Example: 3/20 where all are Bowled against top-order batters: 3×(16+8+4) + 4 (haul) = **88 pts from wickets alone**

### Economy Rate Points (per match spell, minimum 1 full over)

| Economy (Runs per Over) | Points |
|-------------------------|--------|
| < 5.0 | +10 |
| 5.0 – 5.9 | +6 |
| 6.0 – 6.9 | +2 |
| 7.0 – 7.9 | 0 (league average zone) |
| 8.0 – 8.9 | −4 |
| 9.0 – 9.9 | −8 |
| ≥ 10.0 | −12 |

> IPL league average is ~8.5 RPO. Anything below 7.0 is genuinely good bowling.

### Pressure Context Multiplier

| Situation | Multiplier |
|-----------|------------|
| Team defending a low total (1st innings < 160 runs) | ×1.3 |
| Bowling death overs in a chase with RRR > 9 | ×1.3 |
| All other situations | ×1.0 |

---

## Aggregation: Career vs Season

The two views answer **different cricket questions** and intentionally use different scales.

### Career view — Sustained excellence (score range: ~15–45)

**Score = Per-match mean × Bayesian shrinkage weight (k = 25)**

*"Who is the most consistently excellent player all-time?"*

| Player | Career matches | Per-match avg | Weight | Career Impact |
|--------|---------------|---------------|--------|--------------|
| Kohli | 275 | 30 pts | 275/300 = **92%** | **27.6** |
| Suryavanshi | 23 | 52 pts | 23/48 = **48%** | **24.8** |

Kohli's 275 matches of sustained excellence correctly ranks him above a hot debutant with only 23 games. The shrinkage factor k=25 means you need ~25 matches before your average is given full weight.

### Season view — Total season contribution (score range: ~200–1000+)

**Score = Sum of all match points across the season**

*"Who contributed the most this season in total?"*

| Player | Season matches | Per-match avg | Season Total |
|--------|---------------|---------------|-------------|
| Kohli 2016 | 16 | ~56 pts/match | **~895** |
| Good player | 5 | ~70 pts/match | **~350** |

Total scoring means a player who plays 16 matches and contributes consistently *must* outscore someone who played 5 brilliant games. This correctly mirrors how IPL seasons work — availability and sustained performance both matter.

**Why are the scales different?** Career scores (~30) vs Season scores (~895) are not directly comparable — they measure different things. Career = average excellence per match; Season = total season contribution. Both are correct within their own context.

---

## Minimum Thresholds

| View | Batting minimum | Bowling minimum |
|------|----------------|-----------------|
| Career | 20 innings | 30 overs bowled |
| Season | 3 innings | 4 overs bowled |

These filters exclude players who batted at #9 with 2 balls faced, or bowlers who bowled a single over as an experiment. Only genuine contributors qualify.

---

## Worked Example: Virat Kohli, IPL 2016

Kohli's 2016 season is the greatest individual batting season in IPL history — 973 runs across 16 matches, a record that still stands.

**Sample innings calculation:**

| Innings | Runs | SR | 4s | 6s | pts_runs | pts_4s | pts_6s | Milestone | SR Bonus | **Total** |
|---------|------|----|----|----|---------:|-------:|-------:|----------:|---------:|----------:|
| 100 off 63 | 100 | 159 | 7 | 4 | 50.0 | 7 | 8 | +16 | +4 | **85** |
| 75 off 51 | 75 | 147 | 5 | 3 | 37.5 | 5 | 6 | +12 | +2 | **62.5** |
| 54 off 35 | 54 | 154 | 5 | 2 | 27.0 | 5 | 4 | +8 | +4 | **48** |
| 33 off 28 | 33 | 118 | 2 | 1 | 16.5 | 2 | 2 | +4 | 0 | **24.5** |
| 8 off 10 | 8 | 80 | 1 | 0 | 4.0 | 1 | 0 | 0 | −4 | **1** |

Many innings were high-pressure chases (RCB chased frequently in 2016), applying ×1.25–1.5 multipliers.

**Season total: ~895 Batting Impact** — #1 in IPL 2016, as expected.

---

## Worked Example: Jasprit Bumrah, Career

**Typical match (4 overs, 2/22, both premium batters, one Bowled):**

| Component | Calculation | Points |
|-----------|-------------|--------|
| Wicket 1 (premium, Bowled) | 16 + 8 + 4 | 28 |
| Wicket 2 (premium) | 16 + 8 | 24 |
| Economy (5.5 RPO) | +6 band | 6 |
| **Match total** | | **58** |

**Career aggregation:**
- Per-match mean across 158 matches: ~33 pts (after accounting for expensive days and T20 luck)
- Shrinkage weight: 158 / (158+25) = **86%**
- **Career Bowling Impact: ~33** — consistently one of the top 3 bowlers all-time in this metric

---

## Why This Is Better Than Traditional Stats

| Problem with traditional stats | How Impact Score fixes it |
|-------------------------------|--------------------------|
| Average hides failures (one big score) | Season total rewards all-match consistency |
| Small samples inflate averages | Bayesian shrinkage for career view |
| All wickets treated equally | +8 bonus for premium batter, +4 for Bowled/LBW |
| No pressure context | ×1.3–1.5 multipliers for chases and low-total defence |
| Strike rate is separate from average | SR band adjustment built into every innings score |
| 5-match wonder ranks above 200-match legend | Shrinkage weight proportional to career length |
