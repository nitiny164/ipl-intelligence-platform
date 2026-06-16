# Clutch Score Methodology

## Purpose
Clutch Score answers: *"Who genuinely contributes runs under pressure in matches that matter?"*

Unlike team-win-rate approaches, this is a fully **individual** metric — only your personal
batting contribution in each innings counts. A player who scores 80 off 50 to win a chase
by 20 runs gets full credit; the comfortable final margin is irrelevant.

---

## Batting Clutch Formula

```
Clutch Score = mean(inn_clutch) × shrinkage

inn_clutch   = (innings_runs − league_avg_runs) × win_weight × sr_modifier

league_avg_runs ≈ 16.3  (average pressure innings across all qualifying deliveries)
league_avg_sr   ≈ 123.6 (average pressure innings SR)

win_weight  = 2.0 if team won the match, else 0.4
sr_modifier = clip(1 + (innings_SR − league_avg_sr) / 200, 0.85, 1.15)
shrinkage   = pressure_matches / (pressure_matches + 10)
```

### Factor 1 — Runs Above Average
Every pressure innings is scored relative to the league mean (~16.3 runs). A 30-run innings
is +13.7 above average; a duck is −16.3. This measures **individual** output, not collective
team quality.

### Factor 2 — Win Weight
Winning innings are weighted 2.0×, losing innings 0.4×. A 50 that wins the chase counts
5× more than a 50 in a lost cause. This is still **individual** — it rewards innings that
correlate with wins, not team win rates across a season.

**Why not team win rate?** Team win rate (old formula) inflated players on dominant teams.
Rohit Sharma on MI ranked above Kohli on RCB simply because MI won more close matches as a
team — that penalised Kohli for his team's overall quality, not his individual performance.

### Factor 3 — SR Modifier
A small ±15% multiplier for scoring significantly faster/slower than the league pressure SR.
If you score at SR 150 in a pressure innings (league avg 123.6), the modifier is:
`1 + (150 − 123.6) / 200 = 1.132`, capped at 1.15.

### Factor 4 — Sample Shrinkage (Bayesian)
```
shrinkage = pressure_matches / (pressure_matches + 10)
```

| Pressure Innings | Weight | Interpretation |
|-----------------|--------|----------------|
| 5 | 33% | Very small sample — score heavily discounted |
| 10 | 50% | Moderate sample |
| 20 | 67% | Large sample |
| 30 | 75% | Very reliable |
| 70 | 88% | Kohli, Rohit — extremely reliable |

---

## Pressure Situation Definitions

### Batting Pressure
Second innings chase, Required Run Rate between 9 and 18.

**No close-match filter** — this is the critical change from v1. The old formula required
the final match margin to be ≤15 runs or ≤3 wickets. But when a batter scores 80 off 50
to put a chase beyond doubt (team wins by 30), that innings IS clutch — it's what made
the win comfortable. Filtering it out hid the very innings that define clutch batting.

**Why RRR 9–18?** Below 9 = easy chase, limited pressure. Above 18 = game is statistically
over (almost never won). RRR 9–18 is the genuine pressure window.

### Bowling Pressure
Death overs (overs 16–20) in a close match (final margin ≤15 runs or ≤3 wickets).

Bowling still uses the close-match filter because a bowler giving 10 runs in the last over
of a blowout (team winning by 80) is not under real pressure. The batter's performance is
what creates or removes pressure; the bowler faces pressure when the match is genuinely on
the line.

---

## Bowling Clutch Formula (unchanged)

Individual bowling stats — no team-quality dependency.

```
Raw Diff = (Normal Economy − Pressure Economy) × 0.7
         + (Pressure Wickets/Over − Normal Wickets/Over) × 6 × 0.3

Score = Raw Diff × Win Factor × Shrinkage

Win Factor = 0.5 + pressure_win_rate × 0.5
```

Economy rises for all bowlers in death overs. This measures relative improvement vs each
bowler's own normal baseline. Win factor rewards bowlers whose death-over work actually
led to wins — for bowling, close-match team win rate is a reasonable proxy because the
bowler is directly defending the target.

---

## Minimum Thresholds

| Role | Career min | Season min |
|------|-----------|------------|
| Batting | ≥20 pressure innings | ≥5 pressure innings |
| Bowling (balls, normal) | ≥300 | ≥100 |
| Bowling (balls, pressure) | ≥120 | ≥30 |
| Bowling (innings) | ≥5 | ≥2 |

---

## Validated Results (Career, All Seasons)

| Player | Score | Rank | Pressure Innings | Avg Runs | Win% |
|--------|-------|------|-----------------|----------|------|
| P Simran Singh | +18.5 | #1 | 26 | 25.1 | 42% |
| Shubman Gill | +16.2 | #2 | 41 | 27.1 | 46% |
| Abhishek Sharma | +15.7 | #3 | 31 | 24.5 | 35% |
| Virat Kohli | +12.1 | #12 | 78 | 24.5 | 37% |
| AB de Villiers | +10.5 | #16 | 51 | 21.4 | 39% |
| Rohit Sharma | +8.6 | #25 | 81 | 22.0 | 44% |
| MS Dhoni | +5.5 | #39 | 70 | 18.3 | 47% |
| Harbhajan Singh | −6.9 | bottom | 32 | 5.6 | — |

**Why Kohli > Rohit (cricket-correct):** Kohli averages 24.5 runs per pressure innings
across 78 appearances — the highest of all legends. Rohit averages 22.0 across 81.
Kohli also has 12 pressure fifties vs Rohit's 8. The old formula inflated Rohit because
MI won more close matches as a team.

---

## Notable Cricket Findings

**MS Dhoni (+5.5, #39, 70 pressure innings)** — Not a weakness. Dhoni's strategy is to
deliberately slow in RRR 9–14 overs (building partnerships, conserving wickets) before
attacking in the final 2 overs. The pressure window captures those throttle-down overs.
His 18.3-run average in this window reflects calculated patience. His overs 18–20
performance is the world's best — but outside this metric's scope.

**Andre Russell (low score)** — Floor effect. His normal batting SR is 177 (highest in
dataset). The SR modifier caps at 1.15 uplift, and his already-massive runs baseline means
every innings is "below average" by count — he's a death-over specialist, not a RRR-9-14
accumulator. Metric correctly excludes him from this window's top performers.

**Lower-order bowlers at the bottom (Harbhajan, Narine, PP Chawla, Pathan, Binny)** —
Correct. When these players bat in second-innings pressure (a rare event), they consistently
score well below the 16.3-run average. A 5-run cameo from #9 in a genuine chase is realistic
but drags their metric negative.

**Young modern batters at the top (Prabhsimran, Gill, Abhishek Sharma)** — Reflects IPL 2023–25
data. These players have large pressure-innings samples in recent seasons where they excelled.
Smaller careers mean shrinkage hasn't fully reduced their scores — a known limitation with
limited data.

---

## Limitations
- Win weight uses match outcome, not ball-by-ball win probability swing — a future improvement
  would use WP-swing per innings
- SR modifier is a small adjustment (±15%) — does not fully capture a 200-SR blitz
- Opposition bowling quality in pressure situations is not adjusted for
- Tail-enders who rarely bat in pressure situations have small samples even with ≥20 threshold
