# IPL Intelligence Platform

End-to-end IPL analytics platform covering 18 seasons (2008–2026), 1,212 matches, and 288,226 deliveries — built on a Parquet data layer with a calibrated ball-by-ball ML win probability model.

---

## What This Project Does

This is a 7-module interactive analytics platform that goes beyond scorecards. Every number traces back to raw ball-by-ball data. Every finding is falsifiable — you can re-run the query and challenge the result.

| Module | What It Covers |
|--------|---------------|
| **0 — Team vs Team** | Rivalry deep-dive: head-to-head history, phase battle, venue splits, key players |
| **1 — League Pulse** | 18-season evolution of scoring, strategy, toss trends, and competitiveness |
| **2 — Team War Room** | Phase radar, rolling form, home/away splits, season standings with NRR |
| **3 — Player Performance Lab** | Two original metrics: Impact Score and Clutch Differential |
| **4 — Win Probability Engine** | Ball-by-ball ML model, SHAP explainability, improbable finishes leaderboard |
| **5 — Data Explorer** | Universal filter bar — self-serve analytics with CSV export |
| **6 — The Verdict** | 5 falsifiable, source-linked strategic findings with counterfactuals |

---

## Original Metrics

### Impact Score
A composite batting + bowling point system calibrated to what actually matters in T20 cricket — boundary bonuses, milestone rewards, strike rate adjustments, and a pressure chase multiplier (RRR > 10 → ×1.5). Bayesian shrinkage (k=25) prevents small-sample inflation.

### Clutch Differential
Measures a player's performance gap between close matches (margin ≤ 20 runs / ≤ 2 wickets) versus all other matches. Positive = player raises their game under pressure. Negative = performance drops in crunch situations.

---

## Win Probability Model

- **Algorithm:** XGBoost with isotonic regression calibration
- **Training split:** Chronological (train ≤ 2023, val 2024, test 2025–26) — no data leakage
- **Test AUC:** 0.89+ across all three held-out seasons (industry benchmark: 0.82–0.85)
- **Brier Score:** 0.1364 (vs 0.25 baseline for a model that always predicts 50%)
- **Explainability:** SHAP values showing each feature's contribution to every prediction

---

## Key Findings (The Verdict)

1. **Powerplay run rates are rising faster than death overs** — slope 0.123 vs 0.070 per season
2. **Toss advantage is real but declining** — winner bats first 56% of the time, down from 70%+ in early seasons
3. **Venue is the biggest context factor** — 18% spread in chase success rate across grounds
4. **CSK and MI co-dominance** — 10 of 18 titles between two franchises (56%)
5. **Death bowling economy collapses in chases** — 1.8 extra runs per over vs. first innings

Every finding includes a "How to Challenge" panel with the exact query to run to verify or disprove it.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data | Parquet / PyArrow, Pandas, NumPy |
| ML | scikit-learn (LogReg baseline), XGBoost, SHAP, joblib |
| Visualisation | Plotly |
| App | Streamlit (multipage) |

---

## Data

- **Source:** Cricsheet ball-by-ball data + IPL official records
- **Coverage:** IPL 2008–2026 (18 seasons)
- **Scale:** 1,212 matches · 288,226 deliveries · 799 players · 16 franchises
- **Layer:** All raw CSVs processed into a Parquet layer via `src/module_0_foundation.py`

---

## Project Structure

```
IPL/
  data/
    raw/          ← Original CSVs (never modified)
    processed/    ← Parquet tables (output of module 0)
  src/
    data_loader.py
    module_0_foundation.py  ... module_6_verdict.py
  app/
    main.py       ← Streamlit entry point
    pages/        ← One file per module (multipage)
    style.py      ← Shared UI components
  models/         ← Trained XGBoost + calibration pipeline
  docs/           ← Methodology docs for both original metrics
  requirements.txt
```


Built by **Nitin** — Deputy Manager & Data Analyst  
Capstone data analytics project · 2026
