"""
main.py — IPL Intelligence Platform home screen.
Zero business logic — navigation and welcome only.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from app.style import inject_styles, PALETTE, stat_card

st.set_page_config(
    page_title="IPL Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons+Round" rel="stylesheet">
    <div style="
        background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
        border-radius: 14px;
        padding: 36px 40px 32px;
        margin-bottom: 2rem;
        color: white;
    ">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">
            <span class="material-icons-round" style="font-size:40px;opacity:0.9">sports_cricket</span>
            <h1 style="color:white;margin:0;font-size:2rem;font-weight:700;letter-spacing:-0.02em">
                IPL Intelligence Platform
            </h1>
        </div>
        <p style="margin:0;opacity:0.85;font-size:1rem;font-weight:400">
            18 Seasons &nbsp;·&nbsp; 1,212 Matches &nbsp;·&nbsp; 288,226 Deliveries &nbsp;·&nbsp; 799 Players
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Module grid ───────────────────────────────────────────────────────────────
MODULES = [
    ("compare_arrows", "0 — Team vs Team",            "Rivalry deep-dive — phases, venues, win trends, match-winners"),
    ("trending_up",    "1 — League Pulse",            "18-season evolution of scoring, strategy & competitiveness"),
    ("shield",         "2 — Team War Room",           "Phase analysis, rolling form, home/away, season standings"),
    ("science",        "3 — Player Performance Lab",  "Impact Score, Clutch Differential, archetype quadrant"),
    ("model_training", "4 — Win Probability Engine",  "Ball-by-ball ML model, SHAP explainability, improbable finishes"),
    ("manage_search",  "5 — Data Explorer",           "Universal filter bar — self-serve analytics with CSV download"),
    ("gavel",          "6 — The Verdict",             "5 falsifiable, source-linked strategic findings"),
]

cols = st.columns(2)
CARD_S = "background:#FFFFFF;border:1px solid #E3E8EF;border-radius:10px;padding:14px 18px;margin-bottom:12px;box-shadow:0 1px 4px rgba(0,0,0,0.05)"
for i, (icon, title, desc) in enumerate(MODULES):
    with cols[i % 2]:
        st.markdown(
            f"""
            <div style="{CARD_S}">
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
                    <span class="material-icons-round" style="font-size:18px;color:{PALETTE['primary']}">{icon}</span>
                    <span style="font-weight:600;font-size:0.9rem;color:{PALETTE['text']}">{title}</span>
                </div>
                <p style="margin:0;font-size:0.79rem;color:{PALETTE['muted']};line-height:1.45">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    f"<p style='color:{PALETTE['muted']};font-size:0.75rem;margin-top:1.5rem'>"
    "Built as a capstone data analytics project · Data: Cricsheet / IPL official records</p>",
    unsafe_allow_html=True,
)
