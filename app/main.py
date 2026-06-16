"""
main.py — IPL Intelligence Platform home screen.
Zero business logic — navigation and welcome only.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from app.style import inject_styles, PALETTE

st.set_page_config(
    page_title="IPL Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()

# ── Project credit + contact strip — always visible at top ────────────────────
st.markdown(
    """
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons+Round" rel="stylesheet">
    <div style="background:#FFFFFF;border:1px solid #E3E8EF;border-radius:10px;
                padding:11px 20px;margin-bottom:16px;display:flex;align-items:center;
                justify-content:space-between;flex-wrap:wrap;gap:12px;
                box-shadow:0 1px 3px rgba(0,0,0,0.04)">
      <div style="display:flex;align-items:center;gap:10px">
        <span class="material-icons-round" style="font-size:20px;color:#1565C0">code</span>
        <div>
          <span style="font-size:0.8rem;color:#546E7A">Project made by</span>
          <span style="font-weight:700;font-size:0.92rem;color:#1A1A2E;margin-left:6px">Nitin Yadav</span>
          <span style="color:#CBD5E1;margin:0 8px">|</span>
          <span style="font-size:0.8rem;color:#546E7A">Currently working as Data Analyst at Shivalik Small Finance Bank</span>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
        <span style="display:inline-flex;align-items:center;gap:6px;
                  background:#F1F5F9;color:#1A1A2E;border-radius:6px;
                  padding:5px 12px;font-size:0.76rem;font-weight:600;white-space:nowrap">
          <span class="material-icons-round" style="font-size:14px;color:#1565C0">email</span>
          nitin19969@gmail.com
        </span>
        <a href="https://linkedin.com/in/nitin-yadav-ny" target="_blank"
           style="display:inline-flex;align-items:center;gap:6px;text-decoration:none;
                  background:#0077B5;color:white;border-radius:6px;
                  padding:5px 12px;font-size:0.76rem;font-weight:600;white-space:nowrap">
          <span class="material-icons-round" style="font-size:14px">work</span>
          LinkedIn
        </a>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="background:linear-gradient(135deg,#1565C0 0%,#0D47A1 100%);
                border-radius:14px;padding:36px 40px 32px;margin-bottom:2rem;color:white">
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
    ("compare_arrows", "0 — Team vs Team",           "Rivalry deep-dive — phases, venues, win trends, match-winners"),
    ("trending_up",    "1 — League Pulse",           "18-season evolution of scoring, strategy & competitiveness"),
    ("shield",         "2 — Team War Room",          "Phase analysis, rolling form, home/away, season standings"),
    ("science",        "3 — Player Performance Lab", "Impact Score, Clutch Differential, archetype quadrant"),
    ("model_training", "4 — Win Probability Engine", "Ball-by-ball ML model, SHAP explainability, improbable finishes"),
    ("manage_search",  "5 — Data Explorer",          "Universal filter bar — self-serve analytics with CSV download"),
    ("gavel",          "6 — The Verdict",            "5 falsifiable, source-linked strategic findings"),
]

CARD_S = (
    "background:#FFFFFF;border:1px solid #E3E8EF;border-radius:10px;"
    "padding:14px 18px;margin-bottom:12px;box-shadow:0 1px 4px rgba(0,0,0,0.05)"
)
cols = st.columns(2)
for i, (icon, title, desc) in enumerate(MODULES):
    with cols[i % 2]:
        st.markdown(
            f'<div style="{CARD_S}">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
            f'<span class="material-icons-round" style="font-size:18px;color:{PALETTE["primary"]}">{icon}</span>'
            f'<span style="font-weight:600;font-size:0.9rem;color:{PALETTE["text"]}">{title}</span>'
            f'</div>'
            f'<p style="margin:0;font-size:0.79rem;color:{PALETTE["muted"]};line-height:1.45">{desc}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown(
    f"<p style='color:{PALETTE['muted']};font-size:0.75rem;margin-top:1.5rem'>"
    "Data: Cricsheet / IPL official records</p>",
    unsafe_allow_html=True,
)
