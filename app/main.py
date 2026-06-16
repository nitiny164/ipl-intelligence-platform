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
    "Data: Cricsheet / IPL official records</p>",
    unsafe_allow_html=True,
)

# ── About the Author ──────────────────────────────────────────────────────────
st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="background:#FFFFFF;border:1px solid #E3E8EF;border-radius:12px;
                padding:24px 28px;box-shadow:0 1px 4px rgba(0,0,0,0.05)">

      <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px">
        <span class="material-icons-round" style="font-size:20px;color:#1565C0">person</span>
        <span style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                     letter-spacing:0.08em;color:#546E7A">About the Author</span>
      </div>

      <div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:flex-start;gap:20px">

        <div style="flex:1;min-width:260px">
          <p style="margin:0 0 4px;font-size:1.1rem;font-weight:700;color:#1A1A2E">Nitin Yadav</p>
          <p style="margin:0 0 12px;font-size:0.85rem;color:#546E7A">
            Data Analyst &amp; Deputy Manager &nbsp;·&nbsp; Shivalik Small Finance Bank &nbsp;·&nbsp; Delhi, India
          </p>
          <p style="margin:0;font-size:0.82rem;color:#455A64;line-height:1.7">
            2+ years in analytics, dashboard development, reporting automation, and predictive modelling.
            Built 10+ Power BI dashboards and Python ETL pipelines delivering RBI-compliant regulatory
            reporting at a small finance bank. This platform is a self-initiated capstone project to
            apply end-to-end ML and data engineering outside of work.
          </p>
        </div>

        <div style="flex:0 0 auto;min-width:200px">
          <p style="margin:0 0 10px;font-size:0.72rem;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.07em;color:#546E7A">Skills used in this project</p>
          <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:20px">
            """ + "".join([
                f'<span style="background:#EEF2FF;color:#3730A3;border-radius:20px;'
                f'padding:3px 10px;font-size:0.75rem;font-weight:500">{s}</span>'
                for s in ["Python", "Pandas", "XGBoost", "SHAP", "Plotly",
                          "Streamlit", "PyArrow", "scikit-learn", "Git"]
            ]) + """
          </div>
          <p style="margin:0 0 10px;font-size:0.72rem;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.07em;color:#546E7A">Connect</p>
          <div style="display:flex;flex-direction:column;gap:7px">
            <a href="mailto:nitin19969@gmail.com"
               style="display:flex;align-items:center;gap:7px;text-decoration:none;color:#1A1A2E;font-size:0.82rem">
              <span class="material-icons-round" style="font-size:16px;color:#1565C0">email</span>
              nitin19969@gmail.com
            </a>
            <a href="https://linkedin.com/in/nitin-yadav-ny" target="_blank"
               style="display:flex;align-items:center;gap:7px;text-decoration:none;color:#1A1A2E;font-size:0.82rem">
              <span class="material-icons-round" style="font-size:16px;color:#0077B5">work</span>
              linkedin.com/in/nitin-yadav-ny
            </a>
            <a href="https://github.com/nitiny164/ipl-intelligence-platform" target="_blank"
               style="display:flex;align-items:center;gap:7px;text-decoration:none;color:#1A1A2E;font-size:0.82rem">
              <span class="material-icons-round" style="font-size:16px;color:#24292E">code</span>
              github.com/nitiny164
            </a>
          </div>
        </div>

      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
