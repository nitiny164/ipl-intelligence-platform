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

# ── Author strip — always visible at top ──────────────────────────────────────
st.markdown(
    """
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons+Round" rel="stylesheet">
    <div style="background:#FFFFFF;border:1px solid #E3E8EF;border-radius:10px;
                padding:11px 20px;margin-bottom:16px;display:flex;align-items:center;
                justify-content:space-between;flex-wrap:wrap;gap:10px;
                box-shadow:0 1px 3px rgba(0,0,0,0.04)">
      <div style="display:flex;align-items:center;gap:10px">
        <span class="material-icons-round" style="font-size:20px;color:#1565C0">person</span>
        <div>
          <span style="font-weight:700;font-size:0.92rem;color:#1A1A2E">Nitin Yadav</span>
          <span style="color:#CBD5E1;margin:0 8px">|</span>
          <span style="font-size:0.8rem;color:#546E7A">Data Analyst &amp; Deputy Manager &nbsp;·&nbsp; Shivalik Small Finance Bank, Delhi</span>
        </div>
      </div>
      <a href="https://linkedin.com/in/nitin-yadav-ny" target="_blank"
         style="display:inline-flex;align-items:center;gap:6px;text-decoration:none;
                background:#0077B5;color:white;border-radius:6px;
                padding:5px 13px;font-size:0.76rem;font-weight:600;white-space:nowrap">
        <span class="material-icons-round" style="font-size:14px">open_in_new</span>
        LinkedIn Profile
      </a>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── About me expander ─────────────────────────────────────────────────────────
with st.expander("About the Author — click to expand"):
    col_bio, col_contact = st.columns([3, 2])

    with col_bio:
        st.markdown("**Nitin Yadav**")
        st.markdown(
            "<p style='color:#546E7A;font-size:0.85rem;margin-top:-8px'>"
            "Data Analyst &amp; Deputy Manager · Shivalik Small Finance Bank · Delhi, India"
            "</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "2+ years in analytics, dashboard development, reporting automation, and predictive "
            "modelling — backed by 5+ years handling structured data across regulated environments. "
            "At Shivalik Small Finance Bank, built 10+ Power BI dashboards and Python ETL pipelines "
            "delivering RBI-compliant regulatory reporting with zero compliance breaches. "
            "This platform is a self-initiated capstone project to apply end-to-end machine learning "
            "and data engineering outside of work."
        )
        st.markdown("**Skills used in this project**")
        skills_html = "".join([
            f'<span style="background:#EEF2FF;color:#3730A3;border-radius:20px;'
            f'padding:3px 11px;font-size:0.75rem;font-weight:500;margin:2px;display:inline-block">{s}</span>'
            for s in ["Python", "Pandas", "NumPy", "PyArrow", "XGBoost",
                      "scikit-learn", "SHAP", "Plotly", "Streamlit", "Git", "SQL"]
        ])
        st.markdown(
            f'<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:4px">{skills_html}</div>',
            unsafe_allow_html=True,
        )

    with col_contact:
        st.markdown("**Contact & Links**")
        st.markdown(
            """
            <div style="display:flex;flex-direction:column;gap:10px;margin-top:4px">
              <a href="mailto:nitin19969@gmail.com"
                 style="display:flex;align-items:center;gap:8px;text-decoration:none;color:#1A1A2E;font-size:0.84rem">
                <span class="material-icons-round" style="font-size:17px;color:#1565C0">email</span>
                nitin19969@gmail.com
              </a>
              <a href="https://linkedin.com/in/nitin-yadav-ny" target="_blank"
                 style="display:flex;align-items:center;gap:8px;text-decoration:none;color:#1A1A2E;font-size:0.84rem">
                <span class="material-icons-round" style="font-size:17px;color:#0077B5">work</span>
                linkedin.com/in/nitin-yadav-ny
              </a>
              <a href="https://github.com/nitiny164/ipl-intelligence-platform" target="_blank"
                 style="display:flex;align-items:center;gap:8px;text-decoration:none;color:#1A1A2E;font-size:0.84rem">
                <span class="material-icons-round" style="font-size:17px;color:#24292E">code</span>
                github.com/nitiny164
              </a>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Education**")
        st.markdown(
            "<p style='font-size:0.82rem;color:#455A64;line-height:1.8;margin-top:4px'>"
            "MCA — Swami Vivekananda Subharti University, 2023<br>"
            "B.Sc. — IGNOU, 2021"
            "</p>",
            unsafe_allow_html=True,
        )
        st.markdown("**Certifications**")
        st.markdown(
            "<p style='font-size:0.82rem;color:#455A64;line-height:1.8;margin-top:4px'>"
            "Data Science — Ducat Institute, Noida<br>"
            "Power BI Data Analyst — CampusX"
            "</p>",
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
