"""
Shared style utilities for IPL Intelligence Platform.
Streamlit 1.25 compatible: uses <link> tags + inline styles only.
Global theming is handled by .streamlit/config.toml.
"""
import streamlit as st

MATERIAL_ICONS_CDN = "https://fonts.googleapis.com/icon?family=Material+Icons+Round"
GOOGLE_FONTS_CDN   = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"

PALETTE = {
    "primary":   "#1565C0",
    "secondary": "#0D47A1",
    "accent":    "#1976D2",
    "success":   "#2E7D32",
    "warning":   "#F57F17",
    "danger":    "#C62828",
    "text":      "#1A1A2E",
    "muted":     "#546E7A",
    "surface":   "#FFFFFF",
    "bg":        "#F8F9FC",
    "border":    "#E3E8EF",
}

CHART_COLORS = [
    "#1565C0", "#2E7D32", "#C62828", "#E65100",
    "#4527A0", "#00695C", "#AD1457", "#37474F",
]

# Inline style strings for reuse
_CARD_STYLE = (
    "background:#FFFFFF;border:1px solid #E3E8EF;border-radius:10px;"
    "padding:16px 20px;box-shadow:0 1px 4px rgba(0,0,0,0.06);margin-bottom:0"
)
_CARD_LABEL = (
    "font-size:0.7rem;font-weight:600;letter-spacing:0.07em;"
    "text-transform:uppercase;color:#546E7A;margin-bottom:4px"
)
_CARD_VALUE = "font-size:1.5rem;font-weight:700;color:#1A1A2E"


def inject_styles() -> None:
    """
    Load Google Fonts and Material Icons via <link> tags.
    <link> tags work with st.markdown unsafe_allow_html in Streamlit 1.25.
    Global colors/font are set via .streamlit/config.toml.
    """
    st.markdown(
        f'<link href="{GOOGLE_FONTS_CDN}" rel="stylesheet">'
        f'<link href="{MATERIAL_ICONS_CDN}" rel="stylesheet">',
        unsafe_allow_html=True,
    )


def page_header(icon: str, title: str, caption: str = "") -> None:
    """Material Icon + bold title + optional caption."""
    cap_html = (
        f'<p style="color:#546E7A;font-size:0.85rem;margin:4px 0 1rem">{caption}</p>'
        if caption else ""
    )
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
            <span class="material-icons-round"
                  style="font-size:28px;color:#1565C0;vertical-align:middle">{icon}</span>
            <span style="font-size:1.8rem;font-weight:700;color:#1A1A2E;letter-spacing:-0.02em">{title}</span>
        </div>
        {cap_html}
        """,
        unsafe_allow_html=True,
    )


def section_header(icon: str, title: str) -> None:
    """Small section header with Material Icon."""
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:8px;margin-top:1.5rem;margin-bottom:0.5rem">
            <span class="material-icons-round"
                  style="font-size:20px;color:#1565C0">{icon}</span>
            <span style="font-size:1.1rem;font-weight:600;color:#1A1A2E">{title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_card(label: str, value: str, icon: str = "") -> str:
    """Return HTML for a single stat card (for use inside a grid div)."""
    icon_html = (
        f'<span class="material-icons-round" '
        f'style="font-size:16px;color:#1565C0;vertical-align:middle;margin-right:4px">'
        f'{icon}</span>'
        if icon else ""
    )
    return (
        f'<div style="{_CARD_STYLE}">'
        f'  <div style="{_CARD_LABEL}">{icon_html}{label}</div>'
        f'  <div style="{_CARD_VALUE}">{value}</div>'
        f'</div>'
    )


def kpi_grid(items: list[dict], columns: int = 5) -> None:
    """
    Render a responsive grid of stat cards.
    Each item: {label, value, icon (optional)}.
    """
    cards_html = "".join(stat_card(i["label"], i["value"], i.get("icon","")) for i in items)
    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat({columns},1fr);'
        f'gap:12px;margin-bottom:1.5rem">{cards_html}</div>',
        unsafe_allow_html=True,
    )
