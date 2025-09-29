# app.py — AIDMT (updated for new Top-10 + dynamic schema/order)
import streamlit as st
import json
from pathlib import Path
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu

# Prefer skops (portable across numpy versions); fall back to joblib
try:
    from skops.io import load as skops_load
    HAS_SKOPS = True
except Exception:
    HAS_SKOPS = False
import joblib  # fallback

# ============== Streamlit base config ==============
st.set_page_config(
    page_title="AIDMT — Pre-/Diabetes Risk",
    page_icon="🩺",
    layout="wide"
)

# ============== Language switcher ==============
if "lang" not in st.session_state:
    st.session_state.lang = "en"
_, col_lang = st.columns([6, 1])
with col_lang:
    choice = st.selectbox(
        "Language / Sprache",
        ["🇬🇧 English", "🇩🇪 Deutsch"],
        index=0 if st.session_state.lang == "en" else 1,
        label_visibility="collapsed",
        key="lang_select"
    )
    st.session_state.lang = "en" if "English" in choice else "de"
LANG = st.session_state.lang

# ============== Translations ==============
T = {
    "title": {
        "en": "AI-based Diabetes Mellitus Prediction Tool for Trauma Clinics",
        "de": "KI-gestütztes Tool zur Diabetesrisikovorhersage für Unfallkliniken",
    },
    "patient_values": {"en": "Patient values", "de": "Patientenwerte"},
    "get_estimate": {"en": "Get risk estimate", "de": "Risikowert berechnen"},
    "result": {"en": "Result", "de": "Ergebnis"},
    "est_prob": {"en": "Estimated probability:", "de": "Geschätzte Wahrscheinlichkeit:"},
    "low": {"en": "Low risk", "de": "Niedriges Risiko"},
    "med": {"en": "Moderate risk", "de": "Mittleres Risiko"},
    "high": {"en": "High risk", "de": "Hohes Risiko"},
    "batch": {"en": "📄 Batch scoring (optional)", "de": "📄 Stapelauswertung (optional)"},
    "upload_csv": {"en": "Upload CSV", "de": "CSV hochladen"},
    "download": {"en": "Download results", "de": "Ergebnisse herunterladen"},
    "about_algo": {"en": "About the Algorithm", "de": "Über den Algorithmus"},
    "about_aidmt": {"en": "About AIDMT", "de": "Über AIDMT"},
    "contact_us": {"en": "Contact Us", "de": "Kontakt"},
    "contact_text": {
        "en": "For questions or feedback: <a href='mailto:mlktombaz@gmail.com'>mlktombaz@gmail.com</a>",
        "de": "Bei Fragen oder Feedback: <a href='mailto:mlktombaz@gmail.com'>mlktombaz@gmail.com</a>",
    },
    "yes": {"en": "Yes", "de": "Ja"},
    "no": {"en": "No", "de": "Nein"},
    "non_smoker": {"en": "Non-smoker", "de": "Nichtraucher:in"},
    "ex_smoker": {"en": "Ex-smoker", "de": "Ehemalige:r Raucher:in"},
    "curr_smoker": {"en": "Current smoker", "de": "Aktuelle:r Raucher:in"},
    "years_smoking_0": {"en": "Years smoking: 0", "de": "Rauchjahre: 0"},
    "host_notice": {
        "en": ("This application is provided as a free, research-oriented tool. "
               "Due to hosting limitations, it may occasionally experience slowdowns or unexpected crashes. "
               "We appreciate your understanding."),
        "de": ("Diese frei gehostete, forschungsorientierte App kann aufgrund von Hosting-Beschränkungen "
               "gelegentlich langsam reagieren oder abstürzen. Vielen Dank für Ihr Verständnis."),
    },
    "about_text": {
        "en": (
            "AIDMT is intended for use in <strong>trauma clinics</strong> as a decision-support tool "
            "to estimate the risk of <strong>prediabetes and diabetes</strong> based on routinely collected data. "
            "It does not provide a definitive diagnosis, and all results should be interpreted by qualified clinicians."
        ),
        "de": (
            "AIDMT ist für den Einsatz in <strong>Traumakliniken</strong> als Entscheidungsunterstützung gedacht, "
            "um das Risiko für <strong>Prädiabetes und Diabetes</strong> auf Grundlage routinemäßig erhobener Daten zu schätzen. "
            "Es stellt keine definitive Diagnose dar, und alle Ergebnisse sollten von qualifizierten Kliniker:innen interpretiert werden."
        ),
    },
    "example_csv": {"en": "Example CSV Format", "de": "Beispiel-CSV-Format"},
    "csv_caption": {
        "en": "Values should follow the same units as indicated in the input form.",
        "de": "Die Werte sollten denselben Einheiten entsprechen wie im Eingabeformular angegeben.",
    },
}

def t(key: str) -> str:
    return T.get(key, {}).get(LANG, key)

# ============== Global CSS ==============
st.markdown("""
<style>
:root { color-scheme: light; }
header[data-testid="stHeader"]{ display:none; }
html, body, .stApp { background:#004994 !important; }

/* Page container (card) */
.block-container{
  max-width:1280px; width:min(1280px,96vw);
  margin:.25rem auto 2rem;
  padding:.75rem 2rem;
  background:#fff; border-radius:18px;
  box-shadow:0 10px 28px rgba(0,0,0,.18);
}
.block-container > :first-child{ margin-top:0!important; padding-top:0!important; }

/* Typography */
@import url('https://fonts.googleapis.com/css2?family=Roboto+Slab:wght@300;400;700&display=swap');
html, body, [class*="css"]{ font-family:'Roboto Slab',serif !important; }
h1,h2,h3,h4,h5,h6,p,span,div,label{ color:#000 !important; }
h1,h2{ letter-spacing:.2px; } a{ color:#004994 !important; }

/* Inputs */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div{
  background:#f6f8fb !important; color:#000 !important;
  border:1px solid #cfd7e3 !important; border-radius:12px !important;
}
.stNumberInput > div > div > input{ padding:.7rem .9rem !important; }
.stTextInput input::placeholder, .stTextArea textarea::placeholder{ color:#6b7280 !important; }
.stTextInput input:focus, .stNumberInput input:focus,
.stTextArea textarea:focus, .stSelectbox div[data-baseweb="select"] > div:focus{
  outline:none !important; border-color:#004994 !important;
  box-shadow:0 0 0 3px rgba(0,73,148,.15) !important;
}

/* Number steppers */
[data-testid="stNumberInput"] button{
  background:#0f1e2e !important; color:#fff !important;
  border:none !important; width:37px; height:37px; border-radius:10px !important;
}
[data-testid="stNumberInput"] button:hover{ filter:brightness(1.1); }

/* Radios / checkboxes / slider */
input[type="radio"], input[type="checkbox"]{ accent-color:#004994 !important; }
[data-baseweb="slider"] > div{ background:#e6ebf4 !important; }
[data-baseweb="slider"] div[role="slider"]{
  background:#004994 !important; box-shadow:0 0 0 3px rgba(0,73,148,.15) !important;
}

/* Buttons */
.stButton > button, .stButton > button *{
  display:flex !important; align-items:center !important; justify-content:center !important;
  height:3.8rem; padding:0 2rem; font-size:1.2rem; font-weight:700;
  background:#56184a !important; color:#fff !important; border:none !important;
  border-radius:12px !important; box-shadow:0 6px 14px rgba(86,24,74,.25);
}
.stButton > button:hover{ filter:brightness(1.06); box-shadow:0 8px 18px rgba(86,24,74,.32); }
.stButton > button:disabled{ background:#b9b9c2 !important; color:#f2f2f2 !important; box-shadow:none; }
div[data-testid="stButton"] button[purpose="primary"]{ margin:0 auto; display:block; }

/* Cards & badges */
.card{ background:#fff; border:1px solid #eef0f4; border-radius:16px; padding:18px; }
.card h3{ margin:0 0 .8rem 0; }
.badge{ display:inline-block; padding:.25rem .6rem; border-radius:999px; font-size:.85rem; }
.badge-low{ background:#e8f5e9; color:#256029; }
.badge-med{ background:#fff8e1; color:#7a5c00; }
.badge-high{ background:#ffebee; color:#b71c1c; }
.badge-result{ font-size:1.4rem !important; padding:.6rem 1.2rem !important; }
.notice{ font-size:.85rem; color:#5b5e6a; font-style:italic; margin-top:.8rem; }

/* Infobox */
.infobox{
  display:flex; gap:.75rem; align-items:flex-start;
  background:#f4f8ff; border:1px solid #cfd7e3; border-left:6px solid #004994;
  border-radius:12px; padding:12px 14px; color:#000 !important;
}
.infobox .icon{
  width:28px; height:28px; border-radius:999px; background:#004994; color:#fff;
  display:flex; align-items:center; justify-content:center; font-weight:800; font-size:18px; flex:0 0 28px;
}
.infobox h4{ margin:.1rem 0 .25rem 0; font-size:1rem; font-weight:700; }
.infobox p{ margin:0; font-size:.95rem; }

/* Header band */
.header-box{ background:#004994; border-radius:14px; padding:1rem 1.5rem; margin:.25rem 0 .5rem; }
.header-text h1{ color:#fff; font-size:2rem; font-weight:800; line-height:1.25; margin:0; }
@media (max-width:768px){ .header-text h1{ font-size:1.5rem; } }

/* Radios as tabs look */
.tab-radio [role="radiogroup"]{ display:flex; gap:.5rem; background:#004994; padding:.35rem; border-radius:12px; }
.tab-radio [role="radio"]{ cursor:pointer; background:transparent; border:1px solid rgba(255,255,255,.18); border-radius:10px; padding:.45rem .95rem; }
.tab-radio [role="radio"], .tab-radio [role="radio"] *{ color:#000 !important; -webkit-text-fill-color:#000 !important; font-weight:700 !important; }
.tab-radio [role="radio"]:hover{ background:#003b7a !important; }
.tab-radio [role="radio"][aria-checked="true"]{ background:#006226 !important; border-color:#006226 !important; }
.tab-radio [role="radio"][aria-checked="true"], .tab-radio [role="radio"][aria-checked="true"] *{ color:#000 !important; -webkit-text-fill-color:#000 !important; }

/* Remove stray lines */
hr, .stMarkdown hr, div[role="separator"], .stDivider{ display:none !important; }
.streamlit-expanderHeader{ border:none !important; }

/* Emoji font for language select */
.lang-picker [data-baseweb="select"] *,
[data-baseweb="popover"] [role="listbox"] *,
[data-baseweb="popover"] [role="option"] *{
  font-family:'Apple Color Emoji','Segoe UI Emoji','Noto Color Emoji','Noto Emoji', system-ui, sans-serif !important;
}

/* Footer */
.footer{ color:#fff !important; font-size:.9rem; margin-top:2rem; text-align:center; }
.footer a{ color:#fff; text-decoration:underline; margin:0 10px; }
.footer a:hover{ color:#dbeafe; }
</style>
""", unsafe_allow_html=True)

# ============== Branded header ==============
st.markdown("""
    <style>
    .header-box .stColumn > div {
        display: flex;
        align-items: center;
    }
    .header-text h1 {
        margin: 0;
        line-height: 3.5;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-box">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 5])
with col1:
    # Replace with your logo path if needed
    try:
        st.image("background.png", width=175)
    except Exception:
        st.empty()
with col2:
    st.markdown(f"""
        <div class="header-text">
        <h1>{t('title')}</h1>
        </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ============== About box ==============
st.markdown(
    f"""
    <div class="infobox">
      <div class="icon">i</div>
      <div>
        <h4>{t('about_aidmt')}</h4>
        <p>{t('about_text')}</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============== Session defaults ==============
if "mdl_dir_in" not in st.session_state:
    st.session_state.mdl_dir_in = "final_model_top10"
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.50

# ============== Model loading helpers ==============
BASE_DIR = Path(__file__).resolve().parent

def _has_model_files(p: Path) -> bool:
    return (p / "feature_schema.json").exists() and (
        (p / "model.skops").exists() or (p / "diabetes_risk_pipeline.pkl").exists()
    )

def _resolve_model_dir(user_text: str | None) -> Path:
    candidates = []
    if user_text:
        p = Path(user_text)
        candidates.append(p if p.is_absolute() else BASE_DIR / p)
    candidates += [BASE_DIR / "final_model_top10", BASE_DIR / "final_model"]
    for c in candidates:
        if _has_model_files(c):
            return c
    for f in BASE_DIR.rglob("feature_schema.json"):
        d = f.parent
        if _has_model_files(d):
            return d
    raise FileNotFoundError("Model files not found. Open Info ▸ Admin and set the correct folder.")

@st.cache_resource
def load_artifacts(model_dir_text: str | None):
    p = _resolve_model_dir(model_dir_text)
    skops_path = p / "model.skops"
    pkl_path   = p / "diabetes_risk_pipeline.pkl"

    if skops_path.exists() and HAS_SKOPS:
        pipe = skops_load(skops_path, trusted=True)
    elif pkl_path.exists():
        pipe = joblib.load(pkl_path)
    elif skops_path.exists() and not HAS_SKOPS:
        raise RuntimeError("Found model.skops but skops is not installed. Try: pip install skops")
    else:
        raise FileNotFoundError("No model file found (model.skops or diabetes_risk_pipeline.pkl).")

    with open(p / "feature_schema.json", "r", encoding="utf-8") as f:
        schema = json.load(f)
    features = schema.get("features", [])
    return pipe, features, str(p)

# Load model + schema
try:
    pipe, FEATURES, RESOLVED_DIR = load_artifacts(st.session_state.mdl_dir_in)
except Exception as e:
    st.error(str(e))
    st.stop()

# ============== Optional: UI order from selected_features_top10.json ==============
def _load_selected_features_order(resolved_dir: str) -> list[str] | None:
    candidates = [
        Path(resolved_dir) / "selected_features_top10.json",
        BASE_DIR / "selected_features_top10.json",
        Path("/mnt/data/selected_features_top10.json"),  # local dev fallback
    ]
    for c in candidates:
        try:
            if c.exists():
                with open(c, "r", encoding="utf-8") as f:
                    sel = json.load(f)
                sel_list = sel.get("features", [])
                # keep only ones that exist in the active model schema
                return [f for f in sel_list if f in set(FEATURES)]
        except Exception:
            pass
    return None

FEATURES_ORDER = _load_selected_features_order(RESOLVED_DIR)

# ============== Friendly labels, aliases, units, helpers ==============
YESNO_2_0_FEATURE = "Previous High Blood Sugar Levels"
SMOKING_YEARS_FEATURE = "Smoking for how long"

LABELS = {
    "Age": "Age",
    YESNO_2_0_FEATURE: "Previous high blood sugar levels",
    "Leukocytes": "Leukocytes",
    "MCHC": "MCHC",
    "Waist Circumference": "Waist circumference",
    "APTT": "APTT",
    "QUICK": "QUICK",
    "Potassium": "Potassium",
    "MCH": "MCH",
    SMOKING_YEARS_FEATURE: "Smoking history",
    "Sodium": "Sodium",
    "CREA": "Creatinine",
    "Creatinine": "Creatinine",
}

# Friendly -> schema key
ALIASES = {
    "Age": "Age",
    YESNO_2_0_FEATURE: YESNO_2_0_FEATURE,
    SMOKING_YEARS_FEATURE: SMOKING_YEARS_FEATURE,
    "Leukocytes": "LEU",
    "Waist Circumference": "WaistCircumference",
    "APTT": "APTT",
    "QUICK": "QUICK",
    "Potassium": "Potassium",
    "MCHC": "MCHC",
    "MCH": "MCH",
    "Sodium": "Sodium",
    "CREA": "CREA",
    "Creatinine": "CREA",  # allow "Creatinine" header to map to CREA
}

UNITS = {
    "Age": "years",
    "Leukocytes": "10^3/µL",
    "MCHC": "g/dL",
    "Waist Circumference": "cm",
    "APTT": "s",
    "QUICK": "%",
    "Potassium": "mmol/L",
    "MCH": "pg",
    YESNO_2_0_FEATURE: None,
    SMOKING_YEARS_FEATURE: "years",
    "Sodium": "mmol/L",
    "CREA": "mg/dL",  # adjust if your model used µmol/L
}

REF_RANGES = {
    "Leukocytes": "≈ 4.3–10.0",
    "MCHC": "≈ 33–36",
    "Waist Circumference": "adult ≈ 60–160",
    "APTT": "≈ 21–29",
    "QUICK": "≈ 70–120",
    "Potassium": "≈ 3.5–5.1",
    "MCH": "≈ 28–33",
    "Sodium": "≈ 135–145",
    "CREA": "≈ 0.7–1.2",
}

LABELS_DE = {
    "Age": "Alter",
    "Leukocytes": "Leukozyten",
    "MCHC": "MCHC",
    "Waist Circumference": "Taillenumfang",
    "APTT": "APTT",
    "QUICK": "QUICK",
    "Potassium": "Kalium",
    "MCH": "MCH",
    YESNO_2_0_FEATURE: "Frühere hohe Blutzuckerwerte",
    SMOKING_YEARS_FEATURE: "Raucheranamnese",
    "Sodium": "Natrium",
    "CREA": "Kreatinin",
    "Creatinine": "Kreatinin",
}

UNITS_DE = {
    "Age": "Jahre",
    "Leukocytes": "10^3/µL",
    "MCHC": "g/dL",
    "Waist Circumference": "cm",
    "APTT": "s",
    "QUICK": "%",
    "Potassium": "mmol/L",
    "MCH": "pg",
    YESNO_2_0_FEATURE: None,
    SMOKING_YEARS_FEATURE: "Jahre",
    "Sodium": "mmol/L",
    "CREA": "mg/dL",
}

RANGE_PREFIX = {"en": "Typical range:", "de": "Typischer Bereich:"}

def has_feat(friendly_name: str) -> bool:
    if "FEATURES" not in globals() or FEATURES is None:
        return True
    try:
        keys = set(FEATURES)
    except Exception:
        return True
    key = ALIASES.get(friendly_name, friendly_name)
    return key in keys

def key_for(friendly_name: str) -> str:
    return ALIASES.get(friendly_name, friendly_name)

def display_label(key: str) -> str:
    if LANG == "de":
        return LABELS_DE.get(key, LABELS.get(key, key))
    return LABELS.get(key, key)

def display_unit(key: str) -> str | None:
    if LANG == "de":
        return UNITS_DE.get(key, UNITS.get(key))
    return UNITS.get(key)

def label_with_unit(key: str) -> str:
    lab = display_label(key)
    unit = display_unit(key)
    return f"{lab} ({unit})" if unit else lab

def help_with_range(key: str) -> str | None:
    rng = REF_RANGES.get(key)
    if not rng:
        return None
    return f"{RANGE_PREFIX[LANG]} {rng} {display_unit(key) or ''}".strip()

def _coerce_row(inputs: dict, features: list[str]) -> pd.DataFrame:
    row = {}
    for feat in features:
        v = inputs.get(feat, None)
        if v is None or (isinstance(v, str) and v.strip() == ""):
            row[feat] = np.nan
            continue
        try:
            fv = float(v)
            row[feat] = int(fv) if fv.is_integer() else fv
        except Exception:
            row[feat] = np.nan
    return pd.DataFrame([row]).reindex(columns=features)

# ============== NAV (Home / Info) ==============
with st.container():
    st.markdown('<div class="top-nav">', unsafe_allow_html=True)
    selected = option_menu(
        None,
        ["Home", "Info"],
        icons=["house", "info-circle"],
        orientation="horizontal",
        default_index=0,
        styles={
            "container": {"background-color": "#d3d3d3", "padding": "0"},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {"color": "white", "font-size": "16px", "margin": "0px"},
            "nav-link-selected": {"background-color": "green"},
        },
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ============== Pages ==============
if selected == "Home":
    # ------- Input form card -------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"<h3>{t('patient_values')}</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    inputs: dict[str, object] = {}

    # Helper to place a numeric field
    def num_input(friendly_key: str, min_value: float = 0.0, value: float = 0.0, fmt: str = "%.2f"):
        if has_feat(friendly_key):
            inputs[key_for(friendly_key)] = st.number_input(
                label_with_unit(friendly_key), min_value=min_value, value=value, format=fmt,
                help=help_with_range(friendly_key)
            )

    with col1:
        # Age
        if has_feat("Age"):
            inputs[key_for("Age")] = st.number_input(
                label_with_unit("Age"), min_value=0, max_value=120, value=45, step=1
            )

        # Previous high blood sugar (Yes/No -> 2/0)
        if has_feat(YESNO_2_0_FEATURE):
            yn = st.radio(
                label_with_unit(YESNO_2_0_FEATURE),
                [t("no"), t("yes")],
                horizontal=True, index=0
            )
            inputs[key_for(YESNO_2_0_FEATURE)] = 2 if yn == t("yes") else 0

        # Smoking history (if model actually needs it)
        if has_feat(SMOKING_YEARS_FEATURE):
            status = st.radio(
                label_with_unit(SMOKING_YEARS_FEATURE),
                [t("non_smoker"), t("ex_smoker"), t("curr_smoker")],
                index=0
            )
            if status == t("non_smoker"):
                st.caption(t("years_smoking_0"))
                inputs[key_for(SMOKING_YEARS_FEATURE)] = 0
            else:
                years_label = "Years smoking (years)" if LANG == "en" else "Rauchjahre (Jahre)"
                years = st.number_input(years_label, min_value=0, max_value=80, value=5, step=1)
                inputs[key_for(SMOKING_YEARS_FEATURE)] = years

        # Leukocytes, Waist Circumference
        num_input("Leukocytes", value=0.0, fmt="%.2f")
        num_input("Waist Circumference", value=0.0, fmt="%.1f")

    with col2:
        # QUICK, APTT, Potassium, MCHC, MCH
        num_input("QUICK", value=0.0, fmt="%.2f")
        num_input("APTT", value=0.0, fmt="%.2f")
        num_input("Potassium", value=0.0, fmt="%.2f")
        num_input("MCHC", value=0.0, fmt="%.2f")
        num_input("MCH", value=0.0, fmt="%.2f")

        # NEW: Sodium & CREA
        num_input("Sodium", value=0.0, fmt="%.1f")
        num_input("CREA", value=0.0, fmt="%.2f")

    # Button row
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        submit = st.button(t("get_estimate"), type="primary")

    st.markdown('</div>', unsafe_allow_html=True)  # close card

    # ------- Predict & display -------
    threshold = float(max(0.0, min(1.0, st.session_state.threshold)))
    if submit:
        x1 = _coerce_row(inputs, FEATURES)
        try:
            proba = float(pipe.predict_proba(x1)[0, 1])
        except Exception:
            st.error("Unable to calculate risk. Please review inputs." if LANG == "en"
                     else "Risiko konnte nicht berechnet werden. Bitte Eingaben prüfen.")
            st.stop()

        label, cls = (t("low"), "badge-low")
        if proba >= threshold:
            label, cls = (t("high"), "badge-high")
        elif proba >= threshold * 0.75:
            label, cls = (t("med"), "badge-med")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"<h3>{t('result')}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:1.4rem;margin:.2rem 0;'>{t('est_prob')}</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='badge {cls} badge-result'><b>{proba*100:.1f}%</b> &nbsp;•&nbsp; {label}</div>", unsafe_allow_html=True)
        st.markdown(f"<p class='notice'>{t('host_notice')}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ------- Batch CSV -------
    with st.expander(t("batch")):
        st.write(
            "Upload a CSV with the same column names as the model schema. "
            "Friendly names (e.g., 'Leukocytes', 'Waist Circumference') are accepted and will be mapped."
            if LANG == "en" else
            "CSV mit denselben Spaltennamen wie im Modellschema hochladen. "
            "Freundliche Namen (z. B. „Leukozyten“, „Taillenumfang“) werden akzeptiert und zugeordnet."
        )

        csv = st.file_uploader(t("upload_csv"), type=["csv"])
        if csv is not None:
            try:
                df = pd.read_csv(csv)
                # map friendly -> schema
                friendly_to_schema = {**{k: k for k in FEATURES}, **ALIASES}
                df.columns = [friendly_to_schema.get(c, c) for c in df.columns]
                df = df.reindex(columns=FEATURES)
                for c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

                # Yes/No mapping for previous high blood sugar, if present
                if YESNO_2_0_FEATURE in df.columns:
                    df[YESNO_2_0_FEATURE] = df[YESNO_2_0_FEATURE].replace(
                        {"Yes": 2, "No": 0, "Ja": 2, "Nein": 0}
                    )
                    df[YESNO_2_0_FEATURE] = pd.to_numeric(df[YESNO_2_0_FEATURE], errors="coerce")

                # Smoking friendly to numeric years (if someone puts "Non-smoker")
                if SMOKING_YEARS_FEATURE in df.columns:
                    df[SMOKING_YEARS_FEATURE] = df[SMOKING_YEARS_FEATURE].replace(
                        {"Non-smoker": 0, "Nichtraucher:in": 0}
                    )
                    df[SMOKING_YEARS_FEATURE] = pd.to_numeric(df[SMOKING_YEARS_FEATURE], errors="coerce")

                preds = pipe.predict_proba(df)[:, 1]
                out = df.copy()
                out["risk_proba"] = preds

                st.write(out.head())
                st.download_button(
                    t("download"),
                    out.to_csv(index=False).encode("utf-8"),
                    "aidmt_predictions.csv",
                    "text/csv"
                )
            except Exception:
                st.error(
                    "Could not score the file. Please check the columns and values."
                    if LANG == "en" else
                    "Die Datei konnte nicht ausgewertet werden. Bitte überprüfen Sie Spalten und Werte."
                )

elif selected == "Info":
    st.markdown(f"<h3>{t('about_algo')}</h3>", unsafe_allow_html=True)
    st.write({
        "en": (
            "This risk prediction tool (AIDMT) was developed using anonymized trauma clinic data. "
            "Multiple machine learning algorithms (e.g., Random Forest, Gradient Boosting, XGBoost, MLP) "
            "were trained and evaluated with nested cross-validation. "
            "The final model uses a reduced set of the most predictive features, selected by SHAP importance analysis.\n\n"
            "**Disclaimer:** The tool is a decision-support system only and **does not** replace a clinical diagnosis."
        ),
        "de": (
            "Dieses Risikovorhersagetool (AIDMT) wurde mit anonymisierten Traumaklinikdaten entwickelt. "
            "Mehrere maschinelle Lernalgorithmen (z. B. Random Forest, Gradient Boosting, XGBoost, MLP) "
            "wurden mittels verschachtelter Kreuzvalidierung trainiert und evaluiert. "
            "Das Endmodell verwendet eine reduzierte Menge der prädiktivsten Merkmale, ausgewählt durch SHAP-Analyse.\n\n"
            "**Haftungsausschluss:** Das Tool dient ausschließlich als Entscheidungsunterstützungssystem und ersetzt **keine** klinische Diagnose."
        ),
    }[LANG])

    # ----- Dynamic Example CSV (always mirrors active model) -----
    st.markdown(f"<h3>{t('example_csv')}</h3>", unsafe_allow_html=True)

    # Inverse alias map: schema->friendly (best-effort)
    def _friendly_name(schema_key: str) -> str:
        # prefer a known friendly name if present
        for friendly, skey in ALIASES.items():
            if skey == schema_key:
                return friendly
        return schema_key

    def _example_value(schema_key: str):
        EX = {
            "Age": 45,
            "LEU": 6.2, "Leukocytes": 6.2,
            "MCHC": 34, "MCH": 29,
            "WaistCircumference": 95,
            "APTT": 27.5, "QUICK": 105,
            "Potassium": 4.2, "Sodium": 140,
            "CREA": 0.9,
            YESNO_2_0_FEATURE: 0,
            SMOKING_YEARS_FEATURE: 0,
        }
        return EX.get(schema_key, 0)

    order = FEATURES_ORDER if FEATURES_ORDER else FEATURES
    ex_cols = [_friendly_name(k) for k in order]
    ex_row1 = {_friendly_name(k): _example_value(k) for k in order}
    ex_row2 = ex_row1.copy()
    # tweak a few values
    if "Age" in order:
        ex_row2[_friendly_name("Age")] = 62
    if YESNO_2_0_FEATURE in order:
        ex_row2[_friendly_name(YESNO_2_0_FEATURE)] = 2
    if "WaistCircumference" in order:
        ex_row2[_friendly_name("WaistCircumference")] = 110
    example = pd.DataFrame([ex_row1, ex_row2], columns=ex_cols)

    st.dataframe(example)
    st.caption(t("csv_caption"))

    # ----- Admin -----
    with st.expander("⚙️ Admin", expanded=False):
        st.text_input(
            "Settings ▸ Model folder",
            key="mdl_dir_in",
            help="Folder containing model.skops (or diabetes_risk_pipeline.pkl) and feature_schema.json"
        )
        st.slider("Decision threshold", 0.0, 1.0, key="threshold", step=0.01)
        st.caption(f"Active model folder: {RESOLVED_DIR}")

# ============== Footer ==============
st.markdown(
    """
    <div class="footer">
        <p>
        🔗 <a href="https://www.bg-kliniken.de/klinik-tuebingen/ueber-uns/unsere-einrichtungen/siegfried-weller-institut/" target="_blank">Siegfried Weller Institut</a> 
        | <a href="https://www.linkedin.com/company/siegfried-weller-institute-for-trauma-research/posts/?feedView=all" target="_blank">LinkedIn</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
