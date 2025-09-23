# AIDMT — Pre-/Diabetes Risk Prediction • Header-band tabs

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu  # horizontal header nav

# Prefer skops (portable across numpy versions); fall back to joblib
try:
    from skops.io import load as skops_load
    HAS_SKOPS = True
except Exception:
    HAS_SKOPS = False
import joblib  # fallback


# ------------------ Page setup ------------------
st.set_page_config(page_title="AIDMT — Pre-/Diabetes Risk", page_icon="🩺", layout="wide")

# Hide Streamlit system header (keeps only your branded header)
st.markdown(
    """
    <style>
      /* ===== Force consistent (light) app look ===== */
      :root { color-scheme: light; }
      header[data-testid="stHeader"] { display:none; }

      /* ===== Page background ===== */
      html, body, .stApp { background-color:#004994 !important; }

      /* ===== Central content card ===== */
      .block-container{
        max-width:940px;
        margin:2rem auto;
        padding:2rem 2.2rem;
        background:#ffffff;
        border-radius:18px;
        box-shadow:0 10px 28px rgba(0,0,0,.18);
      }

      /* ===== Typography ===== */
      h1,h2,h3,h4,h5,h6,p,span,div,label{ color:#000 !important; }
      h1,h2 { letter-spacing:.2px; }
      a { color:#004994 !important; }
      hr { border:0; height:1px; background:#eef0f4; }

      /* ===== Inputs: text/number/textarea/select ===== */
      .stTextInput input,
      .stNumberInput input,
      .stTextArea textarea,
      .stSelectbox div[data-baseweb="select"] > div {
        background:#f6f8fb !important;
        color:#000 !important;
        border:1px solid #cfd7e3 !important;
        border-radius:12px !important;
      }
      .stNumberInput > div > div > input {
        padding:.7rem .9rem !important;
      }
      .stTextInput input::placeholder,
      .stTextArea textarea::placeholder { color:#6b7280 !important; }

      /* Number steppers (the +/- buttons) */
      [data-testid="stNumberInput"] button {
        background:#0f1e2e !important;       /* dark so visible on light cards */
        color:#fff !important;
        border:none !important;
        box-shadow:none !important;
        width:28px; height:28px; border-radius:8px !important;
      }
      [data-testid="stNumberInput"] button:hover { filter:brightness(1.1); }

      /* Focus ring */
      .stTextInput input:focus,
      .stNumberInput input:focus,
      .stTextArea textarea:focus,
      .stSelectbox div[data-baseweb="select"] > div:focus {
        outline:none !important;
        border-color:#004994 !important;
        box-shadow:0 0 0 3px rgba(0,73,148,.15) !important;
      }

      /* Radios / checkboxes */
      input[type="radio"], input[type="checkbox"] { accent-color:#004994 !important; }
      .stRadio > label, .stCheckbox > label { color:#000 !important; }

      /* Slider */
      div[data-baseweb="slider"] > div {     /* rail */
        background:#e6ebf4 !important;
      }
      div[data-baseweb="slider"] div[role="slider"] { /* handle */
        background:#004994 !important;
        box-shadow:0 0 0 3px rgba(0,73,148,.15) !important;
      }
      div[data-baseweb="slider"] div[role="slider"]:hover { filter:brightness(1.06); }

      /* Buttons */
      .stButton > button, .stButton > button * {
        display:flex !important; align-items:center !important; justify-content:center !important;
        height:3.8rem; padding:0 2rem; font-size:1.2rem; font-weight:700;
        background:#56184a !important; color:#fff !important; border:none !important;
        border-radius:12px !important; box-shadow:0 6px 14px rgba(86,24,74,.25);
      }
      .stButton > button:hover { filter:brightness(1.06); box-shadow:0 8px 18px rgba(86,24,74,.32); }
      .stButton > button:disabled { background:#b9b9c2 !important; color:#f2f2f2 !important; box-shadow:none; }

      /* ===== Cards ===== */
      .card{ background:#ffffff; border:1px solid #eef0f4; border-radius:16px; padding:18px; }
      .card h3{ margin:0 0 .8rem 0; }

      /* ===== Badges / results ===== */
      .badge{ display:inline-block; padding:.25rem .6rem; border-radius:999px; font-size:.85rem; }
      .badge-low{ background:#e8f5e9; color:#256029; }
      .badge-med{ background:#fff8e1; color:#7a5c00; }
      .badge-high{ background:#ffebee; color:#b71c1c; }
      .risk-high{ color:#681c16 !important; }

      /* ===== Infobox ===== */
      .infobox{
        display:flex; gap:.75rem; align-items:flex-start;
        background:#f4f8ff; border:1px solid #cfd7e3; border-left:6px solid #004994;
        border-radius:12px; padding:12px 14px; margin:0 0 1rem 0; color:#000 !important;
      }
      .infobox .icon{
        width:28px; height:28px; border-radius:999px; background:#004994; color:#fff;
        display:flex; align-items:center; justify-content:center; font-weight:800; font-size:16px; flex:0 0 28px;
      }
      .infobox h4{ margin:.1rem 0 .25rem 0; font-size:1rem; font-weight:700; }
      .infobox p{ margin:0; font-size:.95rem; }

      /* ===== Branded header ===== */
      .header-box{
        background-color:#004994;
        border-radius:14px; padding:1rem 1.5rem; margin-bottom:.6rem;
      }
      .header-text h1{
        color:#ffffff; font-size:2rem; font-weight:800; line-height:1.25; margin:0;
      }
      @media (max-width:768px){ .header-text h1{ font-size:1.5rem; } }

      /* ===== Inline hint & footer ===== */
      .inline-hint{ margin:.35rem 0 .1rem 0; font-size:.9rem; color:#5b5e6a !important; }
      .footer{ color:#78808b !important; font-size:.85rem; margin-top:1rem; }

      /* ===== Header NAV band (option_menu) ===== */
      .top-nav { background:#004994; border-radius:12px; padding:.35rem .5rem; margin:.4rem 0 1rem 0; }
      .top-nav .nav-link { color:#ffffff !important; font-weight:700; border-radius:10px; }
      .top-nav .nav-link:hover { background:#003b7a !important; }
      .top-nav .nav-link.active { background:#006226 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
<style>
/* ================================
   CROSS-BROWSER RADIO-AS-TABS FIX
   ================================ */
.tab-radio [role="radiogroup"]{
  display:flex; gap:.5rem;
  background:#004994;               /* blue band */
  padding:.35rem; border-radius:12px;
}

/* Pill container */
.tab-radio [role="radio"]{
  position:relative; cursor:pointer;
  background:transparent;
  border:1px solid rgba(255,255,255,.18);
  border-radius:10px; padding:.45rem .95rem;
  -webkit-tap-highlight-color: transparent;
}

/* Inactive pill text – FORCE on all descendants (Chrome/Safari/Firefox) */
.tab-radio [role="radio"], 
.tab-radio [role="radio"] *, 
.tab-radio [role="radio"] [data-testid="stMarkdownContainer"] p,
.tab-radio [role="radio"] [data-testid="stMarkdownContainer"] span {
  color:#ffffff !important;
  -webkit-text-fill-color:#ffffff !important; /* Safari */
}

/* Hover */
.tab-radio [role="radio"]:hover{ background:#003b7a !important; }

/* Active pill */
.tab-radio [role="radio"][aria-checked="true"]{
  background:#006226 !important; border-color:#006226 !important;
}

/* Active pill text – FORCE again */
.tab-radio [role="radio"][aria-checked="true"],
.tab-radio [role="radio"][aria-checked="true"] *,
.tab-radio [role="radio"][aria-checked="true"] [data-testid="stMarkdownContainer"] p,
.tab-radio [role="radio"][aria-checked="true"] [data-testid="stMarkdownContainer"] span {
  color:#ffffff !important;
  -webkit-text-fill-color:#ffffff !important;
}

/* Keyboard focus */
.tab-radio [role="radio"]:focus-visible{
  outline:3px solid rgba(0,73,148,.35); outline-offset:2px;
}

/* ---------- If you also use st.tabs anywhere ---------- */
.stTabs [data-baseweb="tab-list"]{
  gap:.5rem; background:#004994; padding:.35rem; border-radius:12px;
}
.stTabs [data-baseweb="tab"],
.stTabs [data-baseweb="tab"] *,
.stTabs [data-baseweb="tab"] [data-testid="stMarkdownContainer"] p {
  color:#ffffff !important; -webkit-text-fill-color:#ffffff !important;
  background:transparent; border-radius:10px; padding:.45rem .95rem;
  border:1px solid rgba(255,255,255,.18);
}
.stTabs [data-baseweb="tab"]:hover{ background:#003b7a !important; }
.stTabs [aria-selected="true"],
.stTabs [aria-selected="true"] *{
  background:#006226 !important; color:#ffffff !important; -webkit-text-fill-color:#ffffff !important;
  border-color:#006226 !important;
}
.stTabs [data-baseweb="tab-highlight"]{ background:transparent !important; }

/* ================================
   WIDER LAYOUT (fills more width)
   ================================ */
.block-container{
  max-width:1280px;                 /* widen from 1200 */
  width:min(1280px, 96vw);          /* fill screen on large displays */
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* === Tabs text: black + bold === */
.tab-radio [role="radio"],
.tab-radio [role="radio"] *,
.tab-radio [role="radio"] [data-testid="stMarkdownContainer"] p,
.tab-radio [role="radio"] [data-testid="stMarkdownContainer"] span{
  color:#000 !important;
  -webkit-text-fill-color:#000 !important;
  font-weight:700 !important;
}
.tab-radio [role="radio"][aria-checked="true"],
.tab-radio [role="radio"][aria-checked="true"] *,
.tab-radio [role="radio"][aria-checked="true"] [data-testid="stMarkdownContainer"] p,
.tab-radio [role="radio"][aria-checked="true"] [data-testid="stMarkdownContainer"] span{
  color:#000 !important;
  -webkit-text-fill-color:#000 !important;
  font-weight:700 !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* Center the Get risk estimate button only */
div[data-testid="stButton"] button[purpose="primary"] {
    margin-left: auto;
    margin-right: auto;
    display: block;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Import Roboto Slab from Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Roboto+Slab:wght@300;400;700&display=swap');

/* Apply globally */
html, body, [class*="css"]  {
    font-family: 'Roboto Slab', serif !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
.footer {
    position: relative;
    bottom: 0;
    width: 100%;
    padding: 1rem 0;
    margin-top: 2rem;
    text-align: center;
    font-size: 0.9rem;
    color: #ffffff;
}
.footer a {
    color: #ffffff;
    text-decoration: underline;
    margin: 0 10px;
}
.footer a:hover {
    color: #dbeafe;
}
</style>
""", unsafe_allow_html=True)


# ------------------ Branded header ------------------
st.markdown('<div class="header-box">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 5])
with col1:
    st.image("appLogo2.png", width=150)
with col2:
    st.markdown(
        """
        <div class="header-text">
          <h1>AI-based Diabetes Mellitus<br>Prediction Tool for Trauma Clinics</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
st.markdown('</div>', unsafe_allow_html=True)

# ------------------ About box ------------------
st.markdown(
    """
    <div class="infobox">
      <div class="icon">i</div>
      <div>
        <h4>About AIDMT</h4>
        <p>
          AIDMT is intended for use in <strong>trauma clinics</strong> as a decision-support tool to estimate the risk of <strong>prediabetes and diabetes</strong> based on routinely collected data. It does not provide a definitive diagnosis, and all results should be interpreted by qualified clinicians.”
        </p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="aidmt-sub">Enter values to estimate pre-/diabetes risk.</div>', unsafe_allow_html=True)

# ------------------ Session defaults for Admin settings ------------------
if "mdl_dir_in" not in st.session_state:
    st.session_state.mdl_dir_in = "final_model_top10"
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.50

# ------------------ Model loading ------------------
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

# Use session settings to load
try:
    pipe, FEATURES, RESOLVED_DIR = load_artifacts(st.session_state.mdl_dir_in)
except Exception as e:
    st.error(str(e))
    st.stop()

# ------------------ Friendly labels, aliases, units, helpers ------------------
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
}

ALIASES = {
    "Leukocytes": "LEU",
    "Waist Circumference": "WaistCircumference",
    "Age": "Age",
    "APTT": "APTT",
    "QUICK": "QUICK",
    "Potassium": "Potassium",
    "MCHC": "MCHC",
    "MCH": "MCH",
    YESNO_2_0_FEATURE: YESNO_2_0_FEATURE,
    SMOKING_YEARS_FEATURE: SMOKING_YEARS_FEATURE,
}

UNITS = {
    "Age": "years",
    "Leukocytes": "10^3/µL",  # (aka 10^9/L)
    "MCHC": "g/dL",
    "Waist Circumference": "cm",
    "APTT": "s",
    "QUICK": "%",
    "Potassium": "mmol/L",
    "MCH": "pg",
    YESNO_2_0_FEATURE: None,
    SMOKING_YEARS_FEATURE: "years",
}

REF_RANGES = {
    "Leukocytes": "≈ 4.3–10.0",
    "MCHC": "≈ 33–36",
    "Waist Circumference": "adult ≈ 60–160",
    "APTT": "≈ 21–29",
    "QUICK": "≈ 70–120",
    "Potassium": "≈ 3.5–5.1",
    "MCH": "≈ 28–33",
}

def label_with_unit(name: str) -> str:
    base = LABELS.get(name, name)
    unit = UNITS.get(name)
    return f"{base} ({unit})" if unit else base

def help_with_range(name: str) -> str | None:
    rr = REF_RANGES.get(name)
    return f"Typical range: {rr} {UNITS.get(name, '')}".strip() if rr else None

def inline_range_hint(name: str):
    rr = REF_RANGES.get(name)
    if not rr: return
    unit = UNITS.get(name, "")
    st.markdown(f"<div class='inline-hint'>Typical range: {rr} {unit}</div>", unsafe_allow_html=True)

def has_feat(friendly_name: str) -> bool:
    # Forgiving: render if FEATURES is missing/not iterable
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

# ------------------ Header NAV band ------------------
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

# ------------------ PAGES ------------------
if selected == "Home":
    # ------------------ Input form ------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>Patient values</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    inputs: dict[str, object] = {}

    with col1:
        if has_feat("Age"):
            inputs[key_for("Age")] = st.number_input(
                label_with_unit("Age"), min_value=0, max_value=120, value=45, step=1
            )

        if has_feat(YESNO_2_0_FEATURE):
            yn = st.radio(label_with_unit(YESNO_2_0_FEATURE), ["No", "Yes"], horizontal=True, index=0)
            inputs[key_for(YESNO_2_0_FEATURE)] = 2 if yn == "Yes" else 0

        if has_feat(SMOKING_YEARS_FEATURE):
            status = st.radio(label_with_unit(SMOKING_YEARS_FEATURE),
                              ["Non-smoker", "Ex-smoker", "Current smoker"], index=0)
            if status == "Non-smoker":
                st.markdown("<p style='color:#5b5e6a; font-size:.9rem;'>Years smoking: 0</p>", unsafe_allow_html=True)
                inputs[key_for(SMOKING_YEARS_FEATURE)] = 0
            else:
                years = st.number_input("Years smoking (years)", min_value=0, max_value=80, value=5, step=1)
                inputs[key_for(SMOKING_YEARS_FEATURE)] = years

        if has_feat("Leukocytes"):
            inputs[key_for("Leukocytes")] = st.number_input(
                label_with_unit("Leukocytes"),
                min_value=0.0,                 # 🚫 no negatives
                value=0.0,
                format="%.2f",
                help=help_with_range("Leukocytes")
            )


        if has_feat("Waist Circumference"):
            inputs[key_for("Waist Circumference")] = st.number_input(
                label_with_unit("Waist Circumference"),
                min_value=0.0,                 # 🚫 no negatives
                value=0.0,
                format="%.1f",
                help=help_with_range("Waist Circumference")
            )

    with col2:
        if has_feat("QUICK"):
            inputs[key_for("QUICK")] = st.number_input(
                label_with_unit("QUICK"),
                min_value=0.0,                 # 🚫 no negatives
                value=0.0,
                format="%.2f",
                help=help_with_range("QUICK")
            )

        if has_feat("APTT"):
            inputs[key_for("APTT")] = st.number_input(
                label_with_unit("APTT"),
                min_value=0.0,                 # 🚫 no negatives
                value=0.0,
                format="%.2f",
                help=help_with_range("APTT")
            )

        if has_feat("Potassium"):
            inputs[key_for("Potassium")] = st.number_input(
                label_with_unit("Potassium"),
                min_value=0.0,                 # 🚫 no negatives
                value=0.0,
                format="%.2f",
                help=help_with_range("Potassium")
            )

        if has_feat("MCHC"):
            inputs[key_for("MCHC")] = st.number_input(
                label_with_unit("MCHC"),
                min_value=0.0,                 # 🚫 no negatives
                value=0.0,
                format="%.2f",
                help=help_with_range("MCHC")
            )

        if has_feat("MCH"):
            inputs[key_for("MCH")] = st.number_input(
                label_with_unit("MCH"),
                min_value=0.0,                 # 🚫 no negatives
                value=0.0,
                format="%.2f",
                help=help_with_range("MCH")
            )

    # Button row (centered, spans both columns)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        submit = st.button("Get risk estimate")

    st.markdown('</div>', unsafe_allow_html=True)  # close card

    # Prediction & display
    threshold = st.session_state.threshold
    if submit:
        x1 = _coerce_row(inputs, FEATURES)
        try:
            proba = float(pipe.predict_proba(x1)[0, 1])
        except Exception:
            st.error("Unable to calculate risk. Please review inputs.")
            st.stop()

        t = max(0.0, min(1.0, threshold))
        label, cls = ("Low risk", "badge-low")
        if proba >= t:
            label, cls = ("High risk", "badge-high")
        elif proba >= t * 0.75:
            label, cls = ("Moderate risk", "badge-med")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>Result</h3>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.1rem;margin:.2rem 0;'>Estimated probability:</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='badge {cls}'><b>{proba*100:.1f}%</b> &nbsp;•&nbsp; {label}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Batch CSV stays on Home
    with st.expander("📄 Batch scoring (optional)"):
        st.write("Upload a CSV with the same column names as the model schema. "
                 "Friendly names (e.g., 'Leukocytes', 'Waist Circumference') are accepted and will be mapped.")
        csv = st.file_uploader("Upload CSV", type=["csv"])
        if csv is not None:
            try:
                df = pd.read_csv(csv)
                friendly_to_schema = {**{k: k for k in FEATURES}, **ALIASES}
                df.columns = [friendly_to_schema.get(c, c) for c in df.columns]
                df = df.reindex(columns=FEATURES)
                for c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

                if YESNO_2_0_FEATURE in df.columns:
                    df[YESNO_2_0_FEATURE] = df[YESNO_2_0_FEATURE].replace({"Yes": 2, "No": 0})
                    df[YESNO_2_0_FEATURE] = pd.to_numeric(df[YESNO_2_0_FEATURE], errors="coerce")

                if SMOKING_YEARS_FEATURE in df.columns:
                    df[SMOKING_YEARS_FEATURE] = df[SMOKING_YEARS_FEATURE].replace({"Non-smoker": 0})
                    df[SMOKING_YEARS_FEATURE] = pd.to_numeric(df[SMOKING_YEARS_FEATURE], errors="coerce")

                preds = pipe.predict_proba(df)[:, 1]
                out = df.copy()
                out["risk_proba"] = preds

                st.write(out.head())
                st.download_button(
                    "Download results",
                    out.to_csv(index=False).encode("utf-8"),
                    "aidmt_predictions.csv",
                    "text/csv"
                )
            except Exception:
                st.error("Could not score the file. Please check the columns and values.")

elif selected == "Info":
    st.markdown("<h3>About the Algorithm</h3>", unsafe_allow_html=True)
    st.write(
        """
        This risk prediction tool (AIDMT) was developed using anonymized trauma clinic data.
        Multiple machine learning algorithms (e.g., Random Forest, Gradient Boosting, XGBoost, MLP)
        were trained and evaluated with nested cross-validation. The final model uses a reduced set
        of the most predictive features, selected by SHAP importance analysis.

        **Important:** The tool is a decision-support system only and does not replace a clinical diagnosis.
        """
    )

    st.markdown("<h3>Example CSV Format</h3>", unsafe_allow_html=True)
    example = pd.DataFrame({
        "Age": [45, 62],
        "Previous High Blood Sugar Levels": [0, 2],
        "Smoking for how long": [0, 15],
        "Leukocytes": [6.2, 9.1],
        "MCHC": [34, 35],
        "Waist Circumference": [95, 110],
        "APTT": [27.1, 28.3],
        "QUICK": [105, 92],
        "Potassium": [4.2, 3.8],
        "MCH": [29, 31],
    })
    st.dataframe(example)
    st.caption("Values should follow the same units as indicated in the input form.")

    # Admin lives here now (no more above-tabs)
    with st.expander("⚙️ Admin", expanded=False):
        st.text_input(
            "Settings ▸ Model folder",
            key="mdl_dir_in",
            help="Folder containing model.skops (or diabetes_risk_pipeline.pkl) and feature_schema.json"
        )
        st.slider("Decision threshold", 0.0, 1.0, key="threshold", step=0.01)
        st.caption(f"Active model folder: {RESOLVED_DIR}")

# ------------------ Footer ------------------
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

