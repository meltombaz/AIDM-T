# AIDMT — Pre-/Diabetes Risk Prediction • Minimal, branded UI (no technical details on screen)

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Prefer skops (portable across numpy versions); fall back to joblib
try:
    from skops.io import load as skops_load
    HAS_SKOPS = True
except Exception:
    HAS_SKOPS = False

import joblib  # fallback


def num_with_unknown(label, *, key, min_value=None, max_value=None, step=None, fmt=None, default=None):
    c1, c2 = st.columns([6, 2])
    unknown = c2.checkbox("Unknown", key=f"{key}__unk")
    if unknown:
        return None  # will become NaN in _coerce_row
    return c1.number_input(label, min_value=min_value, max_value=max_value,
                           value=default if default is not None else (min_value or 0),
                           step=step, format=fmt)  # returns a number

def radio_yes_no_unknown(label, *, key):
    choice = st.radio(label, ["No", "Yes", "Unknown"], horizontal=True, key=key, index=0)
    return { "No": 0, "Yes": 2, "Unknown": None }[choice]

# --- Inline range hint helper ---
def inline_range_hint(name: str):
    rr = REF_RANGES.get(name)
    if not rr:
        return
    unit = UNITS.get(name, "")
    st.markdown(
        f"<div class='inline-hint'>Typical range: {rr} {unit}</div>",
        unsafe_allow_html=True
    )



# ------------------ Page setup ------------------
st.set_page_config(page_title="AIDMT — Pre-/Diabetes Risk", page_icon="🩺", layout="wide")


# ------------------ Styling ------------------
st.markdown("""
<style>
/* ===== Page background ===== */
.stApp { background-color:#004994; }

/* ===== Central content card ===== */
.block-container{
  max-width: 940px;
  margin: 2rem auto;
  padding: 2rem 2.2rem;
  background:#fff;
  border-radius: 18px;
  box-shadow: 0 10px 28px rgba(0,0,0,.18);
}

/* ===== Typography ===== */
h1,h2,h3,h4,h5,h6,p,span,div,label{ color:#000 !important; }
h1,h2 { letter-spacing:.2px; }
hr { border:0; height:1px; background:#eef0f4; }

/* ===== Header ===== */
.aidmt-title{ font-size:2.1rem; font-weight:800; color:#004994; margin:0 0 .25rem 0; }
.aidmt-sub{ color:#5b5e6a !important; margin-bottom:1.2rem; }

/* ===== Inputs ===== */
.stTextInput input, .stNumberInput input, .stTextArea textarea {
  background:#f6f8fb !important; color:#000 !important;
  border:1px solid #cfd7e3 !important; border-radius:12px !important;
}
.stNumberInput > div > div > input { padding:.7rem .9rem !important; }
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
  outline:none !important; border-color:#004994 !important;
  box-shadow:0 0 0 3px rgba(0,73,148,.15) !important;
}

/* Selectbox */
[data-baseweb="select"] > div {
  background:#f6f8fb !important; color:#000 !important;
  border:1px solid #cfd7e3 !important; border-radius:12px !important;
}

/* Sliders */
.stSlider > div[data-baseweb="slider"] > div { background:#cfd7e3 !important; }
.stSlider span[data-baseweb="slider"] div[role="slider"]{
  background:#004994 !important;
  box-shadow:0 0 0 4px rgba(0,73,148,.18) !important;
}

/* Radios/checkboxes */
.stRadio > label, .stCheckbox > label { color:#000 !important; }
.stRadio div[role="radio"]{ border:2px solid #004994 !important; }
.stRadio div[aria-checked="true"]{ background:#004994 !important; }

/* Expanders */
.streamlit-expanderHeader { font-weight:600; }
.streamlit-expander {
  border:1px solid #eef0f4 !important; border-radius:14px !important;
  background:#fafbff !important;
}

/* File uploader */
.stFileUploader > div {
  border:1px dashed #cfd7e3 !important; border-radius:14px !important;
  background:#f8fafc !important;
}
            
.stButton > button, .stButton > button * {
  display: flex !important;          
  align-items: center !important;    
  justify-content: center !important;
  height: 3.8rem;                    
  padding: 0 2rem;
  font-size: 1.2rem;
  font-weight: 700;

  background:#56184a !important;
  color:#ffffff !important;          /* force white text */
  border:none !important;
  border-radius:12px !important;
  box-shadow:0 6px 14px rgba(86,24,74,.25);
}


.stButton > button:hover {
  filter:brightness(1.06);
  box-shadow:0 8px 18px rgba(86,24,74,.32);
}
.stButton > button:disabled {
  background:#b9b9c2 !important; color:#f2f2f2 !important;
  box-shadow:none;
}

/* ===== Cards ===== */
.card{ background:#ffffff; border:1px solid #eef0f4; border-radius:16px; padding:18px; }
.card h3{ margin:0 0 .8rem 0; }

/* ===== Badges / results ===== */
.badge{ display:inline-block; padding:.25rem .6rem; border-radius:999px; font-size:.85rem; }
.badge-low{ background:#e8f5e9; color:#256029; }
.badge-med{ background:#fff8e1; color:#7a5c00; }
.badge-high{ background:#ffebee; color:#b71c1c; }

/* Results accent + high-risk color */
.result-accent{ color:#006226 !important; }
.risk-high{ color:#681c16 !important; }

/* ===== Infobox ===== */
.infobox{
  display:flex; gap:.75rem; align-items:flex-start;
  background:#f4f8ff;
  border:1px solid #cfd7e3;
  border-left:6px solid #004994;
  border-radius:12px;
  padding:12px 14px;
  margin: 0 0 1rem 0;
  color:#000 !important;
}
.infobox .icon{
  width:28px; height:28px; border-radius:999px;
  background:#004994; color:#fff;
  display:flex; align-items:center; justify-content:center;
  font-weight:800; font-size:16px;
  flex:0 0 28px;
}
.infobox h4{ margin:.1rem 0 .25rem 0; font-size:1rem; font-weight:700; }
.infobox p{ margin:0; font-size:.95rem; }
            
.header-box{
  background-color:#004994;     /* brand blue */
  border-radius:14px;
  padding:1rem 1.5rem;
  margin-bottom:1rem;
}

.header-text h1{
  color:#ffffff;                /* white text on blue */
  font-size:2rem;               /* bump if you want larger */
  font-weight:800;
  line-height:1.25;
  margin:0;
}

/* Optional: slightly smaller on narrow screens */
@media (max-width: 768px){
  .header-text h1{ font-size:1.5rem; }
}
            
.inline-hint{
  margin:.35rem 0 .1rem 0;
  font-size:.9rem;
  color:#5b5e6a !important;
}

/* ===== Footer ===== */
.footer{ color:#78808b !important; font-size:.85rem; margin-top:1rem; }
</style>
""", unsafe_allow_html=True)


# ------------------ Header & disclaimer ------------------
# --- Branded header: blue bar + logo + two-line title ---
st.markdown('<div class="header-box">', unsafe_allow_html=True)

col1, col2 = st.columns([1, 5])
with col1:
    st.image("appLogo2.png", width=150)  # adjust as you like (90–140 works well)
with col2:
    st.markdown(
        """
        <div class="header-text">
          <h1>AI-based Diabetes Mellitus<br>Prediction Tool for Trauma Clinics</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)  # close header-box


st.markdown("""
<div class="infobox">
  <div class="icon">i</div>
  <div>
    <h4>About AIDMT</h4>
    <p>
      AIDMT is designed for <strong>trauma clinics</strong> to estimate the risk of
      <strong>prediabetes and diabetes</strong> using routine data. It is a decision-support tool and
      <strong>not an official diagnosis</strong>. Results should be interpreted by qualified clinicians.
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="aidmt-sub">Enter values to estimate pre-/diabetes risk.</div>', unsafe_allow_html=True)


# ------------------ Model loading ------------------
BASE_DIR = Path(__file__).resolve().parent

def _has_model_files(p: Path) -> bool:
    return (p / "feature_schema.json").exists() and (
        (p / "model.skops").exists() or (p / "diabetes_risk_pipeline.pkl").exists()
    )

def _resolve_model_dir(user_text: str | None) -> Path:
    # default search order
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
    raise FileNotFoundError("Model files not found. Open Admin ▸ Settings and set the correct folder.")

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


# ------------------ Admin ------------------
with st.expander("⚙️ Admin", expanded=False):
    mdl_dir_in = st.text_input(
        "Settings ▸ Model folder",
        value="final_model_top10",
        help="Folder containing model.skops (or diabetes_risk_pipeline.pkl) and feature_schema.json"
    )
    threshold = st.slider("Decision threshold (optional)", 0.0, 1.0, 0.50, 0.01)
    st.caption("CSV batch scoring lives below the form once the model loads.")

# Load model
try:
    pipe, FEATURES, RESOLVED_DIR = load_artifacts(mdl_dir_in)
except Exception as e:
    st.error(str(e))
    st.stop()

# ------------------ Friendly labels & alias mapping ------------------
YESNO_2_0_FEATURE = "Previous High Blood Sugar Levels"
SMOKING_YEARS_FEATURE = "Smoking for how long"

# Friendly labels for UI
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

# Map friendly names -> actual schema keys (from your schema file)
ALIASES = {
    "Leukocytes": "LEU",
    "Waist Circumference": "WaistCircumference",
    # identity mappings for convenience (no harm if not used)
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
    "Leukocytes": "10^3/µL",      # (aka 10^9/L)
    "MCHC": "g/dL",
    "Waist Circumference": "cm",
    "APTT": "s",
    "QUICK": "%",
    "Potassium": "mmol/L",
    "MCH": "pg",
    YESNO_2_0_FEATURE: None,
    SMOKING_YEARS_FEATURE: "years",
}

# Optional: soft plausibility ranges (used only for help text)
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
    """Return the friendly label plus a unit tag if defined."""
    base = LABELS.get(name, name)
    unit = UNITS.get(name)
    return f"{base} ({unit})" if unit else base

def help_with_range(name: str) -> str | None:
    """Optional: return a small hint with reference-ish range."""
    rr = REF_RANGES.get(name)
    return f"Typical range: {rr} {UNITS.get(name, '')}".strip() if rr else None


def has_feat(friendly_name: str) -> bool:
    # Render by default if FEATURES isn't ready/iterable yet
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


# ------------------ Input form ------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("<h3>Patient values</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
inputs: dict[str, object] = {}

with col1:
    # Age
    if has_feat("Age"):
        inputs[key_for("Age")] = st.number_input(
            label_with_unit("Age"), min_value=0, max_value=120, value=45, step=1
        )

    # Previous high blood sugar (radio -> no unit)
    if has_feat(YESNO_2_0_FEATURE):
        yn = st.radio(label_with_unit(YESNO_2_0_FEATURE), ["No", "Yes"], horizontal=True, index=0)
        inputs[key_for(YESNO_2_0_FEATURE)] = 2 if yn == "Yes" else 0

    # Smoking history + years
    if has_feat(SMOKING_YEARS_FEATURE):
        status = st.radio(label_with_unit(SMOKING_YEARS_FEATURE),
                          ["Non-smoker", "Ex-smoker", "Current smoker"], index=0)
        if status == "Non-smoker":
            st.markdown("<p style='color:#5b5e6a; font-size:.9rem;'>Years smoking: 0</p>", unsafe_allow_html=True)
            inputs[key_for(SMOKING_YEARS_FEATURE)] = 0
        else:
            years = st.number_input("Years smoking (years)", min_value=0, max_value=80, value=5, step=1)
            inputs[key_for(SMOKING_YEARS_FEATURE)] = years

    # Leukocytes
# Leukocytes
    if has_feat("Leukocytes"):
        inputs[key_for("Leukocytes")] = st.number_input(
            label_with_unit("Leukocytes"),
            value=0.0, format="%.2f",
            help=help_with_range("Leukocytes")  # keeps the hover tooltip
    )
        inline_range_hint("Leukocytes")        # always-visible hint under the field

    # Waist circumference
    if has_feat("Waist Circumference"):
        inputs[key_for("Waist Circumference")] = st.number_input(
            label_with_unit("Waist Circumference"),
            value=0.0, format="%.1f",
            help=help_with_range("Waist Circumference")
    )
        inline_range_hint("Waist Circumference")

with col2:
    # QUICK
    if has_feat("QUICK"):
        inputs[key_for("QUICK")] = st.number_input(
            label_with_unit("QUICK"),
            value=0.0, format="%.2f",
            help=help_with_range("QUICK")
    )
        inline_range_hint("QUICK")

# APTT
    if has_feat("APTT"):
        inputs[key_for("APTT")] = st.number_input(
            label_with_unit("APTT"),
            value=0.0, format="%.2f",
            help=help_with_range("APTT")
    )
        inline_range_hint("APTT")

    # Potassium
    if has_feat("Potassium"):
        inputs[key_for("Potassium")] = st.number_input(
            label_with_unit("Potassium"),
            value=0.0, format="%.2f",
            help=help_with_range("Potassium")
    )
        inline_range_hint("Potassium")

    # MCHC
    if has_feat("MCHC"):
        inputs[key_for("MCHC")] = st.number_input(
            label_with_unit("MCHC"),
            value=0.0, format="%.2f",
            help=help_with_range("MCHC")
    )
        inline_range_hint("MCHC")

    # MCH
    if has_feat("MCH"):
        inputs[key_for("MCH")] = st.number_input(
            label_with_unit("MCH"),
            value=0.0, format="%.2f",
            help=help_with_range("MCH")
    )
        inline_range_hint("MCH")


submit = st.button("Get risk estimate")
st.markdown('</div>', unsafe_allow_html=True)  # close card


# ------------------ Prediction & display ------------------
def _coerce_row(inputs: dict, features: list[str]) -> pd.DataFrame:
    """
    Build a one-row DataFrame matching the model's schema.
    All fields are attempted as numeric; non-parsable -> NaN.
    """
    row = {}
    for feat in features:
        v = inputs.get(feat, None)
        if v is None or (isinstance(v, str) and v.strip() == ""):
            row[feat] = np.nan
            continue
        # numeric coercion for this model (all 10 are numeric)
        try:
            fv = float(v)
            row[feat] = int(fv) if fv.is_integer() else fv
        except Exception:
            row[feat] = np.nan
    return pd.DataFrame([row]).reindex(columns=features)

if submit:
    x1 = _coerce_row(inputs, FEATURES)
    try:
        proba = float(pipe.predict_proba(x1)[0, 1])
    except Exception:
        st.error("Unable to calculate risk. Please review inputs.")
        st.stop()

    # Threshold badge
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


# ------------------ Batch CSV ------------------
with st.expander("📄 Batch scoring (optional)"):
    st.write("Upload a CSV with the same column names as the model schema. "
             "Friendly names (e.g., 'Leukocytes', 'Waist Circumference') are accepted and will be mapped.")
    csv = st.file_uploader("Upload CSV", type=["csv"])
    if csv is not None:
        try:
            df = pd.read_csv(csv)

            # Map friendly columns to schema keys, keep others as-is
            friendly_to_schema = {**{k: k for k in FEATURES}, **ALIASES}
            df.columns = [friendly_to_schema.get(c, c) for c in df.columns]

            # Reindex to schema order and numeric coercion
            df = df.reindex(columns=FEATURES)
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            # Special mappings for known fields
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


# ------------------ Footer ------------------
st.markdown('<div class="footer">AIDMT — For decision support and research use only.</div>', unsafe_allow_html=True)
