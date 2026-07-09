import streamlit as st
import numpy as np
import joblib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NephroScan · CKD Intelligence",
    page_icon="🫘",
    layout="centered",
)

# ── Inject full custom CSS + animations ──────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap" rel="stylesheet">

<style>
/* ── Reset & root variables ── */
:root {
  --bg-deep:    #050d1a;
  --bg-card:    #0a1628;
  --bg-panel:   #0d1e35;
  --teal:       #00e5c3;
  --teal-dim:   #00b89a;
  --cyan:       #38bdf8;
  --amber:      #f59e0b;
  --danger:     #f43f5e;
  --success:    #10b981;
  --text-pri:   #e2eaf6;
  --text-sec:   #7a9bbf;
  --border:     rgba(0,229,195,0.15);
  --glow:       0 0 30px rgba(0,229,195,0.18);
  --font-disp:  'Syne', sans-serif;
  --font-body:  'DM Sans', sans-serif;
}

/* ── Global overrides ── */
html, body, [class*="css"], .stApp {
  background: var(--bg-deep) !important;
  font-family: var(--font-body) !important;
  color: var(--text-pri) !important;
}

/* hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 1.5rem 4rem !important; max-width: 860px !important; }

/* ── Animated starfield background ── */
.stApp::before {
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse 80% 60% at 50% -10%, rgba(0,229,195,0.07) 0%, transparent 70%),
    radial-gradient(ellipse 50% 40% at 90% 80%,  rgba(56,189,248,0.05) 0%, transparent 60%);
  pointer-events: none;
  z-index: 0;
}

/* ── Animated grid overlay ── */
.stApp::after {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(0,229,195,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,229,195,0.03) 1px, transparent 1px);
  background-size: 48px 48px;
  animation: gridPulse 8s ease-in-out infinite;
  pointer-events: none;
  z-index: 0;
}
@keyframes gridPulse {
  0%, 100% { opacity: 0.6; }
  50%       { opacity: 1.0; }
}

/* ── Hero header ── */
.hero-wrap {
  position: relative;
  text-align: center;
  padding: 48px 24px 36px;
  margin-bottom: 8px;
}
.hero-badge {
  display: inline-block;
  font-family: var(--font-disp);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--teal);
  border: 1px solid var(--border);
  border-radius: 50px;
  padding: 5px 18px;
  margin-bottom: 20px;
  animation: fadeSlideDown 0.6s ease both;
  background: rgba(0,229,195,0.06);
}
.hero-title {
  font-family: var(--font-disp);
  font-size: clamp(36px, 7vw, 62px);
  font-weight: 800;
  line-height: 1.05;
  color: #fff;
  margin: 0 0 10px;
  animation: fadeSlideDown 0.7s ease both;
  letter-spacing: -1px;
}
.hero-title span {
  background: linear-gradient(135deg, var(--teal) 0%, var(--cyan) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-sub {
  font-size: 16px;
  color: var(--text-sec);
  font-weight: 300;
  margin-bottom: 28px;
  animation: fadeSlideDown 0.8s ease both;
}
.accuracy-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: rgba(16,185,129,0.1);
  border: 1px solid rgba(16,185,129,0.3);
  border-radius: 50px;
  padding: 8px 20px;
  font-size: 13px;
  font-weight: 500;
  color: var(--success);
  animation: fadeSlideDown 0.9s ease both;
}
.accuracy-pill::before {
  content: '●';
  font-size: 8px;
  animation: blink 1.5s ease infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── Divider ── */
.neo-divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--teal-dim), transparent);
  margin: 28px 0;
  opacity: 0.4;
}

/* ── Section label ── */
.section-label {
  font-family: var(--font-disp);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--teal);
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 10px;
}
.section-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}

/* ── Input fields ── */
.stNumberInput > div > div > input,
.stSelectbox > div > div {
  background: var(--bg-panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text-pri) !important;
  font-family: var(--font-body) !important;
  font-size: 15px !important;
  transition: border-color 0.25s, box-shadow 0.25s !important;
}
.stNumberInput > div > div > input:focus,
.stSelectbox > div > div:focus-within {
  border-color: var(--teal) !important;
  box-shadow: 0 0 0 3px rgba(0,229,195,0.12) !important;
}
.stNumberInput label, .stSelectbox label {
  font-family: var(--font-body) !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  color: var(--text-sec) !important;
  letter-spacing: 0.3px !important;
  margin-bottom: 4px !important;
}

/* ── Input card wrappers ── */
.input-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 24px;
  margin-bottom: 16px;
  box-shadow: var(--glow);
  transition: border-color 0.3s;
}
.input-card:hover { border-color: rgba(0,229,195,0.3); }
.input-card-title {
  font-family: var(--font-disp);
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--teal);
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 8px;
}

/* ── Predict button ── */
.stButton > button {
  width: 100% !important;
  background: linear-gradient(135deg, #007a68 0%, var(--teal) 50%, var(--cyan) 100%) !important;
  color: #050d1a !important;
  font-family: var(--font-disp) !important;
  font-size: 16px !important;
  font-weight: 700 !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 18px 40px !important;
  cursor: pointer !important;
  position: relative !important;
  overflow: hidden !important;
  transition: transform 0.2s, box-shadow 0.2s !important;
  box-shadow: 0 4px 24px rgba(0,229,195,0.3) !important;
  margin-top: 8px !important;
}
.stButton > button::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, transparent 40%, rgba(255,255,255,0.15) 60%, transparent 80%);
  transform: translateX(-100%);
  transition: transform 0.5s;
}
.stButton > button:hover::before { transform: translateX(100%); }
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 32px rgba(0,229,195,0.4) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Result cards ── */
.result-danger {
  background: linear-gradient(135deg, rgba(244,63,94,0.1) 0%, rgba(244,63,94,0.05) 100%);
  border: 1px solid rgba(244,63,94,0.4);
  border-radius: 16px;
  padding: 28px 32px;
  text-align: center;
  animation: resultReveal 0.5s cubic-bezier(0.34,1.56,0.64,1) both;
}
.result-safe {
  background: linear-gradient(135deg, rgba(16,185,129,0.1) 0%, rgba(16,185,129,0.05) 100%);
  border: 1px solid rgba(16,185,129,0.4);
  border-radius: 16px;
  padding: 28px 32px;
  text-align: center;
  animation: resultReveal 0.5s cubic-bezier(0.34,1.56,0.64,1) both;
}
@keyframes resultReveal {
  from { opacity:0; transform: scale(0.9) translateY(10px); }
  to   { opacity:1; transform: scale(1) translateY(0); }
}
.result-icon { font-size: 48px; margin-bottom: 12px; }
.result-title {
  font-family: var(--font-disp);
  font-size: 22px;
  font-weight: 800;
  letter-spacing: -0.5px;
  margin-bottom: 8px;
}
.result-danger .result-title { color: var(--danger); }
.result-safe .result-title   { color: var(--success); }
.result-desc {
  font-size: 14px;
  color: var(--text-sec);
  max-width: 400px;
  margin: 0 auto;
  line-height: 1.6;
}

/* ── Footer ── */
.neo-footer {
  text-align: center;
  font-size: 12px;
  color: var(--text-sec);
  margin-top: 48px;
  padding-top: 24px;
  border-top: 1px solid var(--border);
  opacity: 0.7;
  letter-spacing: 0.5px;
}

/* ── Fade/slide animations ── */
@keyframes fadeSlideDown {
  from { opacity:0; transform: translateY(-16px); }
  to   { opacity:1; transform: translateY(0); }
}

/* ── Streamlit element resets ── */
.stAlert { display: none !important; }  /* hide default alerts — we use custom */
div[data-testid="stVerticalBlock"] > div { animation: fadeSlideDown 0.5s ease both; }

/* Selectbox dropdown */
div[data-baseweb="select"] > div {
  background: var(--bg-panel) !important;
  border-color: var(--border) !important;
}
div[data-baseweb="popover"] { background: var(--bg-panel) !important; }
li[role="option"] { color: var(--text-pri) !important; background: var(--bg-panel) !important; }
li[role="option"]:hover { background: rgba(0,229,195,0.1) !important; }

/* ── Fix selectbox selected-value text visibility ── */
div[data-baseweb="select"] div,
div[data-baseweb="select"] span {
  color: var(--text-pri) !important;
}

/* dropdown arrow icon */
div[data-baseweb="select"] svg {
  fill: var(--teal) !important;
}

/* placeholder text (if any) */
div[data-baseweb="select"] div[class*="placeholder"] {
  color: var(--text-sec) !important;
}

/* Number input spinners */
button[kind="icon"] { color: var(--teal) !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
model  = joblib.load("ckd_model.pkl")
scaler = joblib.load("scaler.pkl")

# ── Hero section ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-badge">🫘 &nbsp; AI-Powered Nephrology</div>
  <h1 class="hero-title">Nephro<span>Scan</span></h1>
  <p class="hero-sub">Advanced Chronic Kidney Disease Risk Intelligence</p>
  <div class="accuracy-pill">Model Accuracy &nbsp;·&nbsp; 99.00%</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="neo-divider"></div>', unsafe_allow_html=True)

# ── Numerical inputs ──────────────────────────────────────────────────────────
st.markdown('<div class="section-label">📊 &nbsp; Biomarker Panel</div>', unsafe_allow_html=True)

st.markdown('<div class="input-card"><div class="input-card-title">🔬 Laboratory Values</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    sc   = st.number_input("Serum Creatinine  (mg/dL)", min_value=0.0, step=0.1, format="%.2f")
    al   = st.number_input("Albumin Level  (g/dL)",     min_value=0.0, step=0.1, format="%.2f")
    hemo = st.number_input("Hemoglobin  (g/dL)",        min_value=0.0, step=0.1, format="%.2f")
with col2:
    bp   = st.number_input("Blood Pressure  (mmHg)",    min_value=0.0, step=1.0,  format="%.1f")
    age  = st.number_input("Patient Age  (years)",      min_value=1,   max_value=120, step=1)
st.markdown('</div>', unsafe_allow_html=True)

# ── Categorical inputs ────────────────────────────────────────────────────────
st.markdown('<div class="section-label">🏥 &nbsp; Clinical Indicators</div>', unsafe_allow_html=True)

st.markdown('<div class="input-card"><div class="input-card-title">⚕️ Comorbidities & Symptoms</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    dm     = st.selectbox("Diabetes Mellitus",          options=[0, 1], format_func=lambda x: "Yes" if x else "No")
    htn    = st.selectbox("Hypertension",               options=[0, 1], format_func=lambda x: "Yes" if x else "No")
with col4:
    appet  = st.selectbox("Appetite",                   options=[0, 1], format_func=lambda x: "Good" if x else "Poor")
    ane    = st.selectbox("Anemia",                     options=[0, 1], format_func=lambda x: "Yes" if x else "No")
st.markdown('</div>', unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

predict_clicked = st.button("⟡  Run Diagnostic Analysis")

if predict_clicked:
    input_data   = np.array([[sc, al, hemo, bp, dm, htn, age, appet, ane]])
    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)

    st.markdown('<div class="neo-divider"></div>', unsafe_allow_html=True)

    if prediction[0] == 1:
        st.markdown("""
        <div class="result-danger">
          <div class="result-icon">⚠️</div>
          <div class="result-title">Elevated CKD Risk Detected</div>
          <p class="result-desc">
            The diagnostic model indicates a high probability of Chronic Kidney Disease.
            Immediate referral to a nephrologist and confirmatory testing is strongly advised.
          </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-safe">
          <div class="result-icon">✅</div>
          <div class="result-title">Low Risk — No CKD Detected</div>
          <p class="result-desc">
            Biomarker and clinical indicators fall within acceptable ranges.
            Routine monitoring and a healthy lifestyle are still recommended.
          </p>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="neo-footer">
  NephroScan · AI Clinical Decision Support &nbsp;·&nbsp; For informational use only — not a substitute for professional medical advice.
</div>
""", unsafe_allow_html=True)
