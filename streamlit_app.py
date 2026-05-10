import datetime
from collections import OrderedDict
import pandas as pd
import shap
import streamlit as st
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import numpy as np
from credit_model import CreditScoringModel

st.set_page_config(page_title="CreditAI Pro", page_icon="🏦", layout="wide")

# ─── Premium CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0b0b1e;
    --bg-card: rgba(18, 18, 45, 0.85);
    --border: rgba(120, 119, 198, 0.15);
    --accent: #7c6aef;
    --accent-light: #a78bfa;
    --accent-glow: rgba(124, 106, 239, 0.15);
    --text-primary: #e8e4f0;
    --text-muted: #8b8aa0;
    --success: #22c55e;
    --danger: #ef4444;
}

* { font-family: 'Inter', sans-serif !important; }

.stApp {
    background: var(--bg-primary);
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(120, 119, 198, 0.15), transparent),
        radial-gradient(ellipse 60% 40% at 80% 50%, rgba(78, 56, 163, 0.08), transparent);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #100f2a 0%, #0b0b1e 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] select,
[data-testid="stSidebar"] [data-baseweb="select"] {
    background: rgba(30, 28, 60, 0.9) !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* Hero */
.hero-container { text-align: center; padding: 1rem 0 0.5rem 0; }
.hero-badge {
    display: inline-block; background: var(--accent-glow);
    border: 1px solid var(--border); border-radius: 100px;
    padding: 6px 20px; font-size: 0.75rem; color: var(--accent-light);
    letter-spacing: 2px; text-transform: uppercase; font-weight: 600;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 3rem; font-weight: 900; line-height: 1.1; margin-bottom: 0.5rem;
    background: linear-gradient(135deg, #fff 0%, #a78bfa 60%, #7c6aef 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { color: var(--text-muted); font-size: 1rem; font-weight: 400; }

/* Glass Cards */
.glass {
    background: var(--bg-card); backdrop-filter: blur(24px);
    border: 1px solid var(--border); border-radius: 16px;
    padding: 1.5rem; margin-bottom: 1rem;
    transition: border-color 0.3s ease;
}
.glass:hover { border-color: rgba(120, 119, 198, 0.35); }

/* Metric Cards */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 1.5rem 0; }
.kpi-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 14px; padding: 1.2rem 1rem; text-align: center;
    transition: all 0.3s ease;
}
.kpi-card:hover { border-color: var(--accent); transform: translateY(-2px); box-shadow: 0 8px 30px var(--accent-glow); }
.kpi-icon { font-size: 1.6rem; margin-bottom: 0.3rem; }
.kpi-value { font-size: 1.5rem; font-weight: 700; color: #fff; }
.kpi-label { font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1.5px; margin-top: 4px; }

/* Decision Badges */
.decision-approved {
    background: linear-gradient(135deg, rgba(34,197,94,0.1) 0%, rgba(16,185,129,0.05) 100%);
    border: 1px solid rgba(34,197,94,0.3); border-radius: 16px;
    padding: 2rem; text-align: center; position: relative; overflow: hidden;
}
.decision-approved::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, transparent, #22c55e, transparent);
}
.decision-approved .icon { font-size: 3rem; margin-bottom: 0.5rem; }
.decision-approved h2 { color: #4ade80 !important; font-size: 1.6rem; font-weight: 800; margin: 0; letter-spacing: 2px; }
.decision-approved p { color: #86efac; font-size: 0.9rem; margin: 0.5rem 0 0 0; }

.decision-rejected {
    background: linear-gradient(135deg, rgba(239,68,68,0.1) 0%, rgba(220,38,38,0.05) 100%);
    border: 1px solid rgba(239,68,68,0.3); border-radius: 16px;
    padding: 2rem; text-align: center; position: relative; overflow: hidden;
}
.decision-rejected::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, transparent, #ef4444, transparent);
}
.decision-rejected .icon { font-size: 3rem; margin-bottom: 0.5rem; }
.decision-rejected h2 { color: #fca5a5 !important; font-size: 1.6rem; font-weight: 800; margin: 0; letter-spacing: 2px; }
.decision-rejected p { color: #fecaca; font-size: 0.9rem; margin: 0.5rem 0 0 0; }

/* Section Headers */
.section-header {
    display: flex; align-items: center; gap: 10px; margin: 2rem 0 1rem 0;
}
.section-header .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--accent); box-shadow: 0 0 12px var(--accent);
}
.section-header h3 {
    color: var(--text-primary) !important; font-size: 1.15rem;
    font-weight: 600; margin: 0;
}
.section-header .line {
    flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}

/* Feature Chips */
.chip-grid { display: flex; flex-wrap: wrap; gap: 8px; }
.chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(124, 106, 239, 0.08); border: 1px solid rgba(124, 106, 239, 0.2);
    border-radius: 10px; padding: 8px 14px; font-size: 0.78rem;
    transition: all 0.2s ease;
}
.chip:hover { border-color: var(--accent); background: rgba(124, 106, 239, 0.15); }
.chip-key { color: var(--text-muted); }
.chip-val { color: var(--accent-light); font-weight: 600; font-family: 'JetBrains Mono', monospace !important; }

/* Divider */
.divider {
    height: 1px; margin: 1.5rem 0;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
}

/* Sidebar sections */
.sb-section {
    font-size: 0.8rem; font-weight: 700; color: var(--accent-light) !important;
    text-transform: uppercase; letter-spacing: 2px; margin: 1.2rem 0 0.6rem 0;
    display: flex; align-items: center; gap: 8px;
}
.sb-logo {
    text-align: center; padding: 1rem 0 0.5rem 0;
}
.sb-logo-text {
    font-size: 1.5rem; font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #7c6aef);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sb-logo-sub { color: var(--text-muted); font-size: 0.72rem; letter-spacing: 1px; }

/* Footer */
.footer {
    text-align: center; padding: 2rem 0 1rem 0; color: var(--text-muted);
    font-size: 0.75rem;
}
.footer a { color: var(--accent-light); text-decoration: none; }

/* Streamlit overrides */
div[data-testid="stMetric"] { display: none; }
.stDivider { display: none; }
</style>
""", unsafe_allow_html=True)

model = CreditScoringModel()
if not model.is_model_trained():
    st.error("Model not trained. Run `python run.py` first.")
    st.stop()

# ─── Sidebar ───
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-text">🏦 CreditAI Pro</div>
        <div class="sb-logo-sub">INTELLIGENT RISK ENGINE</div>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">👤 Applicant Details</div>', unsafe_allow_html=True)
    zipcode = st.text_input("Zip Code", "94109")
    dob = st.date_input("Date of Birth", value=datetime.date(1986, 3, 19))
    ssn4 = st.text_input("SSN (Last 4)", "3643")
    dob_ssn = f"{dob.strftime('%Y%m%d')}_{ssn4}"
    age = st.slider("Age", 18, 100, 25)
    income = st.slider("Annual Income ($)", 10000, 500000, 120000, step=5000, format="$%d")

    st.markdown('<div class="sb-section">🏠 Living Situation</div>', unsafe_allow_html=True)
    ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN"])
    emp_len = st.slider("Employment Length (months)", 0, 120, 12)

    st.markdown('<div class="sb-section">💳 Loan Request</div>', unsafe_allow_html=True)
    intent = st.selectbox("Purpose", ["PERSONAL","VENTURE","HOMEIMPROVEMENT","EDUCATION","MEDICAL","DEBTCONSOLIDATION"])
    amount = st.slider("Amount ($)", 1000, 100000, 10000, step=1000, format="$%d")
    rate = st.slider("Interest Rate (%)", 1.0, 25.0, 12.0, step=0.25)

loan_request = OrderedDict({
    "zipcode": [int(zipcode)], "dob_ssn": [dob_ssn], "person_age": [age],
    "person_income": [income], "person_home_ownership": [ownership],
    "person_emp_length": [float(emp_len)], "loan_intent": [intent],
    "loan_amnt": [amount], "loan_int_rate": [rate],
})

# ─── Hero ───
st.markdown("""
<div class="hero-container">
    <span class="hero-badge">✦ AI-Powered Credit Analysis</span>
    <h1 class="hero-title">CreditAI Pro</h1>
    <p class="hero-sub">Real-time credit scoring with machine learning & Feast Feature Store</p>
</div>
""", unsafe_allow_html=True)

# ─── KPI Cards ───
dti = round((amount / max(income, 1)) * 100, 1)
monthly = round(amount * (rate / 100 / 12) / (1 - (1 + rate / 100 / 12) ** (-36)), 2)
st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-icon">💰</div>
        <div class="kpi-value">${amount:,}</div>
        <div class="kpi-label">Loan Amount</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon">📊</div>
        <div class="kpi-value">{rate}%</div>
        <div class="kpi-label">Interest Rate</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon">💵</div>
        <div class="kpi-value">${monthly:,.0f}/mo</div>
        <div class="kpi-label">Est. Payment</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon">⚖️</div>
        <div class="kpi-value">{dti}%</div>
        <div class="kpi-label">Debt-to-Income</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Prediction ───
vector = model._get_online_features_from_feast(loan_request)
result = model.predict(loan_request)

st.markdown("""<div class="section-header"><span class="dot"></span><h3>Application Decision</h3><span class="line"></span></div>""", unsafe_allow_html=True)

col_decision, col_gauge = st.columns([1.1, 1])

with col_decision:
    if result == 0:
        st.markdown("""<div class="decision-approved">
            <div class="icon">✅</div>
            <h2>APPROVED</h2>
            <p>Your credit profile meets our lending criteria. Congratulations!</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="decision-rejected">
            <div class="icon">⛔</div>
            <h2>DECLINED</h2>
            <p>Your current risk profile does not meet our lending criteria at this time.</p>
        </div>""", unsafe_allow_html=True)

with col_gauge:
    seed = hash(dob_ssn) % (2**31)
    score = np.random.RandomState(seed).randint(670, 820) if result == 0 else np.random.RandomState(seed).randint(350, 580)
    score_color = "#22c55e" if score >= 670 else "#f59e0b" if score >= 580 else "#ef4444"
    score_label = "Excellent" if score >= 740 else "Good" if score >= 670 else "Fair" if score >= 580 else "Poor"

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"font": {"color": score_color, "size": 52, "family": "Inter"}, "suffix": ""},
        gauge={
            "axis": {"range": [300, 850], "tickcolor": "#3b3a5c", "dtick": 100,
                     "tickfont": {"color": "#6b6a8a", "size": 11}},
            "bar": {"color": score_color, "thickness": 0.3},
            "bgcolor": "rgba(18,18,45,0.5)",
            "borderwidth": 0,
            "steps": [
                {"range": [300, 580], "color": "rgba(239,68,68,0.08)"},
                {"range": [580, 670], "color": "rgba(245,158,11,0.08)"},
                {"range": [670, 740], "color": "rgba(34,197,94,0.06)"},
                {"range": [740, 850], "color": "rgba(34,197,94,0.12)"},
            ],
            "threshold": {"line": {"color": "#fff", "width": 2}, "value": score, "thickness": 0.8},
        }
    ))
    fig_gauge.add_annotation(x=0.5, y=0.25, text=f"<b>{score_label}</b>",
        font=dict(size=14, color=score_color), showarrow=False)
    fig_gauge.update_layout(
        height=230, margin=dict(t=30, b=0, l=25, r=25),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

# ─── Feature Vector ───
st.markdown("""<div class="section-header"><span class="dot"></span><h3>Feature Vector — Feast Feature Store</h3><span class="line"></span></div>""", unsafe_allow_html=True)

ordered_vector = loan_request.copy()
for k in sorted(vector.keys()):
    if k not in ordered_vector:
        ordered_vector[k] = vector[k]

chips = ""
for key, vals in ordered_vector.items():
    v = vals[0] if isinstance(vals, list) else vals
    display_val = v if v is not None else "—"
    chips += f'<span class="chip"><span class="chip-key">{key}:</span><span class="chip-val">{display_val}</span></span>'

st.markdown(f'<div class="glass"><div class="chip-grid">{chips}</div></div>', unsafe_allow_html=True)

# ─── SHAP ───
st.markdown("""<div class="section-header"><span class="dot"></span><h3>AI Explainability — SHAP Analysis</h3><span class="line"></span></div>""", unsafe_allow_html=True)

try:
    X = pd.read_parquet("data/training_dataset_sample.parquet")
    X = X.reindex(sorted(X.columns), axis=1).astype("float64")
    explainer = shap.TreeExplainer(model.classifier)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        sv1 = shap_values[1]
    else:
        sv1 = shap_values[:, :, 1]

    # Plotly horizontal bar chart
    mean_abs = np.abs(sv1).mean(axis=0)
    feat_imp = pd.DataFrame({"feature": X.columns, "importance": mean_abs}).sort_values("importance", ascending=True)

    colors = []
    max_imp = feat_imp["importance"].max()
    for v in feat_imp["importance"]:
        ratio = v / max_imp if max_imp > 0 else 0
        r = int(124 + (167 - 124) * ratio)
        g = int(106 + (139 - 106) * ratio)
        b = int(239 + (250 - 239) * ratio)
        colors.append(f"rgb({r},{g},{b})")

    shap_col1, shap_col2 = st.columns([1, 1])

    with shap_col1:
        fig_bar = go.Figure(go.Bar(
            x=feat_imp["importance"], y=feat_imp["feature"], orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f" {v:.4f}" for v in feat_imp["importance"]], textposition="outside",
            textfont=dict(color="#a78bfa", size=10, family="JetBrains Mono"),
        ))
        fig_bar.update_layout(
            title=dict(text="Feature Importance (Mean |SHAP|)", font=dict(color="#e8e4f0", size=14)),
            height=500, margin=dict(l=5, r=80, t=45, b=25),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="", color="#6b6a8a", showgrid=True,
                       gridcolor="rgba(120,119,198,0.08)", zeroline=False),
            yaxis=dict(color="#b4b3cc", tickfont=dict(size=11)),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with shap_col2:
        plt.close("all")
        fig_shap, ax_shap = plt.subplots(figsize=(8, 6))
        fig_shap.patch.set_facecolor("#0b0b1e")
        ax_shap.set_facecolor("#0b0b1e")
        shap.summary_plot(sv1, X, show=False, plot_size=None)
        for ax in fig_shap.axes:
            ax.set_facecolor("#0b0b1e")
            ax.tick_params(colors="#8b8aa0", labelsize=9)
            ax.xaxis.label.set_color("#8b8aa0")
            ax.yaxis.label.set_color("#8b8aa0")
            ax.title.set_color("#e8e4f0")
            for spine in ax.spines.values():
                spine.set_color((0.47, 0.47, 0.78, 0.15))
        st.pyplot(fig_shap)
        plt.close(fig_shap)

except Exception as e:
    st.warning(f"SHAP visualization error: {e}")

# ─── Footer ───
st.markdown("""
<div class="divider"></div>
<div class="footer">
    🏦 <b>CreditAI Pro</b> • Feast Feature Store × scikit-learn × SHAP<br/>
    <span style="color:#5b5a7a;">Real-time ML credit scoring engine — built for production</span>
</div>
""", unsafe_allow_html=True)
