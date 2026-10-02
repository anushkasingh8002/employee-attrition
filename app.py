from textwrap import dedent

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px


# =========================================================
# HELPER
# =========================================================
def render_html(html):
    st.html(dedent(html))


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# SESSION STATE
# =========================================================
if "page" not in st.session_state:
    st.session_state.page = "Employee Attrition Prediction"

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    model = joblib.load("rf.pkl")
    columns = joblib.load("columns.pkl")
    return model, columns


model, model_columns = load_model()

# Modern job roles are mapped to the closest role available in the
# original IBM HR dataset. The mapping is a proxy, not a newly learned
# occupation category.
try:
    job_role_proxy = joblib.load("job_role_proxy.pkl")
except FileNotFoundError:
    job_role_proxy = {
        "AI/ML Engineer": "Research Scientist",
        "Software Developer": "Research Scientist"
    }


# =========================================================
# GLOBAL CSS
# =========================================================
render_html("""
<style>

.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(59, 130, 246, 0.10), transparent 30%),
        radial-gradient(circle at 85% 20%, rgba(139, 92, 246, 0.09), transparent 28%),
        #070b17;
    color: #e8eefc;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

h1, h2, h3, h4 {
    color: #f4f7ff !important;
}

p, label {
    color: #aebbd2;
}


/* =========================
   SIDEBAR
   ========================= */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #0a1020 0%,
            #080d1a 55%,
            #070b15 100%
        );
    border-right: 1px solid rgba(148, 163, 184, 0.10);
}

.brand {
    padding: 10px 4px 24px 4px;
}

.brand-title {
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: 2px;
    color: #f5f7ff;
}

.brand-subtitle {
    margin-top: 4px;
    color: #7f8ca5;
    font-size: 0.78rem;
    letter-spacing: 0.4px;
}

.sidebar-section {
    margin-top: 20px;
    margin-bottom: 10px;
    color: #687792;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.status-box {
    margin-top: 14px;
    padding: 14px;
    border-radius: 14px;
    background: rgba(15, 23, 42, 0.78);
    border: 1px solid rgba(96, 165, 250, 0.12);
}

.status-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
    color: #dce8ff;
    font-size: 0.84rem;
    font-weight: 700;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 10px rgba(34, 197, 94, 0.7);
}

.status-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid rgba(148, 163, 184, 0.07);
}

.status-row:last-child {
    border-bottom: none;
}

.status-label {
    color: #71809b;
    font-size: 0.75rem;
}

.status-value {
    color: #dbe7ff;
    font-size: 0.75rem;
    font-weight: 600;
}


/* =========================
   BUTTONS
   ========================= */

div.stButton > button {
    width: 100%;
    border-radius: 10px;
    border: 1px solid rgba(148, 163, 184, 0.12);
    background: rgba(15, 23, 42, 0.55);
    color: #b9c6dc;
    padding: 0.65rem 0.8rem;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    border-color: rgba(96, 165, 250, 0.45);
    color: #ffffff;
    background: rgba(30, 41, 59, 0.75);
}


/* =========================
   HERO
   ========================= */

.hero {
    position: relative;
    overflow: hidden;
    min-height: 380px;
    border-radius: 24px;
    margin-bottom: 28px;
    border: 1px solid rgba(148, 163, 184, 0.12);

    background-image:
        linear-gradient(
            90deg,
            rgba(5, 10, 23, 0.97) 0%,
            rgba(5, 10, 23, 0.88) 42%,
            rgba(5, 10, 23, 0.48) 100%
        ),
        url("https://images.unsplash.com/photo-1556761175-b413da4baf72?auto=format&fit=crop&w=1600&q=80");

    background-size: cover;
    background-position: center;
}

.hero-content {
    position: relative;
    z-index: 2;
    padding: 65px 60px;
    max-width: 760px;
}

.hero-kicker {
    color: #67e8f9;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 14px;
}

.hero-title {
    font-size: 3.4rem;
    line-height: 1.05;
    font-weight: 850;
    margin: 0;
    color: #ffffff;
}

.hero-highlight {
    background: linear-gradient(90deg, #67e8f9, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-description {
    margin-top: 18px;
    max-width: 650px;
    color: #b9c6dc;
    font-size: 1.03rem;
    line-height: 1.7;
}


/* =========================
   CARDS
   ========================= */

.glass-card {
    padding: 24px;
    border-radius: 18px;
    background: rgba(15, 23, 42, 0.68);
    border: 1px solid rgba(148, 163, 184, 0.10);
    box-shadow: 0 18px 45px rgba(0, 0, 0, 0.18);
}

.card-title {
    color: #f1f5ff;
    font-size: 1.05rem;
    font-weight: 750;
    margin-bottom: 7px;
}

.card-description {
    color: #8796b0;
    font-size: 0.86rem;
    line-height: 1.55;
}

.feature-icon {
    font-size: 1.5rem;
    margin-bottom: 10px;
}


/* =========================
   SECTION TITLES
   ========================= */

.section-title {
    font-size: 1.55rem;
    font-weight: 800;
    color: #f4f7ff;
    margin-bottom: 6px;
}

.section-subtitle {
    color: #8492aa;
    font-size: 0.88rem;
    margin-bottom: 20px;
}


/* =========================
   PIPELINE
   ========================= */

.pipeline-card {
    min-height: 155px;
    padding: 20px;
    border-radius: 16px;
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(148, 163, 184, 0.09);
}

.pipeline-number {
    color: #67e8f9;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 1px;
    margin-bottom: 10px;
}

.pipeline-title {
    color: #eef4ff;
    font-weight: 750;
    margin-bottom: 7px;
}

.pipeline-text {
    color: #8190aa;
    font-size: 0.8rem;
    line-height: 1.5;
}


/* =========================
   FORMS
   ========================= */

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: rgba(15, 23, 42, 0.78);
    border-color: rgba(148, 163, 184, 0.13);
}

input {
    color: #edf4ff !important;
}


/* =========================
   PREDICTION RESULT
   ========================= */

.result-card {
    padding: 30px;
    border-radius: 22px;
    background:
        radial-gradient(
            circle at 80% 20%,
            rgba(99, 102, 241, 0.16),
            transparent 35%
        ),
        rgba(15, 23, 42, 0.80);
    border: 1px solid rgba(129, 140, 248, 0.18);
    text-align: center;
    margin-top: 20px;
}

.result-label {
    color: #8998b1;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
}

.probability {
    margin: 8px 0 12px;
    font-size: 4rem;
    line-height: 1;
    font-weight: 850;
    background: linear-gradient(90deg, #67e8f9, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.risk-pill {
    display: inline-block;
    padding: 7px 17px;
    border-radius: 999px;
    background: rgba(34, 197, 94, 0.12);
    border: 1px solid rgba(34, 197, 94, 0.25);
    color: #86efac;
    font-size: 0.78rem;
    font-weight: 750;
}

.risk-pill.moderate {
    background: rgba(245, 158, 11, 0.12);
    border-color: rgba(245, 158, 11, 0.25);
    color: #fcd34d;
}

.risk-pill.high {
    background: rgba(239, 68, 68, 0.12);
    border-color: rgba(239, 68, 68, 0.25);
    color: #fca5a5;
}

.risk-note {
    margin-top: 18px;
    color: #71809b;
    font-size: 0.74rem;
    line-height: 1.55;
}


/* =========================
   SIGNALS
   ========================= */

.signal-card {
    padding: 18px;
    border-radius: 15px;
    background: rgba(15, 23, 42, 0.68);
    border: 1px solid rgba(148, 163, 184, 0.09);
    margin-bottom: 12px;
}

.signal-title {
    color: #e7efff;
    font-size: 0.88rem;
    font-weight: 750;
    margin-bottom: 8px;
}

.signal-text {
    color: #8492aa;
    font-size: 0.79rem;
    line-height: 1.55;
    margin-bottom: 5px;
}

.signal-text b {
    color: #b8c7df;
}

.explanation-card {
    padding: 18px;
    border-radius: 15px;
    background: rgba(15, 23, 42, 0.68);
    border: 1px solid rgba(103, 232, 249, 0.10);
    margin-bottom: 12px;
}

.explanation-title {
    color: #e7efff;
    font-size: 0.92rem;
    font-weight: 750;
    margin-bottom: 7px;
}

.explanation-detail {
    color: #8492aa;
    font-size: 0.79rem;
    line-height: 1.55;
}

.explanation-change {
    margin-top: 8px;
    color: #67e8f9;
    font-size: 0.80rem;
    font-weight: 700;
}

.proxy-note {
    margin-top: 10px;
    padding: 11px 13px;
    border-radius: 10px;
    background: rgba(245, 158, 11, 0.07);
    border: 1px solid rgba(245, 158, 11, 0.14);
    color: #aebbd2;
    font-size: 0.74rem;
    line-height: 1.5;
}


/* =========================
   METRICS
   ========================= */

.metric-card {
    padding: 20px;
    border-radius: 16px;
    background: rgba(15, 23, 42, 0.68);
    border: 1px solid rgba(148, 163, 184, 0.09);
}

.metric-label {
    color: #73819a;
    font-size: 0.73rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 700;
}

.metric-value {
    margin-top: 7px;
    color: #f4f7ff;
    font-size: 1.65rem;
    font-weight: 800;
}


/* =========================
   TABLE
   ========================= */

.dataframe {
    border-radius: 12px;
    overflow: hidden;
}


/* =========================
   FOOTER
   ========================= */

.footer-note {
    text-align: center;
    color: #56647d;
    font-size: 0.72rem;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid rgba(148, 163, 184, 0.07);
}

</style>
""")


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def get_risk_level(probability):
    # Keep the displayed risk band tied to the model probability.
    # A model probability at/above 50% is treated as a positive
    # attrition prediction and displayed as High Risk.
    if probability < 40:
        return "Low Risk", "low"
    elif probability < 50:
        return "Moderate Risk", "moderate"
    return "High Risk", "high"


def get_profile_signals(data):
    signals = []

    if data["StockOptionLevel"] == 0:
        signals.append(
            (
                "Stock Option Level",
                "The selected profile has no stock option level.",
                "Consider reviewing the broader rewards and recognition structure."
            )
        )

    if data["OverTime"] == "Yes":
        signals.append(
            (
                "Overtime",
                "The selected profile indicates overtime.",
                "Review workload distribution and working-hour patterns."
            )
        )

    if data["JobSatisfaction"] <= 2:
        signals.append(
            (
                "Job Satisfaction",
                "The selected profile has a relatively low job satisfaction level.",
                "Consider reviewing employee experience and engagement factors."
            )
        )

    if data["EnvironmentSatisfaction"] <= 2:
        signals.append(
            (
                "Environment Satisfaction",
                "The selected profile has a relatively low environment satisfaction level.",
                "Consider reviewing workplace experience and team environment."
            )
        )

    if data["YearsAtCompany"] <= 2:
        signals.append(
            (
                "Tenure",
                "The selected profile has relatively short tenure at the company.",
                "Consider reviewing onboarding, growth opportunities and early-career support."
            )
        )

    if data["JobLevel"] <= 1:
        signals.append(
            (
                "Job Level",
                "The selected profile is at an early job level.",
                "Consider reviewing career progression and development opportunities."
            )
        )

    if not signals:
        signals.append(
            (
                "Profile Overview",
                "No rule-based elevated signals were identified for the selected profile.",
                "Continue monitoring the broader employee profile and organizational context."
            )
        )

    return signals


def get_model_role(role):
    """Return the dataset-supported role used internally by the model."""
    return job_role_proxy.get(role, role)


def get_live_explanations(data, base_probability):
    """
    Generate profile-specific counterfactual explanations.

    Each explanation changes one input at a time, keeps the other inputs
    unchanged, and measures how the Random Forest probability changes.
    This describes model sensitivity for the current profile; it does not
    establish causation.
    """
    candidates = []

    def add_candidate(label, counterfactual_data, change_text):
        counter_input = prepare_input(counterfactual_data)
        counter_probability = model.predict_proba(counter_input)[0][1] * 100
        delta = counter_probability - base_probability
        candidates.append({
            "label": label,
            "current": base_probability,
            "counter": counter_probability,
            "delta": delta,
            "change": change_text
        })

    # Categorical / binary features
    if data["OverTime"] == "Yes":
        changed = data.copy()
        changed["OverTime"] = "No"
        add_candidate(
            "Overtime",
            changed,
            "If OverTime changes from Yes → No"
        )
    else:
        changed = data.copy()
        changed["OverTime"] = "Yes"
        add_candidate(
            "Overtime",
            changed,
            "If OverTime changes from No → Yes"
        )

    if data["Gender"] == "Male":
        changed = data.copy()
        changed["Gender"] = "Female"
        add_candidate("Gender", changed, "If Gender changes from Male → Female")
    else:
        changed = data.copy()
        changed["Gender"] = "Male"
        add_candidate("Gender", changed, "If Gender changes from Female → Male")

    if data["StockOptionLevel"] == 0:
        changed = data.copy()
        changed["StockOptionLevel"] = 1
        add_candidate(
            "Stock Option Level",
            changed,
            "If Stock Option Level changes from 0 → 1"
        )
    else:
        changed = data.copy()
        changed["StockOptionLevel"] = 0
        add_candidate(
            "Stock Option Level",
            changed,
            f"If Stock Option Level changes from {data['StockOptionLevel']} → 0"
        )

    # Satisfaction / engagement features: compare the current value with
    # the opposite end of the available 1–4 scale.
    for key, label in [
        ("JobSatisfaction", "Job Satisfaction"),
        ("EnvironmentSatisfaction", "Environment Satisfaction"),
        ("RelationshipSatisfaction", "Relationship Satisfaction"),
        ("JobInvolvement", "Job Involvement"),
        ("WorkLifeBalance", "Work Life Balance"),
        ("PerformanceRating", "Performance Rating"),
    ]:
        current = int(data[key])
        target = 4 if current <= 2 else 1
        if current != target:
            changed = data.copy()
            changed[key] = target
            add_candidate(
                label,
                changed,
                f"If {label} changes from {current} → {target}"
            )

    # Career / numerical features. Use a modest one-step counterfactual
    # rather than inventing a large change in the employee profile.
    numeric_changes = [
        ("Age", 5, 18, 70),
        ("DistanceFromHome", 5, 1, 30),
        ("JobLevel", 1, 1, 5),
        ("Education", 1, 1, 5),
        ("MonthlyIncome", 0.25, 1000, 20000),
        ("DailyRate", 0.25, 100, 1500),
        ("HourlyRate", 0.25, 20, 150),
        ("MonthlyRate", 0.25, 2000, 27000),
        ("TotalWorkingYears", 3, 0, 40),
        ("YearsAtCompany", 3, 0, 40),
        ("YearsInCurrentRole", 2, 0, 20),
        ("YearsSinceLastPromotion", 2, 0, 15),
        ("YearsWithCurrManager", 2, 0, 20),
        ("NumCompaniesWorked", 1, 0, 10),
        ("TrainingTimesLastYear", 1, 0, 10),
        ("PercentSalaryHike", 3, 10, 30),
    ]

    for key, step, minimum, maximum in numeric_changes:
        current = float(data[key])

        if key in {"MonthlyIncome", "DailyRate", "HourlyRate", "MonthlyRate"}:
            target = current * (1.0 + step) if current <= (minimum + maximum) / 2 else current * (1.0 - step)
        elif key == "Age":
            target = current + step if current < 40 else current - step
        else:
            target = current + step if current < (minimum + maximum) / 2 else current - step

        target = max(minimum, min(maximum, target))
        target = int(round(target))

        if int(current) == target:
            continue

        # Keep obvious tenure relationships internally consistent.
        changed = data.copy()
        changed[key] = target

        if key == "TotalWorkingYears" and target < changed["YearsAtCompany"]:
            changed["YearsAtCompany"] = target
        if key == "YearsAtCompany":
            changed["TotalWorkingYears"] = max(changed["TotalWorkingYears"], target)
            changed["YearsInCurrentRole"] = min(changed["YearsInCurrentRole"], target)
            changed["YearsWithCurrManager"] = min(changed["YearsWithCurrManager"], target)
        if key == "YearsInCurrentRole":
            changed["YearsInCurrentRole"] = min(target, changed["YearsAtCompany"])
        if key == "YearsWithCurrManager":
            changed["YearsWithCurrManager"] = min(target, changed["YearsAtCompany"])

        add_candidate(
            key,
            changed,
            f"If {key} changes from {int(current)} → {target}"
        )

    # Existing dataset-supported job roles can also be tested. For modern
    # roles, the displayed role is mapped to the configured dataset proxy.
    current_model_role = get_model_role(data["JobRole"])
    role_alternatives = [
        "Research Scientist",
        "Laboratory Technician",
        "Healthcare Representative",
        "Sales Executive",
        "Manager",
    ]
    for alternate_role in role_alternatives:
        if alternate_role == current_model_role:
            continue
        changed = data.copy()
        changed["JobRole"] = alternate_role
        add_candidate(
            "Job Role",
            changed,
            f"If the model-supported job role changes from {current_model_role} → {alternate_role}"
        )

    # Sort by absolute change in model probability and retain the strongest
    # current-profile signals.
    candidates.sort(key=lambda item: abs(item["delta"]), reverse=True)
    return candidates[:8]


def prepare_input(data):
    model_data = data.copy()
    model_data["JobRole"] = get_model_role(model_data["JobRole"])
    input_df = pd.DataFrame([model_data])

    categorical_cols = [
        "BusinessTravel",
        "Department",
        "EducationField",
        "Gender",
        "JobRole",
        "MaritalStatus",
        "OverTime"
    ]

    input_df = pd.get_dummies(
        input_df,
        columns=categorical_cols,
        drop_first=True
    )

    input_df = input_df.reindex(
        columns=model_columns,
        fill_value=0
    )

    return input_df


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    render_html("""
    <div class="brand">
        <div class="brand-title">ATTRITIONAI</div>
        <div class="brand-subtitle">HR Risk Analytics Platform</div>
    </div>
    """)

    render_html("""
    <div class="sidebar-section">Navigation</div>
    """)

    if st.button(
        "⌂  Employee Attrition Prediction",
        use_container_width=True
    ):
        st.session_state.page = "Employee Attrition Prediction"
        st.rerun()

    if st.button(
        "◈  ML Pipeline",
        use_container_width=True
    ):
        st.session_state.page = "ML Pipeline"
        st.rerun()

    if st.button(
        "◉  Predictor",
        use_container_width=True
    ):
        st.session_state.page = "Predictor"
        st.rerun()

    if st.button(
        "▦  Dashboard",
        use_container_width=True
    ):
        st.session_state.page = "Dashboard"
        st.rerun()

    render_html("""
    <div class="sidebar-section">System</div>
    """)

    render_html("""
    <div class="status-box">

        <div class="status-header">
            <span class="status-dot"></span>
            <span>Model Ready</span>
        </div>

        <div class="status-row">
            <span class="status-label">Algorithm</span>
            <span class="status-value">Random Forest</span>
        </div>

        <div class="status-row">
            <span class="status-label">Features</span>
            <span class="status-value">44</span>
        </div>

        <div class="status-row">
            <span class="status-label">Dataset</span>
            <span class="status-value">1,470 records</span>
        </div>

    </div>
    """)

    render_html("<br>")

    if st.button(
        "↻  Reset Session",
        use_container_width=True
    ):
        st.session_state.history = []
        st.session_state.last_result = None
        st.session_state.page = "Employee Attrition Prediction"
        st.rerun()


# =========================================================
# HOME
# =========================================================
if st.session_state.page == "Employee Attrition Prediction":

    render_html("""
    <div class="hero">

        <div class="hero-content">

            <div class="hero-kicker">
                Machine Learning • HR Analytics
            </div>

            <div class="hero-title">
                Employee Attrition
                <br>
                <span class="hero-highlight">Prediction</span>
            </div>

            <div class="hero-description">
                A machine-learning based HR analytics platform that
                estimates employee attrition probability from selected
                demographic, career and workplace profile attributes.
            </div>

        </div>

    </div>
    """)

    render_html("""
    <div class="section-title">
        Employee Attrition Prediction
    </div>

    <div class="section-subtitle">
        Analyze employee profiles using a trained Random Forest classification model.
    </div>
    """)

    c1, c2, c3 = st.columns(3)

    with c1:
        render_html("""
        <div class="glass-card">
            <div class="feature-icon">📊</div>
            <div class="card-title">Attrition Prediction</div>
            <div class="card-description">
                Generate an estimated attrition probability from an employee profile.
            </div>
        </div>
        """)

    with c2:
        render_html("""
        <div class="glass-card">
            <div class="feature-icon">🔎</div>
            <div class="card-title">Key Profile Risk Signals</div>
            <div class="card-description">
                Review selected profile characteristics associated with the prediction.
            </div>
        </div>
        """)

    with c3:
        render_html("""
        <div class="glass-card">
            <div class="feature-icon">📈</div>
            <div class="card-title">HR Insights</div>
            <div class="card-description">
                Explore prediction history and profile-level analytics during the session.
            </div>
        </div>
        """)

    render_html("<br>")

    render_html("""
    <div class="section-title">
        Project Architecture
    </div>

    <div class="section-subtitle">
        From employee profile data to a machine-learning prediction.
    </div>
    """)

    p1, p2, p3, p4 = st.columns(4)

    pipeline_data = [
        (
            p1,
            "01",
            "Employee Data",
            "Collect demographic, work, career and workplace attributes."
        ),
        (
            p2,
            "02",
            "Preprocessing",
            "Encode categorical variables and align them with the trained feature set."
        ),
        (
            p3,
            "03",
            "Random Forest",
            "Apply the trained classification model to the selected employee profile."
        ),
        (
            p4,
            "04",
            "Prediction",
            "Return an estimated attrition probability and interpretation band."
        ),
    ]

    for col, number, title, description in pipeline_data:
        with col:
            render_html(f"""
            <div class="pipeline-card">
                <div class="pipeline-number">{number}</div>
                <div class="pipeline-title">{title}</div>
                <div class="pipeline-text">{description}</div>
            </div>
            """)

    render_html("<br>")

    render_html("""
    <div class="section-title">
        Model Performance
    </div>

    <div class="section-subtitle">
        Evaluation results on the held-out test set.
    </div>
    """)

    performance_df = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest"
            ],
            "Accuracy": [
                "74.49%",
                "76.53%",
                "80.61%"
            ],
            "Precision": [
                "34.09%",
                "34.72%",
                "43.06%"
            ],
            "Recall": [
                "63.83%",
                "53.19%",
                "65.96%"
            ],
            "F1 Score": [
                "44.44%",
                "42.02%",
                "52.10%"
            ]
        }
    )

    st.dataframe(
        performance_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# ML PIPELINE
# =========================================================
elif st.session_state.page == "ML Pipeline":

    render_html("""
    <div class="section-title">
        ML Pipeline
    </div>

    <div class="section-subtitle">
        Overview of the machine-learning workflow used by the application.
    </div>
    """)

    steps = [
        (
            "01",
            "Dataset",
            "IBM HR Employee Attrition dataset containing 1,470 employee records and 35 original attributes."
        ),
        (
            "02",
            "Data Cleaning",
            "Non-predictive fields such as employee identifiers and constant columns are removed."
        ),
        (
            "03",
            "Feature Engineering",
            "Categorical variables are converted into machine-learning compatible encoded features."
        ),
        (
            "04",
            "Train / Test Split",
            "Data is divided into training and testing subsets using a stratified 80/20 split."
        ),
        (
            "05",
            "Model Training",
            "Logistic Regression, Decision Tree and Random Forest models are trained and compared."
        ),
        (
            "06",
            "Deployment",
            "The trained Random Forest model is used by the Streamlit prediction interface."
        )
    ]

    for i in range(0, len(steps), 3):

        cols = st.columns(3)

        for j, col in enumerate(cols):

            if i + j >= len(steps):
                continue

            number, title, description = steps[i + j]

            with col:
                render_html(f"""
                <div class="pipeline-card">
                    <div class="pipeline-number">{number}</div>
                    <div class="pipeline-title">{title}</div>
                    <div class="pipeline-text">{description}</div>
                </div>
                """)

        render_html("<br>")

    render_html("""
    <div class="section-title">
        Model Comparison
    </div>
    """)

    comparison_df = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest"
            ],
            "Accuracy": [
                "74.49%",
                "76.53%",
                "80.61%"
            ],
            "Precision": [
                "34.09%",
                "34.72%",
                "43.06%"
            ],
            "Recall": [
                "63.83%",
                "53.19%",
                "65.96%"
            ],
            "F1 Score": [
                "44.44%",
                "42.02%",
                "52.10%"
            ]
        }
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# PREDICTOR
# =========================================================
elif st.session_state.page == "Predictor":

    render_html("""
    <div class="section-title">
        Employee Attrition Predictor
    </div>

    <div class="section-subtitle">
        Enter an employee profile to generate a model-based attrition estimate.
    </div>
    """)

    render_html("""
    <div class="glass-card">
        <div class="card-title">Employee Profile</div>
        <div class="card-description">
            Demographic and organizational information.
        </div>
    </div>
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=70,
            value=30
        )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        marital_status = st.selectbox(
            "Marital Status",
            ["Divorced", "Married", "Single"]
        )

        distance_from_home = st.number_input(
            "Distance From Home",
            min_value=1,
            max_value=30,
            value=5
        )

    with col2:
        department = st.selectbox(
            "Department",
            [
                "Human Resources",
                "Research & Development",
                "Sales"
            ]
        )

        job_role = st.selectbox(
            "Job Role",
            [
                "AI/ML Engineer",
                "Software Developer",
                "Healthcare Representative",
                "Human Resources",
                "Laboratory Technician",
                "Manager",
                "Manufacturing Director",
                "Research Director",
                "Research Scientist",
                "Sales Executive",
                "Sales Representative"
            ]
        )

        job_level = st.number_input(
            "Job Level",
            min_value=1,
            max_value=5,
            value=2
        )

        education = st.number_input(
            "Education Level",
            min_value=1,
            max_value=5,
            value=3
        )

    with col3:
        education_field = st.selectbox(
            "Education Field",
            [
                "Human Resources",
                "Life Sciences",
                "Marketing",
                "Medical",
                "Other",
                "Technical Degree"
            ]
        )

        business_travel = st.selectbox(
            "Business Travel",
            [
                "Non-Travel",
                "Travel_Rarely",
                "Travel_Frequently"
            ]
        )

        overtime = st.selectbox(
            "OverTime",
            ["No", "Yes"]
        )

    render_html("<br>")

    render_html("""
    <div class="glass-card">
        <div class="card-title">Work & Career</div>
        <div class="card-description">
            Compensation, experience and career progression attributes.
        </div>
    </div>
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        monthly_income = st.number_input(
            "Monthly Income",
            min_value=1000,
            max_value=20000,
            value=5000
        )

        daily_rate = st.number_input(
            "Daily Rate",
            min_value=100,
            max_value=1500,
            value=700
        )

        hourly_rate = st.number_input(
            "Hourly Rate",
            min_value=20,
            max_value=150,
            value=65
        )

    with col2:
        total_working_years = st.number_input(
            "Total Working Years",
            min_value=0,
            max_value=40,
            value=8
        )

        years_at_company = st.number_input(
            "Years At Company",
            min_value=0,
            max_value=40,
            value=4
        )

        years_in_current_role = st.number_input(
            "Years In Current Role",
            min_value=0,
            max_value=20,
            value=2
        )

    with col3:
        years_since_last_promotion = st.number_input(
            "Years Since Last Promotion",
            min_value=0,
            max_value=15,
            value=1
        )

        years_with_curr_manager = st.number_input(
            "Years With Current Manager",
            min_value=0,
            max_value=20,
            value=2
        )

        num_companies_worked = st.number_input(
            "Number Of Companies Worked",
            min_value=0,
            max_value=10,
            value=2
        )

    render_html("<br>")

    render_html("""
    <div class="glass-card">
        <div class="card-title">Workplace Experience</div>
        <div class="card-description">
            Satisfaction and engagement-related attributes.
        </div>
    </div>
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        job_satisfaction = st.slider(
            "Job Satisfaction",
            1,
            4,
            3
        )

        environment_satisfaction = st.slider(
            "Environment Satisfaction",
            1,
            4,
            3
        )

        relationship_satisfaction = st.slider(
            "Relationship Satisfaction",
            1,
            4,
            3
        )

    with col2:
        job_involvement = st.slider(
            "Job Involvement",
            1,
            4,
            3
        )

        work_life_balance = st.slider(
            "Work Life Balance",
            1,
            4,
            3
        )

        performance_rating = st.slider(
            "Performance Rating",
            1,
            4,
            3
        )

    with col3:
        stock_option_level = st.slider(
            "Stock Option Level",
            0,
            3,
            1
        )

        training_times = st.number_input(
            "Training Times Last Year",
            min_value=0,
            max_value=10,
            value=3
        )

        percent_salary_hike = st.number_input(
            "Percent Salary Hike",
            min_value=10,
            max_value=30,
            value=14
        )

    render_html("<br>")

    render_html("""
    <div class="glass-card">
        <div class="card-title">Advanced Model Features</div>
        <div class="card-description">
            Additional numerical attributes used by the trained model.
        </div>
    </div>
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        monthly_rate = st.number_input(
            "Monthly Rate",
            min_value=2000,
            max_value=27000,
            value=14000
        )

    with col2:
        pass

    with col3:
        pass

    predict_button = st.button(
        "Generate Attrition Prediction",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        employee_data = {
            "Age": age,
            "DailyRate": daily_rate,
            "DistanceFromHome": distance_from_home,
            "Education": education,
            "EnvironmentSatisfaction": environment_satisfaction,
            "HourlyRate": hourly_rate,
            "JobInvolvement": job_involvement,
            "JobLevel": job_level,
            "JobSatisfaction": job_satisfaction,
            "MonthlyIncome": monthly_income,
            "MonthlyRate": monthly_rate,
            "NumCompaniesWorked": num_companies_worked,
            "PercentSalaryHike": percent_salary_hike,
            "PerformanceRating": performance_rating,
            "RelationshipSatisfaction": relationship_satisfaction,
            "StockOptionLevel": stock_option_level,
            "TotalWorkingYears": total_working_years,
            "TrainingTimesLastYear": training_times,
            "WorkLifeBalance": work_life_balance,
            "YearsAtCompany": years_at_company,
            "YearsInCurrentRole": years_in_current_role,
            "YearsSinceLastPromotion": years_since_last_promotion,
            "YearsWithCurrManager": years_with_curr_manager,
            "BusinessTravel": business_travel,
            "Department": department,
            "EducationField": education_field,
            "Gender": gender,
            "JobRole": job_role,
            "MaritalStatus": marital_status,
            "OverTime": overtime
        }

        input_df = prepare_input(employee_data)

        probability = model.predict_proba(input_df)[0][1] * 100

        prediction = int(probability >= 50)

        risk_text, risk_class = get_risk_level(probability)

        result = {
            "probability": probability,
            "prediction": prediction,
            "risk": risk_text,
            "data": employee_data
        }

        st.session_state.last_result = result
        st.session_state.history.append(result)

    # =====================================================
    # RESULT
    # =====================================================
    if st.session_state.last_result is not None:

        result = st.session_state.last_result

        probability = result["probability"]
        risk_text = result["risk"]
        employee_data = result["data"]

        _, risk_class = get_risk_level(probability)

        # IMPORTANT:
        # Entire result card is rendered in ONE HTML block.
        # This prevents raw HTML from appearing on the page.
        render_html(f"""
        <div class="result-card">

            <div class="result-label">
                Estimated Attrition Probability
            </div>

            <div class="probability">
                {probability:.1f}%
            </div>

            <div class="risk-pill {risk_class}">
                {risk_text}
            </div>

            <div class="risk-note">
                Risk interpretation: Low &lt; 40% ·
                Moderate 40–49.9% ·
                High ≥ 50%.
                These thresholds are application-defined interpretation
                bands and are not calibrated organizational risk categories.
            </div>

        </div>
        """)

        render_html("<br>")

        render_html("""
        <div class="section-title">
            Factors Influencing This Model Prediction
        </div>

        <div class="section-subtitle">
            Live counterfactual analysis of the current employee profile.
            Each signal changes one input while keeping the other inputs unchanged.
        </div>
        """)

        live_explanations = get_live_explanations(employee_data, probability)

        for item in live_explanations:
            direction = "higher" if item["delta"] > 0 else "lower"
            magnitude = abs(item["delta"])

            render_html(f"""
            <div class="explanation-card">
                <div class="explanation-title">
                    {item["label"]}
                </div>

                <div class="explanation-detail">
                    {item["change"]}. The model estimates a {direction}
                    attrition probability under that one-change scenario.
                </div>

                <div class="explanation-change">
                    {item["current"]:.1f}% → {item["counter"]:.1f}%
                    &nbsp; ({"+" if item["delta"] > 0 else "-"}{magnitude:.1f} percentage points)
                </div>
            </div>
            """)

        if employee_data["JobRole"] in job_role_proxy:
            proxy_role = get_model_role(employee_data["JobRole"])
            render_html(f"""
            <div class="proxy-note">
                <b>Job-role note:</b> {employee_data["JobRole"]} is not a category in the
                original IBM HR training data, so the current model evaluates it using
                the configured proxy role <b>{proxy_role}</b>. This is an application
                mapping, not a role-specific model learned from AI/ML Engineer or
                Software Developer records.
            </div>
            """)

        render_html("""
        <div class="risk-note">
            These signals describe how the trained Random Forest responds to
            the supplied profile. They are model-sensitivity indicators, not
            causal explanations or predictions of individual employee behavior.
        </div>
        """)


# =========================================================
# DASHBOARD
# =========================================================
elif st.session_state.page == "Dashboard":

    render_html("""
    <div class="section-title">
        Prediction Dashboard
    </div>

    <div class="section-subtitle">
        Session-level analytics for profiles analyzed through this application.
    </div>
    """)

    history = st.session_state.history

    if len(history) == 0:

        render_html("""
        <div class="glass-card">
            <div class="card-title">
                No predictions yet
            </div>

            <div class="card-description">
                Go to the Predictor page and generate a prediction to populate
                this dashboard.
            </div>
        </div>
        """)

    else:

        probabilities = [
            item["probability"]
            for item in history
        ]

        high_risk_count = sum(
            item["risk"] == "High Risk"
            for item in history
        )

        avg_probability = np.mean(probabilities)

        m1, m2, m3, m4 = st.columns(4)

        metrics = [
            ("Profiles Analyzed", len(history)),
            (
                "Predicted Attrition",
                sum(item["prediction"] for item in history)
            ),
            ("High Risk Profiles", high_risk_count),
            ("Average Probability", f"{avg_probability:.1f}%")
        ]

        for col, (label, value) in zip(
            [m1, m2, m3, m4],
            metrics
        ):

            with col:

                render_html(f"""
                <div class="metric-card">

                    <div class="metric-label">
                        {label}
                    </div>

                    <div class="metric-value">
                        {value}
                    </div>

                </div>
                """)

        render_html("<br>")

        chart_df = pd.DataFrame(
            {
                "Estimated Attrition Probability": probabilities
            }
        )

        fig = px.histogram(
            chart_df,
            x="Estimated Attrition Probability",
            nbins=10,
            title="Prediction Probability Distribution"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#dbe7ff")
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        table_rows = []

        for i, item in enumerate(history, start=1):

            table_rows.append(
                {
                    "Profile": f"Profile {i}",
                    "Probability": f"{item['probability']:.1f}%",
                    "Risk": item["risk"],
                    "Prediction": (
                        "Attrition Predicted"
                        if item["prediction"] == 1
                        else "No Attrition Predicted"
                    )
                }
            )

        history_df = pd.DataFrame(table_rows)

        render_html("""
        <div class="section-title">
            Analyzed Profiles
        </div>
        """)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Prediction History",
            data=csv,
            file_name="attrition_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        render_html("""
        <div class="risk-note">
            Dashboard values represent predictions generated during the
            current application session. They are not organizational
            workforce statistics or measured company-wide attrition rates.
        </div>
        """)


# =========================================================
# FOOTER
# =========================================================
render_html("""
<div class="footer-note">
    Employee Attrition Prediction • Random Forest • 44 Model Features
</div>
""")