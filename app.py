import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

# =========================
# SESSION STATE
# =========================
if "page" not in st.session_state:
    st.session_state.page = 0

if "history" not in st.session_state:
    st.session_state.history = []

def go_to(page):
    st.session_state.page = page

# =========================
# LOAD MODEL
# =========================
def load_model(path):
    try:
        return joblib.load(path)
    except:
        return None

rf_model = load_model("rf.pkl")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "⚙️ Pipeline", "🤖 Predictor", "📊 Dashboard"],
    index=st.session_state.page
)
st.session_state.page = ["🏠 Home", "⚙️ Pipeline", "🤖 Predictor", "📊 Dashboard"].index(page)

# =========================
# CSS
# =========================
st.markdown("""
<style>
body {background-color:#0E1117;color:white;}
.neon-title {
    font-size:48px;font-weight:900;
    background: linear-gradient(90deg,#00f5ff,#9d4edd);
    -webkit-background-clip:text;color:transparent;
    text-shadow:0 0 20px #00f5ff;
}
.card {
    background: rgba(255,255,255,0.05);
    padding:20px;border-radius:15px;
    backdrop-filter: blur(6px);
    margin:10px 0;
}
.flow {
    border:1px solid #00f5ff;
    padding:15px;margin:10px 0;
    border-radius:10px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# =========================
# PAGE 1 HOME
# =========================
if st.session_state.page == 0:
    col1, col2 = st.columns([2,1])

    with col1:
        st.markdown("## ML PROJECT")
        st.markdown('<div class="neon-title">Employee Attrition Prediction</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">AI-powered system to predict employee churn.</div>', unsafe_allow_html=True)

    with col2:
        st.image("https://images.unsplash.com/photo-1551434678-e076c223a692", width=300)

    st.button("🚀 Get Started", on_click=go_to, args=(1,))
    


# ==============================================================================
# PAGE 2 PIPELINE (RESTORED SAME & ENHANCED VISUALS)
# ==============================================================================
elif st.session_state.page == 1:
    st.markdown('<p class="shiny-subheader" style="font-size: 2.2rem !important; font-weight: 700; background: linear-gradient(45deg, #00F2FE, #4FACFE, #9B51E0); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">📊 Machine Learning Pipeline Blueprint</p>', unsafe_allow_html=True)
    
    # 2 Column layout to add the aesthetic technical image on the side cleanly
    pipe_col1, pipe_col2 = st.columns([1.3, 0.7], gap="large")
    
    with pipe_col1:
        st.markdown("<p style='color: #CBD5E1; font-size: 1.05rem; font-weight: 600; margin-bottom: 20px;'>A structured step-by-step sequential breakdown mapping of how raw data travels from collection to final core diagnostic models.</p>", unsafe_allow_html=True)
        
        # --- FLOWCHART NODES WITH STRICK INLINE CARD STYLINGS ---
        st.markdown("""
        <div style="background: rgba(22, 27, 34, 0.85); border: 2px solid #00F2FE; border-left: 7px solid #00F2FE; border-radius: 12px; padding: 20px; margin-bottom: 10px; box-shadow: 0 4px 20px rgba(0, 242, 254, 0.15);">
            <div style="color: #00F2FE; font-weight: 800; font-size: 1.3rem; margin-bottom: 8px;">🧼 1. Data Preprocessing & Cleaning</div>
            <p style="margin:0; color:#FFFFFF; font-weight: 600; font-size:0.98rem; line-height:1.6;">
                Handled missing record cells, detected statistical outlier boundaries, and normalized data distributions to feed clean arrays into estimators.
            </p>
        </div>
        
        <div style="text-align:center; color:#00F2FE; font-size:1.8rem; margin: 5px 0; font-weight:900; text-shadow: 0 0 10px #00F2FE;">▼</div>
        
        <div style="background: rgba(22, 27, 34, 0.85); border: 2px solid #4FACFE; border-left: 7px solid #4FACFE; border-radius: 12px; padding: 20px; margin-bottom: 10px; box-shadow: 0 4px 20px rgba(79, 172, 254, 0.15);">
            <div style="color: #4FACFE; font-weight: 800; font-size: 1.3rem; margin-bottom: 8px;">🔑 2. Categorical Label Conversion & Encoding</div>
            <p style="margin:0; color:#FFFFFF; font-weight: 600; font-size:0.98rem; line-height:1.6;">
                Safely mapped descriptive values (like Job Roles or Overtime strings) into operational numerical dimensions using standard Label Encoders.
            </p>
        </div>
        
        <div style="text-align:center; color:#4FACFE; font-size:1.8rem; margin: 5px 0; font-weight:900; text-shadow: 0 0 10px #4FACFE;">▼</div>
        
        <div style="background: rgba(22, 27, 34, 0.85); border: 2px solid #9B51E0; border-left: 7px solid #9B51E0; border-radius: 12px; padding: 20px; margin-bottom: 10px; box-shadow: 0 4px 20px rgba(155, 81, 224, 0.15);">
            <div style="color: #9B51E0; font-weight: 800; font-size: 1.3rem; margin-bottom: 8px;">⚙️ 3. Feature Selection Engine (44-Matrix Structure)</div>
            <p style="margin:0; color:#FFFFFF; font-weight: 600; font-size:0.98rem; line-height:1.6;">
                Isolated high-variance components to establish the complete 44-feature training vector shape, cutting down baseline variance limitations.
            </p>
        </div>
        
        <div style="text-align:center; color:#9B51E0; font-size:1.8rem; margin: 5px 0; font-weight:900; text-shadow: 0 0 10px #9B51E0;">▼</div>
        
        <div style="background: rgba(22, 27, 34, 0.85); border: 2px solid #FF0055; border-left: 7px solid #FF0055; border-radius: 12px; padding: 20px; margin-bottom: 10px; box-shadow: 0 4px 20px rgba(255, 0, 85, 0.15);">
            <div style="color: #FF0055; font-weight: 800; font-size: 1.3rem; margin-bottom: 8px;">🧪 4. Train-Test Matrix Split Partitioning</div>
            <p style="margin:0; color:#FFFFFF; font-weight: 600; font-size:0.98rem; line-height:1.6;">
                Divided input samples on a precise 80/20 balance layout configuration, securing an independent hold-out dataset to rigorously target overfitting bugs.
            </p>
        </div>
        
        <div style="text-align:center; color:#FF0055; font-size:1.8rem; margin: 5px 0; font-weight:900; text-shadow: 0 0 10px #FF0055;">▼</div>
        
        <div style="background: rgba(22, 27, 34, 0.85); border: 2px solid #00FF87; border-left: 7px solid #00FF87; border-radius: 12px; padding: 20px; margin-bottom: 10px; box-shadow: 0 4px 20px rgba(0, 255, 135, 0.15);">
            <div style="color: #00FF87; font-weight: 800; font-size: 1.3rem; margin-bottom: 8px;">🤖 5. Multi-Model Array Training & Benchmarking</div>
            <p style="margin:0; color:#FFFFFF; font-weight: 600; font-size:0.98rem; line-height:1.6;">
                Exposed arrays simultaneously to Logistic Regression, Decision Tree, and Random Forest classifiers, computing full precision and recall metrics.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with pipe_col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        # Fallback raw SVG vector diagram block for 100% successful rendering in Streamlit local systems
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.5); border: 1px dashed #00F2FE; border-radius: 12px; padding: 15px; text-align: center;">
            <svg viewBox="0 0 100 100" width="100%" height="180" style="margin-bottom: 10px;">
                <circle cx="50" cy="20" r="12" fill="none" stroke="#00F2FE" stroke-width="2"/>
                <path d="M50 32 L50 48" stroke="#4FACFE" stroke-width="2" stroke-dasharray="3,3"/>
                <rect x="35" y="48" width="30" height="15" rx="3" fill="none" stroke="#9B51E0" stroke-width="2"/>
                <path d="M50 63 L50 78" stroke="#FF0055" stroke-width="2" stroke-dasharray="3,3"/>
                <polygon points="50,78 45,73 55,73" fill="#00FF87"/>
            </svg>
            <p style="color: #00F2FE; font-weight: bold; margin: 0; font-size: 0.95rem;">Pipeline Blueprint Stack</p>
            <p style="color: #94A3B8; font-size: 0.8rem; margin-top: 5px;">Automated Data Matrix Flow</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br><hr style='border-color:rgba(0, 242, 254, 0.1);'/>", unsafe_allow_html=True)
    
    # --- NAVIGATION BUTTON CONTROL LAYOUT ---
    col1, col2 = st.columns(2)
    col1.button("← Back", on_click=go_to, args=(0,), use_container_width=True)
    col2.button("Next →", on_click=go_to, args=(2,), type="primary", use_container_width=True)

# =========================
# PAGE 3 PREDICTOR
# =========================
elif st.session_state.page == 2:

    st.title("🤖 Attrition Predictor")

    with st.form("prediction_form"):

        job_role = st.selectbox("Job Role",
            ["Sales Executive","Research Scientist","Software Engineer","Human Resources"])

        overtime = st.selectbox("Overtime", ["Yes","No"])
        income = st.slider("Monthly Income", 1000, 20000, 5000)
        satisfaction = st.slider("Job Satisfaction",1,4,2)
        age = st.slider("Age",18,60,30)
        years = st.slider("Total Working Years",0,40,5)

        submit = st.form_submit_button("🔍 Predict")

    if submit:

        def prepare_input(model):
            cols = model.feature_names_in_
            df = pd.DataFrame(np.zeros((1,len(cols))), columns=cols)

            if "Age" in cols: df["Age"] = age
            if "MonthlyIncome" in cols: df["MonthlyIncome"] = income
            if "JobSatisfaction" in cols: df["JobSatisfaction"] = satisfaction
            if "TotalWorkingYears" in cols: df["TotalWorkingYears"] = years
            if "OverTime" in cols: df["OverTime"] = 1 if overtime=="Yes" else 0

            for c in cols:
                if "JobRole" in c and job_role in c:
                    df[c] = 1

            return df

        try:
            data = prepare_input(rf_model)
            pred = rf_model.predict(data)[0]
            prob = rf_model.predict_proba(data)[0][1]

            st.markdown(f"""
            <div class="card">
            <h2 style="color:{'red' if pred else 'lime'}">
            {"⚠️ Employee Likely to Leave" if pred else "✅ Employee Likely to Stay"}
            </h2>
            Confidence: {round(prob*100,2)}%
            </div>
            """, unsafe_allow_html=True)

            st.session_state.history.append({
                "Age": age,
                "Income": income,
                "Satisfaction": satisfaction,
                "Prediction": "Leave" if pred else "Stay"
            })

        except Exception as e:
            st.error(f"Prediction Error: {e}")

    col1,col2 = st.columns(2)
    col1.button("← Back", on_click=go_to, args=(1,))
    col2.button("Dashboard →", on_click=go_to, args=(3,))

# =========================
# PAGE 4 DASHBOARD (SMART INSIGHTS)
# =========================
elif st.session_state.page == 3:

    st.title("📊 Live HR Dashboard")

    history = pd.DataFrame(st.session_state.history)

    if history.empty:
        st.warning("No predictions yet. Go to Predictor.")
    else:

        col1, col2 = st.columns(2)
        total = len(history)
        leave = (history["Prediction"]=="Leave").sum()

        col1.metric("Total Predictions", total)
        col2.metric("Attrition Count", leave)

        st.markdown("---")

        # Chart 1
        st.subheader("Prediction Distribution")
        dist = history["Prediction"].value_counts()
        st.bar_chart(dist)

        # Chart 2
        st.subheader("Income vs Age")
        st.scatter_chart(history, x="Income", y="Age")

        # Chart 3
        st.subheader("Satisfaction Distribution")
        sat = history["Satisfaction"].value_counts()
        st.bar_chart(sat)

        # =====================
        # SMART GRAPH INSIGHTS
        # =====================
        st.markdown("### 📌 Insights")

        leave_pct = (leave/total)*100

        # Insight 1
        st.write(f"🔹 Attrition Rate: **{round(leave_pct,2)}%**")
        if leave_pct > 50:
            st.error("High attrition observed in predictions")
        else:
            st.success("Attrition under control")

        # Insight 2 (Satisfaction)
        low_sat = history[history["Satisfaction"] <= 2]
        if not low_sat.empty:
            risk = (low_sat["Prediction"]=="Leave").mean()*100
            st.write(f"🔹 Low satisfaction employees leaving: **{round(risk,2)}%**")

        # Insight 3 (Income)
        low_income = history[history["Income"] < history["Income"].median()]
        if not low_income.empty:
            risk2 = (low_income["Prediction"]=="Leave").mean()*100
            st.write(f"🔹 Lower income attrition risk: **{round(risk2,2)}%**")

        # Insight 4 (Age)
        young = history[history["Age"] < 30]
        if not young.empty:
            risk3 = (young["Prediction"]=="Leave").mean()*100
            st.write(f"🔹 Younger employees attrition: **{round(risk3,2)}%**")

    st.button("← Back", on_click=go_to, args=(2,))