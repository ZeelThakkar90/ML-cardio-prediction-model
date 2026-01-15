import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
bundle = joblib.load("cardio_model.pkl")
model = bundle['model']
feature_order = bundle['columns']
scaler_hi = bundle['scaler_hi']
scaler_lo = bundle['scaler_lo']

st.markdown("""
    <style>
    /* Vitality Green Background */
    .stApp {
        background: #f1f8e9; /* Very light mint green */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Global Text Reset to Black */
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #000000 !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: left;
        margin-bottom: 0.1rem;
        letter-spacing: -0.5px;
        color: #000000 !important;
    }
    
    .subtitle {
        text-align: left;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 500;
        color: #000000 !important;
    }
    
    /* --- FORM STYLING --- */
    
    /* Form container */
    .stForm {
        background: #ffffff;
        border-radius: 12px;
        padding: 2.5rem;
        border: 1px solid #dcedc8;
        box-shadow: 0 10px 25px rgba(27, 94, 32, 0.05);
    }

    /* Widget Labels (Input titles) */
    .stSelectbox label, .stNumberInput label, .stRadio label, .stCheckbox label {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Number Input Fields */
    .stNumberInput input {
        background: #ffffff !important;
        border: 1px solid #c5e1a5 !important;
        border-radius: 6px !important;
        color: #000000 !important; /* Force Black Text */
        font-weight: 600 !important;
        caret-color: #000000;
    }
    
    .stNumberInput input:focus {
        border-color: #43a047 !important;
        box-shadow: 0 0 0 2px rgba(67, 160, 71, 0.2) !important;
    }

    /* Selectbox */
    div[data-baseweb="select"] > div {
        background: #ffffff !important;
        border: 1px solid #c5e1a5 !important;
        border-radius: 6px !important;
        color: #000000 !important;
    }
    
    div[role="listbox"] li {
        color: #000000 !important;
        background-color: white !important;
    }
    
    /* Radio & Checkbox Options */
    .stRadio div[role="radiogroup"] label p {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    .stCheckbox label p {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    /* Section Headers inside Form */
    .form-header {
        color: #000000 !important;
        font-size: 1.1rem;
        font-weight: 800;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #a5d6a7;
        padding-bottom: 5px;
    }

    /* --- NEW SUBMIT BUTTON STYLE (LIFT EFFECT) --- */
    .stButton > button {
        background: linear-gradient(to bottom, #66bb6a 5%, #43a047 100%);
        background-color: #43a047;
        border-radius: 8px;
        border: 2px solid #2e7d32;
        color: #000000 !important; /* Force Black Text */
        font-family: 'Segoe UI', sans-serif;
        font-size: 1.2rem;
        font-weight: 800;
        padding: 12px 30px;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        
        /* Default State: Subtle Shadow, No Lift */
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
        transform: translateY(0);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(to bottom, #43a047 5%, #66bb6a 100%);
        
        /* Hover State: Lifts UP and Shadow Grows */
        transform: translateY(-4px); 
        box-shadow: 0 10px 20px rgba(46, 125, 50, 0.4); 
    }
    
    .stButton > button:active {
        /* Click State: Sinks slightly */
        transform: translateY(-1px);
        box-shadow: 0 5px 10px rgba(46, 125, 50, 0.3);
    }
    
    /* Result cards */
    .result-card {
        padding: 2.5rem;
        border-radius: 12px;
        text-align: left;
        font-size: 1.1rem;
        font-weight: 500;
        margin: 2rem 0;
        border-left: 6px solid;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .high-risk {
        background: #fff3e0;
        border-color: #ef6c00;
        color: #000000 !important;
    }
    
    .low-risk {
        background: #e8f5e9;
        border-color: #2e7d32;
        color: #000000 !important;
    }
    
    /* Metric boxes */
    .metric-box {
        background: #ffffff;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 0.5rem;
        text-align: center;
        border: 1px solid #dcedc8;
    }
    
    .metric-label {
        color: #000000 !important;
        font-size: 0.75rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 1px;
    }
    
    .metric-value {
        color: #000000 !important;
        font-size: 1.8rem;
        font-weight: 800;
    }
    
    /* Main Section Header */
    .section-header {
        color: #000000 !important;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 2rem 0 1.2rem 0;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 2px solid #a5d6a7;
        padding-bottom: 0.5rem;
    }
    
    /* Info box */
    .info-box {
        background: #e8f5e9;
        border-left: 5px solid #43a047;
        padding: 1.2rem;
        border-radius: 4px;
        margin: 1.5rem 0;
        color: #000000 !important;
        font-size: 0.95rem;
    }
    
    /* Tab label */
    .tab-label {
        color: #000000 !important;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        display: block;
    }
    </style>
""", unsafe_allow_html=True)

# Page config
st.set_page_config(page_title="CardioGuard AI", page_icon="💚", layout="centered")

# Header
st.markdown('<h1 class="main-title">CardioGuard AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Clinical-Grade Cardiovascular Risk Assessment</p>', unsafe_allow_html=True)

# Info box
st.markdown("""
    <div class="info-box">
        <strong>System Active:</strong> Securely process your biometric data below to generate a predictive risk analysis 
        using our calibrated Random Forest algorithm.
    </div>
""", unsafe_allow_html=True)

# --- Input form ---
st.markdown('<div class="section-header">Patient Vitals & History</div>', unsafe_allow_html=True)

with st.form("input_form"):
    # Personal Information
    st.markdown('<div class="form-header">Demographics</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    age_years = col1.number_input("Age (years)", 20, 90, 50)
    gender = col2.selectbox("Gender", ["Male", "Female"])
    
    st.markdown('<div class="form-header">Anthropometrics</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    height = col3.number_input("Height (cm)", 120, 220, 170)
    weight = col4.number_input("Weight (kg)", 40, 150, 70)
    
    st.markdown('<div class="form-header">Hemodynamics</div>', unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    ap_hi = col5.number_input("Systolic BP (mmHg)", 90, 250, 120)
    ap_lo = col6.number_input("Diastolic BP (mmHg)", 40, 180, 80)
    
    st.markdown('<div class="form-header">Clinical Labs</div>', unsafe_allow_html=True)
    
    # Cholesterol with radio buttons
    st.markdown('<span class="tab-label">Total Cholesterol</span>', unsafe_allow_html=True)
    cholesterol = st.radio(
        "cholesterol_radio",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Normal (<200 mg/dL)", 2: "Borderline (200-239)", 3: "High (≥240)"}[x],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Glucose with radio buttons
    st.markdown('<span class="tab-label">Glucose Levels</span>', unsafe_allow_html=True)
    gluc = st.radio(
        "glucose_radio",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Normal", 2: "Pre-diabetes", 3: "Diabetes"}[x],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown('<div class="form-header">Lifestyle Factors</div>', unsafe_allow_html=True)
    
    # Toggle switches for lifestyle factors
    col7, col8, col9 = st.columns(3)
    
    with col7:
        smoke = st.checkbox("Active Smoker", value=False, key="smoke_toggle")
        smoke = 1 if smoke else 0
    
    with col8:
        alco = st.checkbox("Alcohol Consumption", value=False, key="alco_toggle")
        alco = 1 if alco else 0
    
    with col9:
        active = st.checkbox("Regular Exercise", value=False, key="active_toggle")
        active = 1 if active else 0
    
    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Generate Risk Profile")

# --- Prediction logic ---
if submitted:
    # Load scalers (now available in bundle)
    scaler_hi = bundle['scaler_hi']
    scaler_lo = bundle['scaler_lo']

    # Fix gender: 1=Female, 2=Male
    gender_val = 2 if gender == "Male" else 1

    # Base input DataFrame
    features_df = pd.DataFrame([{
        'gender': gender_val, 'height': height, 'weight': weight,
        'ap_hi': ap_hi, 'ap_lo': ap_lo, 'cholesterol': cholesterol,
        'gluc': gluc, 'smoke': smoke, 'alco': alco, 'active': active,
        'age_years': age_years
    }])

    # Clip age to min=33 for OOD (young = low risk in data)
    if age_years < 33:
        features_df['age_years'] = 33

    # BMI
    features_df['bmi'] = features_df['weight'] / ((features_df['height'] / 100) ** 2)

    # EXACT training binning: pd.cut([0,40,50,60,np.inf], labels=[1,2,3,4])
    features_df['age_group'] = pd.cut(
        features_df['age_years'],
        bins=[0, 40, 50, 60, np.inf],
        labels=[1, 2, 3, 4],
        right=False
    ).astype(int)

    # EXACT z-scores using loaded scalers
    features_df['ap_hi_z'] = scaler_hi.transform(features_df[['ap_hi']]).flatten()
    features_df['ap_lo_z'] = scaler_lo.transform(features_df[['ap_lo']]).flatten()

    # Exact logs (np.log, as in notebook)
    features_df['ap_hi_log'] = np.log(features_df['ap_hi'])
    features_df['ap_lo_log'] = np.log(features_df['ap_lo'])
    features_df['bmi_log'] = np.log(features_df['bmi'])

    # Reindex to training columns
    features_df = features_df.reindex(columns=feature_order, fill_value=0)

    # Cast categoricals to int
    cat_cols = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'age_group']
    for col in cat_cols:
        if col in features_df.columns:
            features_df[col] = features_df[col].astype(int)

    # Predict with RF model
    prediction = model.predict(features_df)[0]
    proba = model.predict_proba(features_df)[0][1] * 100  # P(high risk)

    st.markdown('<div class="section-header">Analysis Report</div>', unsafe_allow_html=True)
    
    if prediction == 1:
        st.markdown(f"""
            <div class="result-card high-risk">
                <strong>⚠️ ELEVATED RISK DETECTED</strong><br>
                <div style="font-size: 3rem; font-weight: 800; margin: 10px 0; color: #000000;">{proba:.1f}%</div>
                <span style="font-size: 0.9rem;">Estimated probability of cardiovascular disease</span>
            </div>
        """, unsafe_allow_html=True)
        st.error("Action Required: Please consult with a cardiologist for further diagnostic testing.")
    else:
        st.markdown(f"""
            <div class="result-card low-risk">
                <strong>✅ OPTIMAL RISK PROFILE</strong><br>
                <div style="font-size: 3rem; font-weight: 800; margin: 10px 0; color: #000000;">{proba:.1f}%</div>
                <span style="font-size: 0.9rem;">Estimated probability of cardiovascular disease</span>
            </div>
        """, unsafe_allow_html=True)
        st.success("Analysis: Your biometrics suggest a healthy cardiovascular system.")
    
    # Display calculated metrics
    bmi = features_df['bmi'].values[0]
    st.markdown('<div class="section-header">Key Biometrics</div>', unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Calculated BMI</div>
                <div class="metric-value">{bmi:.1f}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col_m2:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Blood Pressure</div>
                <div class="metric-value">{ap_hi}/{ap_lo}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col_m3:
        bmi_category = "Normal" if 18.5 <= bmi < 25 else ("Underweight" if bmi < 18.5 else "Overweight")
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Weight Status</div>
                <div class="metric-value" style="font-size: 1.2rem;">{bmi_category}</div>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; color: #000000; padding: 20px;">
        <small>CardioGuard AI © 2026 | Clinical Decision Support Tool | Not for emergency use</small>
    </div>
""", unsafe_allow_html=True)