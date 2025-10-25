import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained models
# -----------------------------
try:
    log_model = joblib.load("logistic_model.pkl")
    rf_model = joblib.load("rf_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    feature_names = list(log_model.feature_names_in_) 
except FileNotFoundError:
    st.error("Error: Model files not found. Please ensure 'logistic_model.pkl', 'rf_model.pkl', and 'xgb_model.pkl' are available.")
    feature_names = [f"Feature_{i}" for i in range(10)] 

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="IBD Risk Prediction", layout="wide")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
    <style>
    .stApp { background-color: #ADD8E6; }  /* Light Blue */

    .stSelectbox>div>div>div>select { text-align: center; } 
    
    .stSelectbox label {
        font-weight: bold !important;
        font-size: 22px !important;
        color: #000000 !important;
        text-align: center;
        width: 100%; 
        display: block; 
        margin-bottom: 5px; 
    }

    .stSelectbox select {
        border: 2px solid black; 
        border-radius: 5px; 
        padding: 5px 10px; 
    }

    .logo-left, .logo-right { width: 120px; display:block; margin:auto; }
    .institute-name { text-align:center; font-weight:bold; font-size:16px; margin-top:5px; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Logos and Title
# -----------------------------
col_logo_left, col_title, col_logo_right = st.columns([1, 5, 1])

with col_logo_left:
    st.markdown('<img src="https://brandlogovector.com/wp-content/uploads/2022/04/IIT-Delhi-Icon-Logo.png" class="logo-left">', unsafe_allow_html=True)
    st.markdown('<div class="institute-name">Indian Institute of Technology Delhi</div>', unsafe_allow_html=True)

with col_title:
    st.markdown(
        "<h1 style='text-align:center; font-size:38px; color:black;'>DMCH-IITD Machine Learning Tool for Estimating the Diet Percentage Similarity with respect to Diets Consumed by Inflammatory Bowel Disease Patients Prior to Diagnosis</h1>",
        unsafe_allow_html=True
    )
    # Intro paragraph below title with spacing
    st.markdown(
        """
        <p style='text-align:center; font-size:22px; color:black; line-height:1.5; margin-top:22px;'>
        This tool uses machine learning models to estimate the similarity of your diet with those consumed by patients prior to an Inflammatory Bowel Disease (IBD) diagnosis.
        It combines Logistic Regression, Random Forest, and XGBoost to provide reliable predictions.
        The dietary survey was conducted by Dayanand Medical College and Hospital, Ludhiana.
        </p>
        """,
        unsafe_allow_html=True
    )

with col_logo_right:
    st.markdown('<img src="https://tse2.mm.bing.net/th/id/OIP.fNb1hJAUj-8vwANfP3SDJgAAAA?pid=Api&P=0&h=180" class="logo-right">', unsafe_allow_html=True)
    st.markdown('<div class="institute-name">Dayanand Medical College and Hospital Ludhiana</div>', unsafe_allow_html=True)

# -----------------------------
# Feature names and cleaning utility
# -----------------------------
def clean_feature_name(name):
    return name.replace("_", " ").title()

# -----------------------------
# Layout input and output in same row
# -----------------------------
col_input, col_output = st.columns([3, 1])  # input wider, predictions on right

# -------- Left column: Input features --------
features = {}
with col_input:
    st.markdown("<br>", unsafe_allow_html=True)
    st.header("Provide information about your dietary habits. Select the level of consumption for each food item (0 = None, 20 = High).")
    st.markdown("<br>", unsafe_allow_html=True)

    n = len(feature_names)
    half = n // 2
    options = list(range(21))  # dropdown 0-15

    for i in range(half):
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            features[feature_names[i]] = st.selectbox(
                label=clean_feature_name(feature_names[i]),
                options=options,
                index=0,
                key=f"{feature_names[i]}"
            )
        with c2:
            features[feature_names[i+half]] = st.selectbox(
                label=clean_feature_name(feature_names[i+half]),
                options=options,
                index=0,
                key=f"{feature_names[i+half]}"
            )

input_df = pd.DataFrame([features], columns=feature_names)

# -------- Right column: Predictions --------
with col_output:
    st.markdown("<br><br><br>", unsafe_allow_html=True)  # align with left column inputs
    st.header("Predictions")
    predict_clicked = st.button("Predict")

    if predict_clicked:
        try:
            # scale probabilities to 0-100
            logistic_score = log_model.predict_proba(input_df)[0][1] * 100
            rf_score = rf_model.predict_proba(input_df)[0][1] * 100
            xgb_score = xgb_model.predict_proba(input_df)[0][1] * 100
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            logistic_score, rf_score, xgb_score = 50, 50, 50

        st.subheader("Similarity Score (0-100)")
        st.write(f"**Logistic Regression:** {logistic_score:.0f}")
        st.write(f"**Random Forest:** {rf_score:.0f}")
        st.write(f"**XGBoost:** {xgb_score:.0f}")
