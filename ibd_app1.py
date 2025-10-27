import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained models
# -----------------------------
try:
    log_model = joblib.load("logistic_model.pkl")
    feature_names = list(log_model.feature_names_in_) 
except FileNotFoundError:
    st.error("Error: Model file 'logistic_model.pkl' not found. Please ensure it is available.")
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
    
    /* New style for the large similarity score */
    .large-score {
        font-size: 70px !important; /* Make the font very large */
        font-weight: bold;
        color: #8B0000; /* Dark Red color for emphasis */
        text-align: center;
        margin-top: 20px; /* Add some space above the score */
    }
    
    /* Reduce margin/padding of the paragraph for reduced space */
    .intro-paragraph {
        margin-bottom: 0px; 
        padding-bottom: 0px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Logos and Title (Row 1)
# -----------------------------
col_logo_left, col_title, col_logo_right = st.columns([1, 5, 1])

with col_logo_left:
    st.markdown('<img src="https://brandlogovector.com/wp-content/uploads/2022/04/IIT-Delhi-Icon-Logo.png" class="logo-left">', unsafe_allow_html=True)
    st.markdown('<div class="institute-name">Indian Institute of Technology Delhi</div>', unsafe_allow_html=True)

with col_title:
    # Main Title remains CENTERED
    st.markdown(
        "<h1 style='text-align:center; font-size:38px; color:black;'>DMCH-IITD Machine Learning Tool for Estimating the Diet Percentage Similarity with Respect to Diets Consumed by Inflammatory Bowel Disease Patients Prior to Diagnosis</h1>",
        unsafe_allow_html=True
    )

with col_logo_right:
    st.markdown('<img src="https://tse2.mm.bing.net/th/id/OIP.fNb1hJAUj-8vwANfP3SDJgAAAA?pid=Api&P=0&h=180" class="logo-right">', unsafe_allow_html=True)
    st.markdown('<div class="institute-name">Dayanand Medical College and Hospital Ludhiana</div>', unsafe_allow_html=True)

# -----------------------------
# Introductory Paragraph (Row 2, full width)
# -----------------------------
st.markdown("<br>", unsafe_allow_html=True) # Add some space below the logos/title
st.markdown(
    """
    <p class='intro-paragraph' style='text-align:left; font-size:22px; color:black; line-height:1.5;'>
    This tool uses a machine learning model to estimate the similarity of your diet with those consumed by patients prior to an Inflammatory Bowel Disease (IBD) diagnosis.
    It Uses a Logistic Regression model to estimate prediction. The ML model was trained based on data from a dietary survey conducted by DMCH Ludhiana among IBD patients and controls without IBD. IBD patients were asked to report their dietary habits prior to diagnosis, and controls were asked to report current food habits.
    </p>
    """,
    unsafe_allow_html=True
)
st.markdown("<hr style='border: 1px solid black;'>", unsafe_allow_html=True) # Separator before inputs

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
    # REMOVED st.markdown("<br>", unsafe_allow_html=True) to reduce space after the horizontal line
    st.header("In the below fields,provide information about your dietary habits.Select the level of consumption for each food item (0=None,37=High).")
    st.markdown("<br>", unsafe_allow_html=True)

    n = len(feature_names)
    half = n // 2
    options = list(range(38))  # dropdown 0-37

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
    st.header("Prediction") 
    predict_clicked = st.button("Predict")

    if predict_clicked:
        try:
            # scale probabilities to 0-100
            logistic_score = log_model.predict_proba(input_df)[0][1] * 100
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            logistic_score = 50 

        st.subheader("Similarity Score (0-100):")
        
        
        # Display the score using the custom large CSS class
        st.markdown(f"<div class='large-score'>{logistic_score:.0f}</div>", unsafe_allow_html=True)
