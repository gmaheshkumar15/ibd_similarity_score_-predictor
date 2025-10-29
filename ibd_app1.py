import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained model
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
    .stApp { background-color: #ADD8E6; }

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
    
    .large-score {
        font-size: 70px !important;
        font-weight: bold;
        color: #8B0000;
        text-align: center;
        margin-top: 20px;
    }

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
    st.markdown(
        "<h1 style='text-align:center; font-size:36px; color:black;'>DMCH-IITD Machine Learning Tool for Estimating the Diet Percentage Similarity with Respect to Diets Consumed by Inflammatory Bowel Disease Patients Prior to Diagnosis</h1>",
        unsafe_allow_html=True
    )


with col_logo_right:
    st.markdown('<img src="https://raw.githubusercontent.com/gmaheshkumar15/ibd_similarity_score_-predictor/main/dmch.jpeg" class="logo-right">', unsafe_allow_html=True)
    st.markdown('<div class="institute-name">Dayanand Medical College and Hospital Ludhiana</div>', unsafe_allow_html=True)

# -----------------------------
# Intro Paragraph
# -----------------------------
st.markdown("<hr style='border: 1px solid black;'>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:left; font-size:20px; color:black; line-height:1.5;'>
This tool uses a machine learning model to estimate the similarity of your diet with those consumed by patients prior to an Inflammatory Bowel Disease (IBD) diagnosis. It uses a Logistic Regression model to estimate prediction. The ML model was trained based on data from a dietary survey conducted by DMCH Ludhiana among IBD patients and controls without IBD. IBD patients were asked to report their dietary habits prior to diagnosis, and controls were asked to report current food habits.</p>
""", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid black;'>", unsafe_allow_html=True)

# -----------------------------
# Helper: Clean feature names
# -----------------------------
def clean_feature_name(name):
    return name.replace("_", " ").title()

# -----------------------------
# Feature value limits
# -----------------------------
feature_value_limits = {
    "Wheat(Chapati,Roti,Naan,Dalia,Rawa/Sooji,Seviyaan": list(range(0, 6)),
    "Wheat Free Cereals": list(range(0, 36)),
    "Fruits": list(range(0, 21)),
    "Other Vegetables": list(range(0, 26)),
    "Starchy(Potato,Sweet Patato,Arbi Etc)": list(range(0, 6)),
    "Pulses And Legumes": list(range(0, 16)),
    "Predominant Saturated Fats": list(range(0, 11)),
    "Predominant Unsaturated Fats": list(range(0, 11)),
    "Trans Fats": list(range(0, 6)),
    "Nuts And Oilseeds": list(range(0, 6)),
    "Eggs,Fish And Poultry": list(range(0, 16)),
    "Red Meat": list(range(0, 6)),
    "Milk": list(range(0, 6)),
    "Low Lactose Dairy": list(range(0, 16)),
    "Sweetend Beverages": list(range(0, 21)),
    "Ultra Processed Foods": list(range(0, 76)),
    "Readt To Eat Packaged Snacks": list(range(0, 11)),
    "Savory Snacks": list(range(0, 21)),
    "Processed Foods": list(range(0, 46)),
    "Indian Sweet Meats": list(range(0, 11)),
    "Food Supplements": list(range(0, 26)),
    "Ergogenic Supplements": list(range(0, 6))
}

# -----------------------------
# Layout input and output
# -----------------------------
col_input, col_output = st.columns([3, 1])

# -------- Input Section --------
features = {}
with col_input:
    st.header("In the below fields, provide information about your dietary habits. Select the level of consumption for each food item (higher values indicate higher consumption, and vice versa)")


    n = len(feature_names)
    half = (n + 1) // 2  # Handles odd numbers

    # Pre-clean mapping for better matching
    clean_map = {k.lower().replace(" ", "").replace("_", ""): v for k, v in feature_value_limits.items()}

    for i in range(half):
        c1, c2 = st.columns(2, gap="medium")
        for col, idx in zip([c1, c2], [i, i + half]):
            if idx >= n:
                continue

            feature_name = feature_names[idx]
            clean_key = feature_name.lower().replace(" ", "").replace("_", "")

            # Match value limits ignoring case, spaces, underscores
            options = clean_map.get(clean_key, list(range(0, 38)))

            with col:
                features[feature_name] = st.selectbox(
                    label=clean_feature_name(feature_name),
                    options=options,
                    index=0,
                    key=feature_name
                )

# Convert to DataFrame
input_df = pd.DataFrame([features], columns=feature_names)

# -------- Output Section --------
with col_output:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.header("Prediction")
    predict_clicked = st.button("Predict")

    if predict_clicked:
        try:
            logistic_score = log_model.predict_proba(input_df)[0][1] * 100
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            logistic_score = 50

        st.subheader("Similarity Score (0-100):")
        st.markdown(f"<div class='large-score'>{logistic_score:.0f}</div>", unsafe_allow_html=True)
