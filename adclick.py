import streamlit as st
import pandas as pd
import pickle

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Ad Click Prediction",
    page_icon="📢",
    layout="centered"
)

# ---------------------------------
# Custom CSS
# ---------------------------------
st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(135deg, #f0f4ff, #e6fff2);
}

/* Title */
h1 {
    color: #1e8449;
    font-weight: 700;
}

/* Sub headers */
h3 {
    color: #145a32;
}

/* Input labels */
label {
    font-weight: 600;
    color: #2c3e50;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(90deg, #27ae60, #2ecc71);
    color: white;
    border-radius: 12px;
    padding: 0.6em 1.5em;
    font-size: 16px;
    border: none;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.15);
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1e8449, #27ae60);
    transform: scale(1.02);
}

/* Success box */
.stSuccess {
    background-color: #e9f7ef !important;
    border-left: 6px solid #27ae60;
}

/* Divider */
hr {
    border: 1px solid #d5f5e3;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 12px;
    color: #7f8c8d;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# Load Model & Features
# ---------------------------------
@st.cache_resource
def load_model():
    with open("ad_click_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_features():
    with open("ad_click_features.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
feature_cols = load_features()

# ---------------------------------
# Title
# ---------------------------------
st.markdown(
    "<h1 style='text-align:center;'>📢 Ad Click Prediction App</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Predict whether a user will click an advertisement</p>",
    unsafe_allow_html=True
)

st.divider()

# ---------------------------------
# Input Section
# ---------------------------------
st.subheader("👤 User Information")

full_name = st.selectbox(
    "Full Name",
    ["User001", "User002", "User003", "User004", "User005"]
)

age = st.selectbox(
    "Age",
    list(range(18, 70))
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female", "Non-Binary", "Unknown"]
)

device_type = st.selectbox(
    "Device Type",
    ["Mobile", "Desktop", "Tablet", "Unknown"]
)

ad_position = st.selectbox(
    "Ad Position",
    ["Top", "Side", "Bottom", "Unknown"]
)

browsing_history = st.selectbox(
    "Browsing History",
    ["Shopping", "Education", "Entertainment", "News", "Social Media", "Unknown"]
)

time_of_day = st.selectbox(
    "Time of Day",
    ["Morning", "Afternoon", "Evening", "Night", "Unknown"]
)

st.divider()

# ---------------------------------
# Prediction
# ---------------------------------
if st.button("🔮 Predict Click"):
    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "device_type": device_type,
        "ad_position": ad_position,
        "browsing_history": browsing_history,
        "time_of_day": time_of_day
    }])

    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(
        columns=feature_cols,
        fill_value=0
    )

    prediction = model.predict(input_encoded)[0]

    st.markdown("### 🧾 Prediction Result")
    st.success(f"Click Prediction (1 = Yes, 0 = No): {prediction}")

# ---------------------------------
# Footer
# ---------------------------------
st.markdown(
    "<div class='footer'>Made with ❤️ using Streamlit & Machine Learning</div>",
    unsafe_allow_html=True
)
