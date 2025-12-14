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
st.markdown(
    """
    <style>
    /* MAIN BACKGROUND */
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #141e30, #243b55);
        color: white;
    }

    /* HEADINGS */
    h1, h2, h3, h4 {
        color: #e6f2ff;
        font-weight: 700;
    }

    /* LABELS */
    label {
        color: #e0eafc !important;
        font-weight: 600;
    }

    /* SELECT BOX / INPUT */
    div[data-baseweb="select"] > div {
        background-color: #1e1e2f !important;
        color: white !important;
        border-radius: 12px;
        border: 1px solid #3a7bd5;
    }

    /* DROPDOWN OPTIONS */
    ul {
        background-color: #1e1e2f !important;
        color: white !important;
    }

    /* BUTTON */
    button[kind="primary"] {
        background: linear-gradient(90deg, #2193b0, #6dd5ed);
        color: black !important;
        border-radius: 14px;
        font-weight: bold;
        padding: 10px 25px;
        border: none;
    }

    button[kind="primary"]:hover {
        background: linear-gradient(90deg, #6dd5ed, #2193b0);
        color: black !important;
    }

    /* SUCCESS BOX */
    .stSuccess {
        background-color: rgba(0, 255, 255, 0.15);
        border-left: 6px solid #00e5ff;
        color: white;
        font-size: 18px;
    }

    </style>
    """,
    unsafe_allow_html=True
)
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

