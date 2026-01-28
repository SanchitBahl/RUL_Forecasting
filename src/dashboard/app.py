import streamlit as st
import pandas as pd
import requests
import time

st.set_page_config(page_title="Engine RUL Predictor", layout="centered")

st.title("Turbofan Engine Remaining Useful Life (RUL) Predictor")

st.write(
    """
    Upload engine sensor data (.txt file) and receive predicted Remaining Useful Life (RUL).
    This system uses a machine learning model trained on NASA CMAPSS data.
    """
)

# Accept TXT files
uploaded_file = st.file_uploader(
    "Upload engine sensor data (.txt file)",
    type=["txt"]
)

API_URL = "https://rul-forecasting.onrender.com"

def check_api_status():
    """Attempts to contact the API and returns True if active."""
    try:
        # We set a short timeout for the check, 
        # but Render's cold start happens at the networking layer.
        response = requests.get(f"{API_URL}/status", timeout=5)
        if response.status_code == 200:
            return True
    except:
        return False
    return False

# --- API WAKE-UP LOGIC ---
if "api_online" not in st.session_state:
    st.session_state.api_online = False

status_container = st.empty()

if not st.session_state.api_online:
    with status_container.container():
        st.warning("⚠️ API Server is sleeping (Free Tier).")
        if st.button("Wake Up Server"):
            with st.status("Waking up the engine... This usually takes 1-2 minutes.", expanded=True) as status:
                # Loop until the API responds
                max_retries = 30 
                for i in range(max_retries):
                    if check_api_status():
                        st.session_state.api_online = True
                        status.update(label="✅ Server Ready!", state="complete", expanded=False)
                        break
                    time.sleep(5) # Wait 5 seconds between pings
                else:
                    status.update(label="❌ Server timed out. Please refresh.", state="error")


if uploaded_file is not None:
    # Preview uploaded file
    try:
        # For NASA .txt files: space-separated, no header
        raw_cols = (
            ["engine", "cycle"]
            + [f"setting{i}" for i in range(1, 4)]
            + [f"sensor{i}" for i in range(1, 22)]
        )
        df = pd.read_csv(uploaded_file, sep=r"\s+", header=None, names=raw_cols, engine="python")

    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        st.stop()

    st.subheader("Input Data Preview")
    st.dataframe(df.head())

    if st.button("Predict RUL"):
        with st.spinner("Predicting..."):
            try:
                file_bytes = uploaded_file.getvalue()
                response = requests.post(
                    f"{API_URL}/predict",
                    files={"file": (uploaded_file.name, file_bytes, "text/plain")}

                )

                if response.status_code == 200:
                    result = pd.DataFrame(response.json())
                    st.subheader("Predicted RUL")
                    st.dataframe(result)
                else:
                    st.error(f"Prediction failed. Status code: {response.status_code}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")