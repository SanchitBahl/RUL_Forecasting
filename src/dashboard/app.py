import streamlit as st
import pandas as pd
import requests

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

API_URL = "https://rul-forecasting.onrender.com/predict"

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
                    API_URL,
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