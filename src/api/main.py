import io
from fastapi import FastAPI, HTTPException, UploadFile, File
import pandas as pd
import joblib

app = FastAPI(
    title="Turbofan Engine RUL Prediction API",
    description="Upload engine sensor data (.txt) to predict Remaining Useful Life (RUL).",
    version="1.0"
)

# ------------------------------
# Load trained model & feature columns
# ------------------------------
column_names = (
            ["engine", "cycle"]
            + [f"setting{i}" for i in range(1, 4)]
            + [f"sensor{i}" for i in range(1, 22)]
        )

MODEL_PATH = "models/random-forest.pkl"
SCALER_PATH = "models/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ------------------------------
# Prediction endpoint
# ------------------------------
#

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts a .txt file of engine sensor data.
    Returns predicted RUL for each engine cycle.
    """
    try:
        # Read and parse file
        contents = await file.read()
        text_content = contents.decode('utf-8')
        
        # Convert to DataFrame
        df = pd.read_csv(
            io.StringIO(text_content),
            sep=r"\s+",
            skipinitialspace=True
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read uploaded file: {e}"
        )
        
    try:
        df.columns = column_names
        df = df.groupby('engine').last().reset_index().drop(columns=["engine", "cycle", "setting1", "setting2"])
        df.drop(columns=['setting3','sensor1','sensor5','sensor6','sensor10','sensor16','sensor18','sensor19'], inplace=True)
    
        # Scale features
        scaled_test = scaler.transform(df)
    
        # Make predictions
        y_pred = model.predict(scaled_test)

        df["predicted_RUL"] = y_pred

    except Exception as e:
        return {"status": "error", "message": f"Prediction failed: {e}"}

    # Return only relevant columns
    result = df[["predicted_RUL"]]
    result.insert(0, 'engine', range(1, len(result) + 1))

    
    result = result.to_dict(orient="records")
    return result