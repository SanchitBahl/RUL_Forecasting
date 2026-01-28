# 🛠️ RUL Forecasting: Predictive Maintenance System

A full-stack machine learning application designed to predict the **Remaining Useful Life (RUL)** of industrial assets (engines). This project features a decoupled architecture with a high-performance **FastAPI** backend and an interactive **Streamlit** frontend.

## 🌐 Live Application

* **Frontend:** [https://rul-forecasting.streamlit.app/](https://rul-forecasting.streamlit.app/)
* **Backend API:** [https://rul-forecasting.onrender.com](https://rul-forecasting.onrender.com)

> **⚠️ Note on Cold Starts:** The API is hosted on Render's free tier. If the service has been inactive, the server "goes to sleep." Upon your first request, please allow **~2 minutes** for the instance to spin up. A status indicator is provided in the app to manage this.

---

## 🏗️ System Architecture

Unlike standard monolithic ML scripts, this project is built as a distributed system:



* **Frontend (Streamlit):** A user-friendly interface for data ingestion and result presentation.
* **Backend (FastAPI):** A robust REST API that serves the trained model, handles preprocessing, and returns batch inferences.
* **Model:** A Random Forest Regressor optimized via Grid Search.

---

## 🧠 Model Selection & Benchmarking

The final model was selected after comparing several algorithms on the **NASA C-MAPSS dataset**. While deep learning models (like LSTMs) are common for this task, **Random Forest** provided the best balance of generalization and RMSE for this specific implementation.

### Tournament Results (on Validation Data)

| Model | RMSE | R² Score | Note |
| :--- | :--- | :--- | :--- |
| Linear Regression | 43.81493469688726 | -0.11169280094599832 | Baseline model; struggled with non-linearity. |
| Ridge Regression | 43.79330805705907 | -0.11059562956026037 | Almost no improvement over linear baseline. |
| Lasso Regression | 43.70021941161003 | -0.10587920282044339 | Minimal improvement over linear baseline. |
| **Random Forest (Grid Search)** | **34.85578057483267** | **0.2964579146677415** | **Selected: Most robust against sensor noise.** |
| SVR (RBF Kernel) | 25.617379476156497 | 0.6199770201378237 | High accuracy but slow inference on large batches. |
| XGBoost | 30.035320597563754 | 0.47759777307510376 | Strong performance, but prone to noise sensitivity. |

> **Deep Dive:** View the full training pipeline, feature engineering steps, and sensor correlation analysis in the `notebooks/` directory.

---

## 🚀 Key Features

* **Decoupled Design:** API-first approach allows the model to be consumed by other services beyond the Streamlit UI.
* **Cold-Start Management:** Integrated UI logic to "wake up" the API server before processing data.
* **Bulk Inference:** Supports `.txt` file uploads for unit-based RUL predictions.
* **Tabular Output:** Results are returned in a clean, structured format for easy maintenance logging.

---

## 💻 Tech Stack

* **ML/Data:** Scikit-Learn, Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **API:** FastAPI, Uvicorn (Hosted on Render)
* **UI:** Streamlit (Hosted on Streamlit Cloud)
* **Version Control:** Git/GitHub

---

## 🛠️ Usage

1.  **Check API Status:** Use the **"Wake Server"** button on the dashboard. Wait for the status to turn green (**✅ Server Ready**).
2.  **Upload Data:** Upload the engine sensor data (currently optimized for the `.txt` format).
3.  **View Results:** The model will process the operational settings and 21 sensor readings to provide the predicted RUL for each unit.