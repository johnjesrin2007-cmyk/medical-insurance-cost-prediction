from fastapi import FastAPI
import mlflow
from mlflow.tracking import MlflowClient
from pydantic import BaseModel
import pandas as pd

app = FastAPI(title="Medical Insurance Prediction API")


# -------------------------
# 🔥 LOAD LATEST MODEL FROM MLFLOW
# -------------------------
def load_latest_model():
    try:
        client = MlflowClient()

        experiment = client.get_experiment_by_name(
            "medical_insurance_cost_prediction"
        )

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )

        latest_run_id = runs[0].info.run_id

        model_uri = f"runs:/{latest_run_id}/model"
        model = mlflow.pyfunc.load_model(model_uri)

        print(f"✅ Loaded model from run: {latest_run_id}")
        return model

    except Exception as e:
        print("❌ Model loading failed:", str(e))
        return None


model = load_latest_model()


# -------------------------
# 📥 INPUT SCHEMA
# -------------------------
class InputData(BaseModel):
    age: float
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str


# -------------------------
# 🏠 HOME ROUTE
# -------------------------
@app.get("/")
def home():
    return {"message": "MLflow-based API is running"}


# -------------------------
# 🔮 PREDICTION ROUTE
# -------------------------
@app.post("/predict")
def predict(data: InputData):

    global model

    if model is None:
        return {"error": "Model not loaded"}

    try:
        input_dict = data.dict()
        input_df = pd.DataFrame([input_dict])

        prediction = model.predict(input_df)[0]

        # Import here to avoid circular issues
        from src.monitoring.logger import log_prediction
        log_prediction(input_dict, float(prediction))

        return {
            "input": input_dict,
            "predicted_charges": float(prediction)
        }

    except Exception as e:
        return {"error": str(e)}