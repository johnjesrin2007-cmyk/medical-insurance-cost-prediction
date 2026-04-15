import os
import sys
import mlflow
from prefect import flow

# -------------------------
# 🔥 MLFLOW CONFIG
# -------------------------


# -------------------------
# 📦 IMPORT PATH FIX
# -------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# -------------------------
# 📥 IMPORT MODULES
# -------------------------
from src.preprocess import preprocess_data, get_preprocessor
from src.train import train_model
from src.model import get_model_pipeline   # make sure this file exists

# -------------------------
# 🚀 PIPELINE
# -------------------------
@flow(name="ML Training Pipeline")
def training_pipeline():

    print("🚀 Starting training pipeline...")

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("medical_insurance_cost_prediction")

    data_path = "data/raw.csv"

    X, y = preprocess_data(
        path=data_path,
        training=True,
        target_col="charges"
    )

    print("📊 Data loaded and preprocessed")

    preprocess = get_preprocessor(X)

    pipeline = get_model_pipeline(preprocess)

    try:
        trained_pipeline = train_model(pipeline, X, y)
        print("✅ Model training completed")
    except Exception as e:
        print("❌ Training failed:", str(e))
        raise

    return trained_pipeline

if __name__ == "__main__":
    training_pipeline()