# 🚀 Medical Insurance Cost Prediction (MLOps Project)

## 📌 Overview

This project is an **end-to-end MLOps pipeline** for predicting medical insurance charges based on user data.
It demonstrates how to move from a simple ML model to a **production-ready system** with:

* Automated training pipeline
* Experiment tracking
* Model versioning
* API deployment
* Monitoring system
* CI/CD pipeline

---

## 🧠 Problem Statement

Predict **medical insurance charges** based on factors like:

* Age
* BMI
* Smoking status
* Region
* Number of children

---

## 🏗️ Project Architecture

```
project/
│
├── pipeline/
│   └── training_pipeline.py        # Orchestrates training
│
├── src/
│   ├── preprocess.py               # Data preprocessing
│   ├── train.py                    # Model training + MLflow logging
│   ├── model.py                    # Pipeline creation
│   └── logger.py                   # Prediction logging
│
├── logs/                           # Auto-created (monitoring logs)
├── main.py                         # FastAPI app
├── requirements.txt
├── Dockerfile                      # Containerization setup
├── .gitignore                      # Ignore unnecessary files
├── .github/
│   └── workflows/
│       └── ci.yaml                 # CI/CD pipeline
└── README.md

---

## ⚙️ Tech Stack

* **Python**
* **Scikit-learn**
* **MLflow** (experiment tracking & model registry)
* **FastAPI** (model serving)
* **Prefect** (pipeline orchestration)
* **Docker** (containerization)
* **GitHub Actions** (CI/CD)

---

## 🔥 Features

### ✅ Training Pipeline

* Data preprocessing
* Cross-validation
* Hyperparameter tuning (RandomizedSearchCV)
* Model evaluation

---

### 📊 Experiment Tracking

* Logs parameters, metrics, and models
* Model versioning using MLflow

---

### 🤖 Model Serving

* FastAPI-based REST API
* Predicts insurance cost from user input

---

### 📈 Monitoring

* Logs predictions to CSV
* Stores:

  * Input data
  * Predictions
  * Timestamp

---

### 🔄 CI/CD Pipeline

* Runs training on push
* Builds Docker image
* Pushes to Docker Hub

---

## 🚀 How to Run Locally

### 1️⃣ Clone repo

```bash
git clone <your-repo-link>
cd project
```

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run training pipeline

```bash
python pipeline/training_pipeline.py
```

---

### 4️⃣ Start MLflow UI

```bash
mlflow ui
```

👉 Open: http://127.0.0.1:5000
Assign model alias (e.g., `champion`)

---

### 5️⃣ Run API

```bash
uvicorn main:app --reload
```

👉 Open: http://127.0.0.1:8000/docs

---

## 📥 API Usage

### POST `/predict`

#### Input:

```json
{
  "age": 25,
  "sex": "male",
  "bmi": 27.5,
  "children": 0,
  "smoker": "no",
  "region": "southwest"
}
```

#### Output:

```json
{
  "predicted_charges": 3200.45
}
```

---

## 📊 Monitoring Output

File auto-created:

```
logs/predictions.csv
```

Example:

```
age,sex,bmi,children,smoker,region,prediction,timestamp
25,male,27.5,0,no,southwest,3200.45,2026-04-15 18:45:12
```

---

## 🧠 Key Learnings

* Building modular ML pipelines
* Using MLflow for tracking & model management
* Serving ML models via API
* Implementing basic monitoring
* Automating workflows with CI/CD

---

## ⚠️ Notes

* No `.pkl` files used (model handled via MLflow)
* `mlruns/`, `logs/` are auto-generated
* Dataset not included (use external link)

---

## 🔮 Future Improvements

* Add data drift detection
* Build monitoring dashboard
* Deploy on cloud (Render / AWS)
* Add authentication to API

---

## 💪 Conclusion

This project demonstrates a **complete MLOps workflow**, moving beyond basic ML into **real-world system design**.

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
