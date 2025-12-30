from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
import os

# --- Configuration ---
MODEL_PATH = r'd:\SupNum\Formation IA GIZ\Loan Approval\models\model.pkl'

# --- App Initialization ---
app = FastAPI(title="Loan Approval API", description="API pour l'évaluation des demandes de crédit")

# --- Load Model ---
model_pipeline = None

@app.on_event("startup")
def load_model():
    global model_pipeline
    if os.path.exists(MODEL_PATH):
        model_pipeline = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print(f"Error: Model not found at {MODEL_PATH}")

# --- Data Models ---
class LoanApplication(BaseModel):
    person_age: float
    person_gender: str
    person_education: str
    person_income: float
    person_emp_exp: int
    person_home_ownership: str
    loan_amnt: float
    loan_intent: str
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: float
    credit_score: int
    previous_loan_defaults_on_file: bool

# --- Endpoints ---

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model_pipeline is not None}

@app.post("/predict")
def predict_loan(application: LoanApplication):
    if not model_pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Hard Rules
    if application.person_age < 18:
        return {
            "decision": "REJECTED",
            "reason": "Age below 18",
            "probability": 0.0,
            "status": 0
        }

    # Prepare DataFrame
    data = application.dict()
    # Convert 'previous_loan_defaults_on_file' boolean to proper format if needed
    # The training data might rely on categorical/boolean handling.
    # In the CSV it might be 'Yes'/'No' or 0/1. The metadata says Categorical (Boolean).
    # Checking analyze_loans.py, it was loaded as is.
    
    # We create a DataFrame
    df = pd.DataFrame([data])
    
    # Prediction
    try:
        prediction = model_pipeline.predict(df)[0]
        probability = model_pipeline.predict_proba(df)[0][1]
        
        # Decision Logic
        threshold = 0.5
        decision = "APPROVED" if probability >= threshold else "REJECTED"
        
        return {
            "decision": decision,
            "probability": float(probability),
            "status": int(prediction),
            "reason": "Model Score"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# --- Static Files (Frontend) ---
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
