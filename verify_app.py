from fastapi.testclient import TestClient
from app import app
import os

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] == True
    print("[PASS] Health Check")

def test_predict_approve():
    # Data that should likely be approved (High income, low debt)
    payload = {
        "person_age": 30,
        "person_gender": "male",
        "person_education": "Master",
        "person_income": 80000,
        "person_emp_exp": 10,
        "person_home_ownership": "OWN",
        "loan_amnt": 5000,
        "loan_intent": "PERSONAL",
        "loan_int_rate": 8.0,
        "loan_percent_income": 0.06,
        "cb_person_cred_hist_length": 5,
        "credit_score": 750,
        "previous_loan_defaults_on_file": False
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    print(f"Prediction: {data['decision']}, Probability: {data['probability']}")
    assert "decision" in data
    print("[PASS] Predict Approval")

def test_predict_reject_hard_rule():
    # Hard rule: Age < 18
    payload = {
        "person_age": 17,
        "person_gender": "male",
        "person_education": "High School",
        "person_income": 10000,
        "person_emp_exp": 0,
        "person_home_ownership": "RENT",
        "loan_amnt": 1000,
        "loan_intent": "PERSONAL",
        "loan_int_rate": 10.0,
        "loan_percent_income": 0.1,
        "cb_person_cred_hist_length": 0,
        "credit_score": 500,
        "previous_loan_defaults_on_file": False
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["decision"] == "REJECTED"
    assert response.json()["reason"] == "Age below 18"
    print("[PASS] Hard Rule (Age < 18)")

def check_files():
    assert os.path.exists("static/index.html")
    assert os.path.exists("static/style.css")
    assert os.path.exists("static/script.js")
    print("[PASS] Static Files Exist")

if __name__ == "__main__":
    try:
        from app import load_model
        load_model() # Manually trigger startup event logic if TestClient doesn't do it automatically in all versions
        
        test_health()
        test_predict_approve()
        test_predict_reject_hard_rule()
        check_files()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        exit(1)
