import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# --- Configuration ---
DATA_FILE = r'd:\SupNum\Formation IA GIZ\Loan Approval\loan_data.csv'
MODEL_DIR = r'd:\SupNum\Formation IA GIZ\Loan Approval\models'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 1. Load Data ---
print("Loading data...")
try:
    df = pd.read_csv(DATA_FILE)
    print(f"Data loaded. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: File not found at {DATA_FILE}")
    exit()

# --- 2. Preprocessing ---
print("Preprocessing data...")

# Handle Outliers (Age > 100)
df = df[df['person_age'] <= 100]

# Define features and target
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Identify numeric and categorical columns
numeric_features = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 
                    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']
categorical_features = ['person_gender', 'person_education', 'person_home_ownership', 
                        'loan_intent', 'previous_loan_defaults_on_file']

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Model Training ---
print("Training model (Random Forest)...")
# Using Random Forest as it is robust and generally performs well without heavy tuning
model = RandomForestClassifier(random_state=42, n_estimators=100)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)])

pipeline.fit(X_train, y_train)

# --- 4. Evaluation ---
print("Evaluating model...")
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")

# --- 5. Save Model ---
print(f"Saving model to {MODEL_PATH}...")
joblib.dump(pipeline, MODEL_PATH)
print("Model saved successfully.")
