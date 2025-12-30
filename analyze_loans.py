import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

try:
    from xgboost import XGBClassifier
    xgboost_available = True
except ImportError:
    xgboost_available = False
    print("XGBoost not installed. Skipping XGBoost model.")

# --- Configuration ---
DATA_FILE = r'd:\SupNum\Formation IA GIZ\Loan Approval\loan_data.csv'
REPORT_FILE = r'd:\SupNum\Formation IA GIZ\Loan Approval\analysis_report.html'

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
# Removing records where age is > 100 as per metadata warning
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
print("Training models...")

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'k-NN': KNeighborsClassifier(),
    'SVM': SVC(probability=True, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

if xgboost_available:
    models['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

results = []

for name, model in models.items():
    print(f"Training {name}...")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline.named_steps['classifier'], "predict_proba") else None
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'AUC': auc,
        'y_prob': y_prob # Store for ROC curve plotting
    })

results_df = pd.DataFrame(results).drop('y_prob', axis=1)
best_model_row = results_df.sort_values(by='F1 Score', ascending=False).iloc[0]
best_model_name = best_model_row['Model']

# --- 4. Generate Visualizations (Base64) ---
print("Generating visualizations...")

def plot_to_base64(plt):
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

# Metrics Comparison Bar Plot
plt.figure(figsize=(10, 6))
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
results_melted = results_df.melt(id_vars='Model', value_vars=metrics_to_plot, var_name='Metric', value_name='Score')
sns.barplot(data=results_melted, x='Model', y='Score', hue='Metric')
plt.title('Model Comparison Metrics')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
metrics_plot_b64 = plot_to_base64(plt)
plt.close()

# ROC Curve Plot
plt.figure(figsize=(8, 6))
for res in results:
    if res['y_prob'] is not None:
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        plt.plot(fpr, tpr, label=f"{res['Model']} (AUC = {res['AUC']:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
roc_plot_b64 = plot_to_base64(plt)
plt.close()

# --- 5. Generate HTML Report ---
print("Generating HTML report...")

html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loan Approval Prediction Analysis</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; line-height: 1.6; color: #333; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .highlight {{ background-color: #e8f5e9; font-weight: bold; }}
        .container {{ max-width: 1200px; margin: auto; }}
        .plot {{ margin: 20px 0; text-align: center; }}
        .plot img {{ max-width: 100%; height: auto; border: 1px solid #eee; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .conclusion {{ background-color: #e3f2fd; padding: 20px; border-radius: 8px; border-left: 5px solid #2196f3; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Loan Approval Prediction Analysis</h1>
        <p>Analysis generated on {pd.Timestamp.now()}</p>

        <h2>1. Model Performance Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>AUC</th>
                </tr>
            </thead>
            <tbody>
"""

for _, row in results_df.iterrows():
    row_class = "highlight" if row['Model'] == best_model_name else ""
    html_content += f"""
                <tr class="{row_class}">
                    <td>{row['Model']}</td>
                    <td>{row['Accuracy']:.4f}</td>
                    <td>{row['Precision']:.4f}</td>
                    <td>{row['Recall']:.4f}</td>
                    <td>{row['F1 Score']:.4f}</td>
                    <td>{row['AUC']:.4f}</td>
                </tr>
    """

html_content += f"""
            </tbody>
        </table>

        <h2>2. Visual Learning Comparison</h2>
        <div class="plot">
            <h3>Metrics by Model</h3>
            <img src="data:image/png;base64,{metrics_plot_b64}" alt="Metrics Comparison Plot">
        </div>
        <div class="plot">
            <h3>ROC Curves</h3>
            <img src="data:image/png;base64,{roc_plot_b64}" alt="ROC Curve Plot">
        </div>

        <h2>3. Conclusion and Recommendation</h2>
        <div class="conclusion">
            <p>Based on the evaluation metrics, the <strong>{best_model_name}</strong> is the best performing model for this task.</p>
            <p><strong>Reasoning:</strong></p>
            <ul>
                <li>It achieved the highest F1 Score of <strong>{best_model_row['F1 Score']:.4f}</strong>, which balances Precision and Recall effectively.</li>
                <li>It has an Accuracy of <strong>{best_model_row['Accuracy']:.4f}</strong>.</li>
                <li>The AUC score of <strong>{best_model_row['AUC']:.4f}</strong> indicates strong capability in distinguishing between approved and rejected loans.</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

# Save HTML file
with open(REPORT_FILE, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"Analysis complete. Report saved to: {REPORT_FILE}")
