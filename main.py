"""
FastAPI Backend for Explainable AI Loan Eligibility System
"""
import os
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import requests

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import lime
import lime.lime_tabular

app = FastAPI(title="XAI Loan Eligibility API", version="1.0.0")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Data models
class LoanApplication(BaseModel):
    age: float
    income: float
    loan_amount: float
    debt_to_income: float
    employment_years: float
    credit_history_years: float
    num_delinquencies: int
    num_existing_loans: int
    purpose_of_loan: str
    collateral: str
    residence_status: str

class PredictionResponse(BaseModel):
    decision: str
    probability: float
    threshold: float
    lime_weights: Dict[str, float]
    rationale: Dict[str, Any]
    timestamp: str

# Global variables for model components
model = None
preprocessor = None
feature_names = None
lime_explainer = None

# Database setup
def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect('loan_predictions.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            age REAL,
            income REAL,
            loan_amount REAL,
            debt_to_income REAL,
            employment_years REAL,
            credit_history_years REAL,
            num_delinquencies INTEGER,
            num_existing_loans INTEGER,
            purpose_of_loan TEXT,
            collateral TEXT,
            residence_status TEXT,
            probability REAL,
            decision TEXT,
            threshold_used REAL,
            lime_weights TEXT,
            llm_rationale TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def generate_synthetic_data(n_samples=1000):
    """Generate synthetic loan data for training"""
    np.random.seed(42)
    
    data = []
    
    for _ in range(n_samples):
        # Generate correlated features that make business sense
        age = np.random.normal(35, 10)
        age = max(18, min(80, age))
        
        # Income tends to increase with age (to a point)
        income_base = 30000 + (age - 25) * 1000 + np.random.normal(0, 15000)
        income = max(20000, income_base)
        
        loan_amount = np.random.uniform(5000, 100000)
        
        # Debt to income ratio - lower is better
        debt_to_income = np.random.beta(2, 5)  # Skewed toward lower values
        
        employment_years = min(age - 18, np.random.exponential(5))
        credit_history_years = min(employment_years, np.random.exponential(4))
        
        num_delinquencies = np.random.poisson(0.5)
        num_existing_loans = np.random.poisson(1.2)
        
        purpose_of_loan = np.random.choice(['education', 'auto', 'business', 'home'])
        collateral = np.random.choice(['none', 'vehicle', 'property', 'savings'])
        residence_status = np.random.choice(['rent', 'own', 'mortgage', 'other'])
        
        # Calculate eligibility based on business rules with some noise
        score = 0
        score += min((income - 25000) / 50000, 1) * 0.3  # Income factor
        score += min(employment_years / 10, 1) * 0.2  # Employment stability
        score += min(credit_history_years / 10, 1) * 0.15  # Credit history
        score -= debt_to_income * 0.4  # Debt burden (negative)
        score -= (loan_amount / income) * 0.3 if income > 0 else 0.5  # Loan to income ratio
        score -= min(num_delinquencies / 3, 1) * 0.2  # Delinquency penalty
        score -= min(num_existing_loans / 5, 1) * 0.1  # Existing loans penalty
        
        # Collateral bonus
        if collateral == 'property':
            score += 0.15
        elif collateral == 'savings':
            score += 0.1
        elif collateral == 'vehicle':
            score += 0.05
        
        # Add some randomness
        score += np.random.normal(0, 0.1)
        
        # Convert to binary outcome
        eligible = 1 if score > 0.3 else 0
        
        data.append({
            'age': age,
            'income': income,
            'loan_amount': loan_amount,
            'debt_to_income': debt_to_income,
            'employment_years': employment_years,
            'credit_history_years': credit_history_years,
            'num_delinquencies': num_delinquencies,
            'num_existing_loans': num_existing_loans,
            'purpose_of_loan': purpose_of_loan,
            'collateral': collateral,
            'residence_status': residence_status,
            'eligible': eligible
        })
    
    return pd.DataFrame(data)

def train_model():
    """Train the logistic regression model with preprocessing"""
    global model, preprocessor, feature_names, lime_explainer
    
    print("Generating synthetic training data...")
    df = generate_synthetic_data(1000)
    
    # Separate features and target
    feature_columns = [
        'age', 'income', 'loan_amount', 'debt_to_income',
        'employment_years', 'credit_history_years', 'num_delinquencies',
        'num_existing_loans', 'purpose_of_loan', 'collateral', 'residence_status'
    ]
    
    X = df[feature_columns]
    y = df['eligible']
    
    # Define preprocessing
    numeric_features = [
        'age', 'income', 'loan_amount', 'debt_to_income',
        'employment_years', 'credit_history_years', 'num_delinquencies',
        'num_existing_loans'
    ]
    categorical_features = ['purpose_of_loan', 'collateral', 'residence_status']
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Fit preprocessor and transform training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_processed, y_train)
    
    # Get feature names after preprocessing
    numeric_feature_names = numeric_features
    categorical_feature_names = []
    
    # Get categorical feature names from OneHotEncoder
    cat_encoder = preprocessor.named_transformers_['cat']
    for i, feature in enumerate(categorical_features):
        categories = cat_encoder.categories_[i]
        # OneHotEncoder with drop='first' drops the first category
        for cat in categories[1:]:
            categorical_feature_names.append(f"{feature}_{cat}")
    
    feature_names = numeric_feature_names + categorical_feature_names
    
    # Evaluate model
    y_pred = model.predict(X_test_processed)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained successfully! Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Initialize LIME explainer
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_processed,
        feature_names=feature_names,
        class_names=['Not Eligible', 'Eligible'],
        mode='classification'
    )
    
    print("LIME explainer initialized successfully!")

def get_lime_explanation(input_data):
    """Get LIME explanation for a single prediction"""
    global lime_explainer, model, preprocessor
    
    # Preprocess the input
    input_processed = preprocessor.transform(input_data)
    
    # Get LIME explanation
    explanation = lime_explainer.explain_instance(
        input_processed[0],
        model.predict_proba,
        num_features=len(feature_names)
    )
    
    # Extract feature weights
    lime_weights = {}
    for feature, weight in explanation.as_list():
        lime_weights[feature] = weight
    
    return lime_weights

def call_phi3_llm(prediction_data):
    """Call Phi-3 LLM for rationale generation"""
    # Construct structured prompt for LLM
    prompt = f"""
You are an AI assistant explaining loan decisions. Based on the following data, provide a concise rationale in JSON format.

Prediction Data:
- Decision: {"Eligible" if prediction_data['probability'] >= prediction_data['threshold'] else "Not Eligible"}
- Probability: {prediction_data['probability']:.3f}
- Threshold: {prediction_data['threshold']}

Key Features Impact (LIME weights):
"""
    
    # Add top influencing features
    sorted_weights = sorted(prediction_data['lime_weights'].items(), 
                           key=lambda x: abs(x[1]), reverse=True)
    
    for feature, weight in sorted_weights[:5]:
        impact = "positive" if weight > 0 else "negative"
        prompt += f"- {feature}: {weight:.3f} ({impact} impact)\n"
    
    prompt += """
Please provide a JSON response with this exact structure:
{
  "decision": "Eligible" or "Not Eligible",
  "probability": [probability value],
  "rationale": "[2-3 sentence explanation focusing on key factors]"
}
"""
    
    try:
        # Call Phi-3 API (assuming it's running on port 11434)
        response = requests.post(
            "http://phi3:11434/v1/chat/completions",
            json={
                "model": "phi3",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 200
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            # Try to parse JSON from the response
            try:
                rationale_json = json.loads(content)
                return rationale_json
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                pass
        
    except requests.exceptions.RequestException as e:
        print(f"LLM call failed: {e}")
    
    # Fallback rationale generation
    decision = "Eligible" if prediction_data['probability'] >= prediction_data['threshold'] else "Not Eligible"
    
    # Generate simple rationale based on top features
    top_features = sorted(prediction_data['lime_weights'].items(), 
                         key=lambda x: abs(x[1]), reverse=True)[:3]
    
    if decision == "Not Eligible":
        negative_factors = [f for f, w in top_features if w < 0]
        rationale = f"Application declined primarily due to {', '.join(negative_factors[:2])} factors."
    else:
        positive_factors = [f for f, w in top_features if w > 0]
        rationale = f"Application approved based on strong {', '.join(positive_factors[:2])} indicators."
    
    return {
        "decision": decision,
        "probability": prediction_data['probability'],
        "rationale": rationale
    }

def save_prediction_to_db(application_data, prediction_result):
    """Save prediction results to SQLite database"""
    conn = sqlite3.connect('loan_predictions.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO predictions (
            timestamp, age, income, loan_amount, debt_to_income,
            employment_years, credit_history_years, num_delinquencies,
            num_existing_loans, purpose_of_loan, collateral, residence_status,
            probability, decision, threshold_used, lime_weights, llm_rationale
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        prediction_result['timestamp'],
        application_data.age,
        application_data.income,
        application_data.loan_amount,
        application_data.debt_to_income,
        application_data.employment_years,
        application_data.credit_history_years,
        application_data.num_delinquencies,
        application_data.num_existing_loans,
        application_data.purpose_of_loan,
        application_data.collateral,
        application_data.residence_status,
        prediction_result['probability'],
        prediction_result['decision'],
        prediction_result['threshold'],
        json.dumps(prediction_result['lime_weights']),
        json.dumps(prediction_result['rationale'])
    ))
    
    conn.commit()
    conn.close()

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    init_database()
    train_model()

@app.get("/")
async def read_root():
    """Serve the main application page"""
    return FileResponse("static/index.html")

@app.post("/predict", response_model=PredictionResponse)
async def predict_loan_eligibility(
    application: LoanApplication,
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Decision threshold")
):
    """
    Predict loan eligibility with explainability
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        # Prepare input data
        input_df = pd.DataFrame([application.dict()])
        
        # Get prediction
        input_processed = preprocessor.transform(input_df)
        probability = model.predict_proba(input_processed)[0][1]  # Probability of class 1 (eligible)
        decision = "Eligible" if probability >= threshold else "Not Eligible"
        
        # Get LIME explanation
        lime_weights = get_lime_explanation(input_df)
        
        # Prepare data for LLM
        prediction_data = {
            'probability': probability,
            'threshold': threshold,
            'lime_weights': lime_weights
        }
        
        # Get LLM rationale
        rationale = call_phi3_llm(prediction_data)
        
        # Create response
        result = PredictionResponse(
            decision=decision,
            probability=probability,
            threshold=threshold,
            lime_weights=lime_weights,
            rationale=rationale,
            timestamp=datetime.now().isoformat()
        )
        
        # Save to database
        save_prediction_to_db(application, result.dict())
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/history")
async def get_prediction_history():
    """Get prediction history from database"""
    try:
        conn = sqlite3.connect('loan_predictions.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT 50
        ''')
        
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        
        history = []
        for row in rows:
            record = dict(zip(columns, row))
            # Parse JSON fields
            record['lime_weights'] = json.loads(record['lime_weights'])
            record['llm_rationale'] = json.loads(record['llm_rationale'])
            history.append(record)
        
        conn.close()
        return {"history": history}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "lime_explainer_loaded": lime_explainer is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)