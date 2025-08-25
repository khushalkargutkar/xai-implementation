-- XAI Loan Eligibility Database Schema
-- SQLite Database Structure

CREATE TABLE IF NOT EXISTS predictions (
    -- Primary key and metadata
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    
    -- Input features (numerical)
    age REAL NOT NULL,
    income REAL NOT NULL,
    loan_amount REAL NOT NULL,
    debt_to_income REAL NOT NULL CHECK (debt_to_income >= 0 AND debt_to_income <= 1),
    employment_years REAL NOT NULL,
    credit_history_years REAL NOT NULL,
    num_delinquencies INTEGER NOT NULL,
    num_existing_loans INTEGER NOT NULL,
    
    -- Input features (categorical)
    purpose_of_loan TEXT NOT NULL CHECK (purpose_of_loan IN ('education', 'auto', 'business', 'home')),
    collateral TEXT NOT NULL CHECK (collateral IN ('none', 'vehicle', 'property', 'savings')),
    residence_status TEXT NOT NULL CHECK (residence_status IN ('rent', 'own', 'mortgage', 'other')),
    
    -- Model outputs
    probability REAL NOT NULL CHECK (probability >= 0 AND probability <= 1),
    decision TEXT NOT NULL CHECK (decision IN ('Eligible', 'Not Eligible')),
    threshold_used REAL NOT NULL CHECK (threshold_used >= 0 AND threshold_used <= 1),
    
    -- Explainability data
    lime_weights TEXT NOT NULL,  -- JSON string containing LIME feature weights
    llm_rationale TEXT NOT NULL  -- JSON string containing LLM explanation
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_decision ON predictions(decision);
CREATE INDEX IF NOT EXISTS idx_predictions_probability ON predictions(probability);

-- Create view for summary statistics
CREATE VIEW IF NOT EXISTS prediction_summary AS
SELECT 
    COUNT(*) as total_predictions,
    SUM(CASE WHEN decision = 'Eligible' THEN 1 ELSE 0 END) as approved_count,
    SUM(CASE WHEN decision = 'Not Eligible' THEN 1 ELSE 0 END) as rejected_count,
    ROUND(AVG(probability), 3) as avg_probability,
    ROUND(AVG(threshold_used), 3) as avg_threshold,
    MIN(timestamp) as first_prediction,
    MAX(timestamp) as last_prediction
FROM predictions;