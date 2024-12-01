"""
Production API using the best model with proper preprocessing
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from glob import glob
from sklearn.preprocessing import StandardScaler

app = FastAPI(title="Fraud Detection API", version="1.0.0")

model = None
predictions_log = []

def load_best_model():
    global model
    models_dir = "models"
    # Load our best performing model (0.9751 AUC)
    final_models = glob(f"{models_dir}/final_tuned_rf_*.joblib")
    if final_models:
        model = joblib.load(sorted(final_models)[-1])
        print(f"Loaded best model: {sorted(final_models)[-1]}")
        return True
    return False

def preprocess_transaction(transaction_dict):
    """Convert original transaction to format expected by best model"""
    
    # Load training data statistics for scaling (in production, save these separately)
    train_data = pd.read_csv('data/raw/creditcard.csv')
    
    # Scale Amount
    amount_scaler = StandardScaler()
    amount_scaler.fit(train_data[['Amount']])
    amount_scaled = amount_scaler.transform([[transaction_dict['Amount']]])[0][0]
    
    # Scale Time
    time_scaler = StandardScaler()  
    time_scaler.fit(train_data[['Time']])
    time_scaled = time_scaler.transform([[transaction_dict['Time']]])[0][0]
    
    # Calculate Time_hour
    time_hour = (transaction_dict['Time'] / 3600) % 24
    
    # Create processed feature vector
    v_features = [transaction_dict[f'V{i}'] for i in range(1, 29)]
    processed_features = v_features + [amount_scaled, time_scaled, time_hour]
    
    # Column names matching training data
    feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount_scaled', 'Time_scaled', 'Time_hour']
    
    return pd.DataFrame([processed_features], columns=feature_names)

load_best_model()

class TransactionRequest(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Time: float
    Amount: float

@app.get("/")
def root():
    return {
        "status": "API Running", 
        "model_loaded": model is not None,
        "model_performance": "AUC: 0.9751"
    }

@app.post("/predict")
def predict(transaction: TransactionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess to match training format
        processed_data = preprocess_transaction(transaction.dict())
        
        # Make prediction with best model
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]
        
        # Enhanced risk levels
        if probability >= 0.8:
            risk_level = "CRITICAL"
        elif probability >= 0.5:
            risk_level = "HIGH"
        elif probability >= 0.2:
            risk_level = "MEDIUM"
        elif probability >= 0.05:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        result = {
            "prediction": int(prediction),
            "probability": float(probability),
            "risk_level": risk_level,
            "confidence": float(max(model.predict_proba(processed_data)[0])),
            "timestamp": datetime.now().isoformat(),
            "model_version": "final_tuned_rf_0.9751"
        }
        
        predictions_log.append(result)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/stats")
def stats():
    if not predictions_log:
        return {"total": 0, "fraud_rate": 0.0, "model_performance": "AUC: 0.9751"}
    
    total = len(predictions_log)
    frauds = sum(1 for p in predictions_log if p["prediction"] == 1)
    
    return {
        "total_predictions": total,
        "fraud_rate": frauds / total,
        "avg_probability": np.mean([p["probability"] for p in predictions_log]),
        "high_risk_count": sum(1 for p in predictions_log if p["risk_level"] in ["HIGH", "CRITICAL"]),
        "model_performance": "AUC: 0.9751",
        "recent_predictions": predictions_log[-5:]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)  # New port
