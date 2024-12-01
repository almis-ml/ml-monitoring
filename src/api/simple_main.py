"""
Simplified working API for demonstration
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from glob import glob

app = FastAPI(title="Fraud Detection API", version="1.0.0")

# Load model
model = None
predictions_log = []

def load_model():
    global model
    models_dir = "models"
    final_models = glob(f"{models_dir}/final_tuned_rf_*.joblib")
    if final_models:
        model = joblib.load(sorted(final_models)[-1])
        return True
    return False

load_model()

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
    return {"status": "API Running", "model_loaded": model is not None}

@app.post("/predict")
def predict(transaction: TransactionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Use the data as-is (our model was trained on original format)
        data = [[
            transaction.V1, transaction.V2, transaction.V3, transaction.V4, transaction.V5,
            transaction.V6, transaction.V7, transaction.V8, transaction.V9, transaction.V10,
            transaction.V11, transaction.V12, transaction.V13, transaction.V14, transaction.V15,
            transaction.V16, transaction.V17, transaction.V18, transaction.V19, transaction.V20,
            transaction.V21, transaction.V22, transaction.V23, transaction.V24, transaction.V25,
            transaction.V26, transaction.V27, transaction.V28, transaction.Time, transaction.Amount
        ]]
        
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1]
        
        result = {
            "prediction": int(prediction),
            "probability": float(probability),
            "risk_level": "HIGH" if probability > 0.5 else "LOW",
            "timestamp": datetime.now().isoformat()
        }
        
        predictions_log.append(result)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def stats():
    if not predictions_log:
        return {"total": 0, "fraud_rate": 0.0}
    
    total = len(predictions_log)
    frauds = sum(1 for p in predictions_log if p["prediction"] == 1)
    
    return {
        "total_predictions": total,
        "fraud_rate": frauds / total,
        "recent_predictions": predictions_log[-5:]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Different port
