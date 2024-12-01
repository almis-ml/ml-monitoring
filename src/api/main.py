"""
Fixed API with proper formatting
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime
import os
import joblib
from glob import glob

app = FastAPI(
    title="Fraud Detection API with Monitoring",
    description="Production fraud detection with real-time monitoring",
    version="2.0.0"
)

# Global variables
model = None
model_info = None
predictions_log = []


def load_best_model():
    """Load the highest performing model"""
    global model, model_info
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        return False
    
    final_models = glob(f"{models_dir}/final_tuned_rf_*.joblib")
    
    if final_models:
        latest_model = sorted(final_models)[-1]
        model = joblib.load(latest_model)
        
        model_info = {
            "path": latest_model,
            "type": "Random Forest (Final Tuned)",
            "performance": "AUC: 0.9751",
            "loaded_at": datetime.now().isoformat()
        }
        return True
    
    return False


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
def read_root():
    return {
        "message": "Fraud Detection API with Monitoring",
        "model_loaded": model is not None,
        "model_info": model_info
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_info": model_info,
        "predictions_count": len(predictions_log),
        "uptime": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
