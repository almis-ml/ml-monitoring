"""
Model loading utilities for the API
"""
import joblib
import os
from glob import glob

def load_best_model():
    """Load the most recent best-performing model"""
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        return None, None
    
    # Find the final tuned model (best performance)
    final_models = glob(os.path.join(models_dir, "final_tuned_rf_*.joblib"))
    
    if final_models:
        # Get the most recent one
        latest_model = sorted(final_models)[-1]
        model = joblib.load(latest_model)
        
        model_info = {
            "path": latest_model,
            "type": "Random Forest (Final Tuned)",
            "performance": "AUC: 0.9751"
        }
        
        return model, model_info
    
    return None, None

if __name__ == "__main__":
    model, info = load_best_model()
    if model:
        print(f"Loaded: {info}")
    else:
        print("No model found")
