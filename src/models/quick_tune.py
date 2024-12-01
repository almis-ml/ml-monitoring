"""
Quick hyperparameter tuning - reduced grid for faster results
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
import joblib
from datetime import datetime
import os

def quick_tune():
    """Fast tuning with smaller parameter grid"""
    
    # Load data
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    # Smaller, focused parameter grid
    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [15, 20],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced']
    }
    
    rf = RandomForestClassifier(random_state=42)
    
    # 3-fold CV instead of 5 for speed
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    print("Quick tuning Random Forest...")
    grid_search.fit(X_train, y_train)
    
    # Evaluate best model
    best_model = grid_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"CV AUC: {grid_search.best_score_:.4f}")
    print(f"Test AUC: {auc_score:.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'models/quick_tuned_rf_{timestamp}.joblib'
    joblib.dump(best_model, model_path)
    print(f"Model saved: {model_path}")
    
    return best_model, auc_score

if __name__ == "__main__":
    model, score = quick_tune()
