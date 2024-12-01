"""
XGBoost hyperparameter tuning - often performs well on tabular data
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import mlflow
import mlflow.xgboost
from datetime import datetime
import joblib

def tune_xgboost(X_train, y_train):
    """Tune XGBoost with grid search"""
    
    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'scale_pos_weight': [scale_pos_weight]
    }
    
    # Create model
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='auc',
        use_label_encoder=False
    )
    
    # Grid search
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        xgb_model, param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    print("Tuning XGBoost...")
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def main():
    """XGBoost tuning pipeline"""
    
    # Load data
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    mlflow.set_experiment("fraud_detection_tuning")
    
    with mlflow.start_run(run_name="XGBoost_Tuning"):
        # Tune model
        xgb_model, xgb_params, xgb_cv_score = tune_xgboost(X_train, y_train)
        
        # Evaluate
        y_pred = xgb_model.predict(X_test)
        y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
        
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        auc_pr = average_precision_score(y_test, y_pred_proba)
        
        print(f"\nXGBoost Results:")
        print(f"CV AUC-ROC: {xgb_cv_score:.4f}")
        print(f"Test AUC-ROC: {auc_roc:.4f}")
        print(f"Test AUC-PR: {auc_pr:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Log to MLflow
        mlflow.log_params(xgb_params)
        mlflow.log_metric("cv_auc_roc", xgb_cv_score)
        mlflow.log_metric("test_auc_roc", auc_roc)
        mlflow.log_metric("test_auc_pr", auc_pr)
        mlflow.xgboost.log_model(xgb_model, "model")
        
        # Save model if it's good
        if auc_roc > 0.85:  # Only save if performance is good
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f'models/best_xgboost_{timestamp}.joblib'
            joblib.dump(xgb_model, model_path)
            print(f"Model saved to: {model_path}")
            return xgb_model, model_path
        
        return xgb_model, None

if __name__ == "__main__":
    model, path = main()
