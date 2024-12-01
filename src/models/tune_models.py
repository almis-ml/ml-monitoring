"""
Advanced hyperparameter tuning for fraud detection
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from datetime import datetime
import joblib
import os

def load_data():
    """Load processed training data"""
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    return X_train, X_test, y_train, y_test

def tune_random_forest(X_train, y_train):
    """Tune Random Forest with grid search"""
    
    # Parameter grid for Random Forest
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    # Create model
    rf = RandomForestClassifier(random_state=42)
    
    # Grid search with stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, 
        cv=cv, 
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    print("Tuning Random Forest...")
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def tune_logistic_regression(X_train, y_train):
    """Tune Logistic Regression with scaling"""
    
    # Parameter grid for Logistic Regression
    param_grid = {
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear', 'saga'],
        'classifier__class_weight': ['balanced']
    }
    
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    # Grid search
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        pipeline, param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    print("Tuning Logistic Regression...")
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive model evaluation"""
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)
    
    print(f"\n{model_name} Results:")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR: {auc_pr:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return auc_roc, auc_pr

def main():
    """Main tuning pipeline"""
    
    # Set MLflow experiment
    mlflow.set_experiment("fraud_detection_tuning")
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Fraud rate in training: {y_train.mean():.4f}")
    
    best_model = None
    best_score = 0
    best_model_name = None
    
    # Tune Random Forest
    with mlflow.start_run(run_name="RandomForest_Tuning"):
        rf_model, rf_params, rf_cv_score = tune_random_forest(X_train, y_train)
        rf_auc_roc, rf_auc_pr = evaluate_model(rf_model, X_test, y_test, "Random Forest")
        
        # Log to MLflow
        mlflow.log_params(rf_params)
        mlflow.log_metric("cv_auc_roc", rf_cv_score)
        mlflow.log_metric("test_auc_roc", rf_auc_roc)
        mlflow.log_metric("test_auc_pr", rf_auc_pr)
        mlflow.sklearn.log_model(rf_model, "model")
        
        if rf_auc_roc > best_score:
            best_model = rf_model
            best_score = rf_auc_roc
            best_model_name = "RandomForest"
    
    # Tune Logistic Regression
    with mlflow.start_run(run_name="LogisticRegression_Tuning"):
        lr_model, lr_params, lr_cv_score = tune_logistic_regression(X_train, y_train)
        lr_auc_roc, lr_auc_pr = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
        
        # Log to MLflow
        mlflow.log_params(lr_params)
        mlflow.log_metric("cv_auc_roc", lr_cv_score)
        mlflow.log_metric("test_auc_roc", lr_auc_roc)
        mlflow.log_metric("test_auc_pr", lr_auc_pr)
        mlflow.sklearn.log_model(lr_model, "model")
        
        if lr_auc_roc > best_score:
            best_model = lr_model
            best_score = lr_auc_roc
            best_model_name = "LogisticRegression"
    
    # Save best model
    os.makedirs('models', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_path = f'models/best_{best_model_name.lower()}_{timestamp}.joblib'
    joblib.dump(best_model, best_model_path)
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Best AUC-ROC: {best_score:.4f}")
    print(f"Saved to: {best_model_path}")
    
    return best_model, best_model_path

if __name__ == "__main__":
    best_model, model_path = main()
