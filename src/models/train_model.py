"""
Model training with MLflow tracking
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
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

def train_model(model_type='random_forest', **params):
    """Train a fraud detection model with MLflow tracking"""
    
    # Set MLflow experiment
    mlflow.set_experiment("fraud_detection")
    
    with mlflow.start_run():
        # Load data
        X_train, X_test, y_train, y_test = load_data()
        
        # Log data info
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("fraud_rate_train", y_train.mean())
        
        # Initialize model
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 10),
                random_state=42,
                class_weight='balanced'  # Handle imbalanced data
            )
        elif model_type == 'logistic_regression':
            model = LogisticRegression(
                C=params.get('C', 1.0),
                random_state=42,
                class_weight='balanced'
            )
        
        # Log model parameters
        mlflow.log_params(model.get_params())
        
        # Train model
        print(f"Training {model_type}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Log metrics
        mlflow.log_metric("auc_score", auc_score)
        
        # Save model
        os.makedirs('models', exist_ok=True)
        model_path = f'models/{model_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib'
        joblib.dump(model, model_path)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(model_path)
        
        # Print results
        print(f"Model trained successfully!")
        print(f"AUC Score: {auc_score:.4f}")
        print(f"Model saved to: {model_path}")
        
        return model, auc_score

if __name__ == "__main__":
    # Train different models
    print("Training Random Forest...")
    rf_model, rf_auc = train_model('random_forest', n_estimators=100, max_depth=10)
    
    print("\nTraining Logistic Regression...")
    lr_model, lr_auc = train_model('logistic_regression', C=1.0)
    
    print(f"\nResults Summary:")
    print(f"Random Forest AUC: {rf_auc:.4f}")
    print(f"Logistic Regression AUC: {lr_auc:.4f}")
