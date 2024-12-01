"""
Final tuning with improved data and proper techniques
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve, auc
import joblib
from datetime import datetime
import os

def load_improved_data():
    """Load the improved preprocessed data"""
    X_train = pd.read_csv('data/processed/X_train_improved.csv')
    X_test = pd.read_csv('data/processed/X_test_improved.csv')
    y_train = pd.read_csv('data/processed/y_train_improved.csv').values.ravel()
    y_test = pd.read_csv('data/processed/y_test_improved.csv').values.ravel()
    return X_train, X_test, y_train, y_test

def tune_advanced_rf():
    """Advanced Random Forest tuning with focus on imbalanced data"""
    
    X_train, X_test, y_train, y_test = load_improved_data()
    
    print(f"Training on {len(X_train)} samples with {len(X_train.columns)} features")
    print(f"Fraud rate: {y_train.mean():.4f}")
    
    # Try different configurations optimized for imbalanced fraud detection
    configs = [
        {
            'n_estimators': 500,
            'max_depth': 20,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'class_weight': 'balanced',
            'max_features': 'sqrt',
            'name': 'Balanced_Deep'
        },
        {
            'n_estimators': 300,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'class_weight': 'balanced_subsample',
            'max_features': 'log2',
            'name': 'Balanced_Subsample'
        },
        {
            'n_estimators': 200,
            'max_depth': 25,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'class_weight': {0: 1, 1: 100},  # Heavy penalty for missing fraud
            'max_features': 0.8,
            'name': 'High_Penalty'
        }
    ]
    
    best_model = None
    best_score = 0
    best_config = None
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for config in configs:
        print(f"\nTesting configuration: {config['name']}")
        
        # Create model
        model_params = {k: v for k, v in config.items() if k != 'name'}
        model = RandomForestClassifier(random_state=42, **model_params)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        cv_mean = cv_scores.mean()
        
        print(f"CV AUC: {cv_mean:.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train on full training set and test
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Calculate precision-recall AUC (important for imbalanced data)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        print(f"Test AUC-ROC: {test_auc:.4f}")
        print(f"Test AUC-PR: {pr_auc:.4f}")
        
        if test_auc > best_score:
            best_model = model
            best_score = test_auc
            best_config = config
            
            # Show feature importance for best model
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("Top 10 most important features:")
            print(feature_importance.head(10))
    
    # Final evaluation of best model
    print(f"\n{'='*50}")
    print(f"BEST MODEL: {best_config['name']}")
    print(f"Best Test AUC: {best_score:.4f}")
    
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save best model
    os.makedirs('models', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'models/final_tuned_rf_{timestamp}.joblib'
    joblib.dump(best_model, model_path)
    print(f"\nBest model saved to: {model_path}")
    
    return best_model, best_score, model_path

if __name__ == "__main__":
    model, score, path = tune_advanced_rf()
