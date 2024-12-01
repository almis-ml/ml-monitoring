"""
Improved data preprocessing for better performance
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def improve_preprocessing():
    """Apply proper preprocessing that should achieve >0.95 AUC"""
    
    # Load raw data
    df = pd.read_csv('data/raw/creditcard.csv')
    print(f"Original dataset: {df.shape}")
    print(f"Fraud rate: {df['Class'].mean():.4f}")
    
    # The V1-V28 features are already PCA-normalized, but Amount and Time need work
    
    # Scale Amount feature (log transform often helps)
    df['Amount_scaled'] = StandardScaler().fit_transform(df[['Amount']])
    
    # Transform Time feature to hours/seconds since start
    df['Time_hour'] = (df['Time'] / 3600) % 24  # Hour of day
    df['Time_scaled'] = StandardScaler().fit_transform(df[['Time']])
    
    # Create feature set (drop original Amount and Time)
    feature_cols = [col for col in df.columns if col.startswith('V')] + ['Amount_scaled', 'Time_scaled', 'Time_hour']
    X = df[feature_cols]
    y = df['Class']
    
    print(f"Features used: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols}")
    
    # Proper stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train shape: {X_train.shape}, fraud rate: {y_train.mean():.4f}")
    print(f"Test shape: {X_test.shape}, fraud rate: {y_test.mean():.4f}")
    
    # Save improved data
    os.makedirs('data/processed', exist_ok=True)
    X_train.to_csv('data/processed/X_train_improved.csv', index=False)
    X_test.to_csv('data/processed/X_test_improved.csv', index=False)
    y_train.to_csv('data/processed/y_train_improved.csv', index=False)
    y_test.to_csv('data/processed/y_test_improved.csv', index=False)
    
    print("Improved data saved with '_improved' suffix")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    improve_preprocessing()
