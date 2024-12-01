"""
Dataset download and initial setup
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def download_data():
    """Download and prepare the credit card fraud dataset"""
    
    # For now, we'll create a synthetic dataset similar to the real one
    # In production, you'd download from Kaggle or your data source
    
    print("Creating synthetic credit card transaction dataset...")
    
    np.random.seed(42)
    n_samples = 100000
    
    # Create synthetic features (V1-V28 are PCA transformed in real dataset)
    data = {}
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(0, 1, n_samples)
    
    # Add Time and Amount features
    data['Time'] = np.random.randint(0, 172800, n_samples)  # 48 hours in seconds
    data['Amount'] = np.random.lognormal(3, 1.5, n_samples)  # Transaction amounts
    
    # Create fraud labels (highly imbalanced - 0.17% fraud rate)
    fraud_rate = 0.0017
    data['Class'] = np.random.choice([0, 1], n_samples, p=[1-fraud_rate, fraud_rate])
    
    df = pd.DataFrame(data)
    
    # Make fraudulent transactions have different patterns
    fraud_mask = df['Class'] == 1
    df.loc[fraud_mask, 'Amount'] *= 0.3  # Fraudulent transactions tend to be smaller
    
    # Save raw data
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/creditcard.csv', index=False)
    
    print(f"Dataset created: {len(df)} transactions")
    print(f"Fraud rate: {df['Class'].mean():.4f}")
    print(f"Saved to: data/raw/creditcard.csv")
    
    return df

def prepare_data():
    """Load and split the data"""
    df = pd.read_csv('data/raw/creditcard.csv')
    
    # Split features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    print("Data split and saved to data/processed/")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = download_data()
    X_train, X_test, y_train, y_test = prepare_data()
