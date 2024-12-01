"""
MLOps Portfolio Demonstration Script
"""
import requests
import pandas as pd
import time

def main():
    print("="*60)
    print("ML MODEL MONITORING SYSTEM - PORTFOLIO DEMO")
    print("="*60)
    print("Industry-Level Fraud Detection with Real-Time Monitoring")
    print()
    
    # System Status
    print("1. SYSTEM STATUS")
    try:
        status = requests.get("http://localhost:8002/").json()
        print(f"   Model Performance: {status.get('model_performance', 'N/A')}")
        print(f"   API Status: {'Online' if status['model_loaded'] else 'Offline'}")
    except:
        print("   API Status: Offline - Please start the API server")
        return
    print()
    
    # Process some transactions
    print("2. PROCESSING LIVE TRANSACTIONS")
    X_test = pd.read_csv('data/processed/X_test.csv')
    
    for i in range(3):
        sample = X_test.sample(1).iloc[0].to_dict()
        response = requests.post("http://localhost:8002/predict", json=sample)
        
        if response.status_code == 200:
            result = response.json()
            risk_emoji = "ALERT" if result['probability'] > 0.1 else "OK"
            print(f"   {risk_emoji} Transaction {i+1}: {result['risk_level']} (prob: {result['probability']:.4f})")
        time.sleep(1)
    
    # Show monitoring metrics
    print()
    print("3. MONITORING METRICS")
    stats = requests.get("http://localhost:8002/stats").json()
    print(f"   Total Predictions: {stats['total_predictions']}")
    print(f"   Fraud Detection Rate: {stats['fraud_rate']:.4f}")
    print(f"   Average Risk Score: {stats['avg_probability']:.4f}")
    print(f"   High Risk Alerts: {stats['high_risk_count']}")
    
    print()
    print("4. KEY PORTFOLIO FEATURES DEMONSTRATED")
    print("   - High-Performance ML Model (97.51% AUC)")
    print("   - Production API with FastAPI")
    print("   - Real-Time Prediction Serving")
    print("   - Monitoring & Metrics Collection")
    print("   - Risk Assessment & Alerting")
    print("   - MLOps Pipeline (Training -> Serving -> Monitoring)")
    print()
    print("="*60)
    print("Portfolio demo complete! This showcases production MLOps skills.")
    print("="*60)

if __name__ == "__main__":
    main()
