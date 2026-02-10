"""
Live monitoring dashboard for fraud detection
"""
import requests
import time
import pandas as pd
from datetime import datetime

class FraudMonitor:
    def __init__(self, api_url="http://localhost:8002"):
        self.api_url = api_url
        
    def generate_test_traffic(self, n_transactions=20):
        """Generate realistic test traffic"""
        X_test = pd.read_csv('data/processed/X_test.csv')
        samples = X_test.sample(n_transactions)
        
        print(f"Generating {n_transactions} test transactions...")
        
        for i, (_, row) in enumerate(samples.iterrows()):
            response = requests.post(f"{self.api_url}/predict", json=row.to_dict())
            if response.status_code == 200:
                result = response.json()
                risk_emoji = "🚨" if result['risk_level'] in ['HIGH', 'CRITICAL'] else "✅"
                print(f"{risk_emoji} Transaction {i+1}: {result['risk_level']} (prob: {result['probability']:.4f})")
            time.sleep(0.3)
    
    def show_dashboard(self):
        """Display current monitoring stats"""
        stats = requests.get(f"{self.api_url}/stats").json()

        # Derive a simple risk stability indicator so operators can quickly
        # understand whether probabilities are staying close to baseline.
        # This expects the stats endpoint to optionally expose a rolling
        # baseline average for comparison, but falls back gracefully.
        baseline_avg = stats.get("baseline_avg_probability")
        current_avg = stats["avg_probability"]
        stability_label = "N/A"

        if baseline_avg is not None:
            delta = current_avg - baseline_avg
            abs_delta = abs(delta)
            if abs_delta < 0.01:
                stability_label = "STABLE"
            elif abs_delta < 0.05:
                stability_label = "SLIGHT SHIFT"
            else:
                stability_label = "SIGNIFICANT SHIFT"

        print(f"\n{'='*60}")
        print(f"FRAUD DETECTION MONITORING DASHBOARD")
        print(f"{'='*60}")
        print(f"Model Performance: {stats['model_performance']}")
        print(f"Total Predictions: {stats['total_predictions']}")
        print(f"Fraud Detection Rate: {stats['fraud_rate']:.4f}")
        print(f"Average Risk Score: {current_avg:.4f}")
        print(f"High Risk Alerts: {stats['high_risk_count']}")
        if baseline_avg is not None:
            print(f"Baseline Avg Risk: {baseline_avg:.4f}")
            print(f"Risk Stability: {stability_label}")

        if stats['fraud_rate'] > 0.01:
            print("🚨 ALERT: High fraud rate detected!")
        elif stats['avg_probability'] > 0.1:
            print("⚠️  WARNING: Elevated risk levels")
        else:
            print("✅ System operating normally")

if __name__ == "__main__":
    monitor = FraudMonitor()
    
    # Generate test traffic
    monitor.generate_test_traffic(10)
    
    # Show dashboard
    monitor.show_dashboard()
