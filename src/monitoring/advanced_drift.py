"""
Advanced drift detection with automated alerting
"""
import pandas as pd
import numpy as np
from scipy import stats
import json
import os
from datetime import datetime, timedelta
import requests
import time

class AdvancedDriftDetector:
    def __init__(self, api_url="http://localhost:8002", alert_threshold=0.05):
        self.api_url = api_url
        self.alert_threshold = alert_threshold
        self.reference_data = pd.read_csv('data/processed/X_train_improved.csv')
        
    def collect_live_data(self, hours_back=1):
        """Collect recent predictions from API"""
        try:
            response = requests.get(f"{self.api_url}/monitoring/predictions")
            predictions = response.json()["recent_predictions"]
            
            # Filter to recent data
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            recent_predictions = []
            
            for pred in predictions:
                pred_time = datetime.fromisoformat(pred["timestamp"].replace('Z', '+00:00'))
                if pred_time > cutoff_time:
                    recent_predictions.append(pred["features"])
            
            return pd.DataFrame(recent_predictions) if recent_predictions else None
            
        except Exception as e:
            print(f"Error collecting live data: {e}")
            return None
    
    def detect_drift(self, live_data):
        """Advanced drift detection with multiple methods"""
        if live_data is None or len(live_data) < 30:
            return {"status": "insufficient_data", "message": "Need at least 30 samples"}
        
        # Common features between reference and live data
        common_features = set(self.reference_data.columns) & set(live_data.columns)
        
        drift_results = {
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(live_data),
            "features_tested": len(common_features),
            "drift_detected": False,
            "alerts": [],
            "feature_drift": {}
        }
        
        for feature in common_features:
            ref_values = self.reference_data[feature].dropna()
            live_values = live_data[feature].dropna()
            
            if len(live_values) < 10:
                continue
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.ks_2samp(ref_values, live_values)
            
            # Mann-Whitney U test
            mw_stat, mw_p = stats.mannwhitneyu(ref_values, live_values, alternative='two-sided')
            
            # Population Stability Index
            psi = self.calculate_psi(ref_values, live_values)
            
            feature_result = {
                "ks_statistic": float(ks_stat),
                "ks_p_value": float(ks_p),
                "mw_p_value": float(mw_p),
                "psi": float(psi),
                "is_drifted": ks_p < self.alert_threshold or psi > 0.2,
                "reference_mean": float(ref_values.mean()),
                "live_mean": float(live_values.mean()),
                "mean_shift_pct": float(abs(live_values.mean() - ref_values.mean()) / abs(ref_values.mean()) * 100)
            }
            
            drift_results["feature_drift"][feature] = feature_result
            
            # Generate alerts
            if feature_result["is_drifted"]:
                drift_results["drift_detected"] = True
                alert_msg = f"DRIFT ALERT: {feature} - KS p-value: {ks_p:.4f}, PSI: {psi:.4f}"
                drift_results["alerts"].append(alert_msg)
        
        return drift_results
    
    def calculate_psi(self, expected, actual, buckets=10):
        """Calculate Population Stability Index"""
        try:
            # Create buckets based on expected distribution
            breakpoints = np.linspace(expected.min(), expected.max(), buckets + 1)
            breakpoints[-1] += 0.001  # Ensure last value is included
            
            expected_percents = pd.cut(expected, breakpoints).value_counts() / len(expected)
            actual_percents = pd.cut(actual, breakpoints).value_counts() / len(actual)
            
            # Handle zero values
            expected_percents = expected_percents.replace(0, 0.0001)
            actual_percents = actual_percents.replace(0, 0.0001)
            
            # Calculate PSI
            psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
            return psi
            
        except Exception:
            return 0.0
    
    def save_drift_report(self, drift_results):
        """Save drift detection report"""
        os.makedirs("logs/drift_reports", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/drift_reports/drift_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(drift_results, f, indent=2)
        
        return filename
    
    def run_monitoring_loop(self, check_interval_minutes=5):
        """Run continuous drift monitoring"""
        print(f"Starting drift monitoring (checking every {check_interval_minutes} minutes)...")
        
        while True:
            try:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking for drift...")
                
                # Collect recent data
                live_data = self.collect_live_data(hours_back=1)
                
                if live_data is not None:
                    # Detect drift
                    drift_results = self.detect_drift(live_data)
                    
                    # Report results
                    print(f"Samples analyzed: {drift_results['total_samples']}")
                    print(f"Features tested: {drift_results['features_tested']}")
                    
                    if drift_results["drift_detected"]:
                        print("🚨 DRIFT DETECTED!")
                        for alert in drift_results["alerts"]:
                            print(f"   {alert}")
                        
                        # Save detailed report
                        report_file = self.save_drift_report(drift_results)
                        print(f"   Report saved: {report_file}")
                    else:
                        print("✅ No drift detected")
                else:
                    print("⚠️  Insufficient data for drift analysis")
                
                # Wait for next check
                time.sleep(check_interval_minutes * 60)
                
            except KeyboardInterrupt:
                print("\nStopping drift monitoring...")
                break
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

if __name__ == "__main__":
    detector = AdvancedDriftDetector()
    detector.run_monitoring_loop(check_interval_minutes=2)
