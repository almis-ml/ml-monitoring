"""
Data drift detection for fraud detection model
"""
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import json
import os

class DriftDetector:
    def __init__(self, reference_data_path='data/processed/X_train_improved.csv'):
        """Initialize with reference training data"""
        self.reference_data = pd.read_csv(reference_data_path)
        self.reference_stats = self._calculate_stats(self.reference_data)
        
    def _calculate_stats(self, data):
        """Calculate statistical properties of data"""
        stats = {}
        for col in data.columns:
            stats[col] = {
                'mean': float(data[col].mean()),
                'std': float(data[col].std()),
                'min': float(data[col].min()),
                'max': float(data[col].max()),
                'median': float(data[col].median()),
                'q25': float(data[col].quantile(0.25)),
                'q75': float(data[col].quantile(0.75))
            }
        return stats
    
    def detect_drift(self, new_data, threshold=0.05):
        """Detect drift using statistical tests"""
        
        if isinstance(new_data, list):
            # Convert predictions log to DataFrame
            features = []
            for pred in new_data:
                if 'features' in pred:
                    features.append(pred['features'])
            
            if not features:
                return {"status": "no_data", "drifted_features": []}
                
            new_df = pd.DataFrame(features)
        else:
            new_df = new_data
        
        # Ensure we have the right columns
        common_cols = set(self.reference_data.columns) & set(new_df.columns)
        if not common_cols:
            return {"status": "no_common_features", "drifted_features": []}
        
        drifted_features = []
        drift_details = {}
        
        for col in common_cols:
            # Kolmogorov-Smirnov test
            ref_values = self.reference_data[col].dropna()
            new_values = new_df[col].dropna()
            
            if len(new_values) < 30:  # Need sufficient samples
                continue
                
            try:
                ks_stat, p_value = stats.ks_2samp(ref_values, new_values)
                
                drift_details[col] = {
                    'ks_statistic': float(ks_stat),
                    'p_value': float(p_value),
                    'is_drifted': p_value < threshold,
                    'reference_mean': float(ref_values.mean()),
                    'new_mean': float(new_values.mean()),
                    'mean_shift': float(abs(new_values.mean() - ref_values.mean()))
                }
                
                if p_value < threshold:
                    drifted_features.append(col)
                    
            except Exception as e:
                drift_details[col] = {'error': str(e)}
        
        drift_summary = {
            'timestamp': datetime.now().isoformat(),
            'total_features_checked': len(common_cols),
            'drifted_features_count': len(drifted_features),
            'drifted_features': drifted_features,
            'drift_ratio': len(drifted_features) / len(common_cols) if common_cols else 0,
            'status': 'drift_detected' if drifted_features else 'no_drift',
            'details': drift_details
        }
        
        return drift_summary
    
    def save_drift_report(self, drift_summary, filepath='monitoring/drift_reports'):
        """Save drift detection report"""
        os.makedirs(filepath, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{filepath}/drift_report_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(drift_summary, f, indent=2)
        
        return filename

if __name__ == "__main__":
    # Test drift detection
    detector = DriftDetector()
    
    # Create some synthetic drift data for testing
    test_data = detector.reference_data.sample(1000).copy()
    
    # Introduce artificial drift in V14 (most important feature)
    test_data['V14'] = test_data['V14'] + 2.0
    
    drift_result = detector.detect_drift(test_data)
    print("Drift Detection Test:")
    print(f"Status: {drift_result['status']}")
    print(f"Drifted features: {drift_result['drifted_features']}")
    
    # Save report
    report_file = detector.save_drift_report(drift_result)
    print(f"Report saved to: {report_file}")
