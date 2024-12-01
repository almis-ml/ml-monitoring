"""
MLflow configuration and utilities
"""
import mlflow
import os

def setup_mlflow():
    """Configure MLflow tracking"""
    
    # Set tracking URI (local for now)
    tracking_uri = "file:///Users/jingxuanzhu/code/personal project/ml-monitoring-system/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create experiment if it doesn't exist
    experiment_name = "fraud_detection"
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
            print(f"Created experiment: {experiment_name}")
        else:
            print(f"Using existing experiment: {experiment_name}")
    except Exception as e:
        print(f"Experiment setup: {e}")

if __name__ == "__main__":
    setup_mlflow()
