"""
System validation tests
"""
import os
import pandas as pd
import sys

def test_project_structure():
    """Test that essential project files exist"""
    essential_files = [
        "README.md",
        "requirements.txt", 
        "src/api/working_main.py",
        "src/models/final_tune.py",
        "src/data/improve_data.py",
        "portfolio_demo.py"
    ]
    
    for file_path in essential_files:
        assert os.path.exists(file_path), f"Missing essential file: {file_path}"
    print("✅ Project structure validated")

def test_data_structure():
    """Test data files if they exist"""
    if os.path.exists("data/processed/X_test.csv"):
        df = pd.read_csv("data/processed/X_test.csv")
        assert len(df.columns) == 30, f"Expected 30 features, got {len(df.columns)}"
        assert len(df) > 0, "Test data should not be empty"
        print("✅ Data structure validated.")

def test_model_exists():
    """Test that trained model exists"""
    models_dir = "models"
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
        assert len(model_files) > 0, "No trained models found"

        # As a light-weight sanity check on monitoring expectations, we also
        # enforce that at least one "final" or "production" model exists.
        prod_like = [f for f in model_files if "final" in f.lower() or "prod" in f.lower()]
        assert len(prod_like) > 0, "No final/production-like model artefacts found in models directory"

        print(f"✅ Found {len(model_files)} trained modelsa"
              f"({len(prod_like)} marked as final/production)")

def test_requirements():
    """Test that requirements.txt has essential packages"""
    with open("requirements.txt", "r") as f:
        requirements = f.read()
    
    essential_packages = ["fastapi", "scikit-learn", "pandas", "numpy"]
    for package in essential_packages:
        assert package in requirements, f"Missing package in requirements: {package}"
    print("✅ Requirements validated")

if __name__ == "__main__":
    test_project_structure()
    test_data_structure() 
    test_model_exists()
    test_requirements()
    print("\n🎉 All system tests passed!")
