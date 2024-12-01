# ML Model Monitoring System

A production-ready fraud detection system with real-time monitoring capabilities, achieving **97.51% AUC** performance on the Kaggle Credit Card Fraud Detection dataset.

[![CI/CD Pipeline](https://github.com/jingxuan091/ml-monitoring-system/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/jingxuan091/ml-monitoring-system/actions/workflows/ci-cd.yml)

## Overview

This project demonstrates end-to-end MLOps capabilities including model training, hyperparameter optimization, API serving, and real-time monitoring. The system processes credit card transactions and provides fraud risk assessments with comprehensive monitoring infrastructure.

## System Status

- **CI/CD Pipeline**: Passing all automated tests
- **Model Performance**: 97.51% AUC verified and validated  
- **API Status**: Production-ready FastAPI serving predictions
- **Monitoring**: Real-time drift detection and alerting active
- **Documentation**: Comprehensive technical and business documentation

## Key Features

- **High-Performance Model**: Random Forest achieving 97.51% AUC-ROC
- **Production API**: FastAPI with real-time prediction serving (<100ms)
- **Real-Time Monitoring**: Live dashboard with statistical drift detection
- **Complete MLOps Pipeline**: Data processing through production deployment
- **Automated Testing**: CI/CD pipeline with comprehensive validation

## Performance Metrics

| Metric | Value | Industry Standard |
|--------|-------|-------------------|
| AUC-ROC | **97.51%** | >95% |
| AUC-PR | 85.83% | >80% |
| API Latency | <100ms | <200ms |
| Dataset Size | 284,807 transactions | Production scale |
| Fraud Detection Rate | 0.172% | Realistic imbalance |

## Quick Start

### Prerequisites
- Python 3.10+
- Virtual environment
- Git

### Installation

    # Clone repository
    git clone https://github.com/jingxuan091/ml-monitoring-system.git
    cd ml-monitoring-system

    # Create virtual environment
    python -m venv venv
    source venv/bin/activate

    # Install dependencies
    pip install -r requirements.txt

### Data Setup
1. Download the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place `creditcard.csv` in `data/raw/`
3. Run preprocessing: `python src/data/improve_data.py`

### Model Training

    # Train the high-performance model (achieves 97.51% AUC)
    python src/models/final_tune.py

### Start Production System

    # Start API server
    python src/api/working_main.py

    # In another terminal: Run monitoring dashboard
    python src/monitoring/live_dashboard.py

    # Run complete portfolio demonstration
    python portfolio_demo.py

## Project Structure

    ml-monitoring-system/
    ├── src/
    │   ├── models/           # Model training and hyperparameter optimization
    │   ├── api/              # Production FastAPI serving
    │   ├── monitoring/       # Drift detection and performance tracking
    │   └── data/             # ETL pipeline and preprocessing
    ├── data/
    │   ├── raw/              # Original Kaggle dataset
    │   └── processed/        # Processed training/test data
    ├── models/               # Trained model artifacts (.joblib files)
    ├── tests/                # Automated testing suite
    ├── .github/workflows/    # CI/CD pipeline configuration
    └── requirements.txt      # Project dependencies

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | System status and model performance info |
| `/predict` | POST | Single transaction fraud prediction |
| `/stats` | GET | Real-time monitoring metrics and statistics |

### Example API Usage

    import requests

    # Single fraud prediction
    transaction = {
        "V1": -1.3598071, "V2": -0.0727812, "V3": 2.536347,
        "V4": 1.378155, "V5": -0.338321, "V6": 0.462388,
        # ... include all V1-V28 features
        "Time": 0, "Amount": 149.62
    }

    response = requests.post("http://localhost:8002/predict", json=transaction)
    print(response.json())
    # Output: {"prediction": 0, "probability": 0.0234, "risk_level": "LOW"}

## System Demonstration

### Live Demo Results

    ============================================================
    ML MODEL MONITORING SYSTEM - PORTFOLIO DEMO
    ============================================================
    1. SYSTEM STATUS
       Model Performance: AUC: 0.9751
       API Status: Online

    2. PROCESSING LIVE TRANSACTIONS
       ✅ Transaction 1: MINIMAL (prob: 0.0023)
       ✅ Transaction 2: LOW (prob: 0.0156) 
       ✅ Transaction 3: MINIMAL (prob: 0.0033)

    3. MONITORING METRICS
       Total Predictions: 19
       Fraud Rate: 0.0000 (normal operations)
       Average Risk Score: 0.0056 (baseline)
       High Risk Alerts: 0 (system stable)

    ✅ System operating normally

## Technical Implementation

### Model Development
The 97.51% AUC performance was achieved through:

1. **Feature Engineering**: Scaling Time/Amount features, creating temporal features
2. **Advanced Hyperparameter Tuning**: Grid search with 5-fold stratified cross-validation
3. **Class Imbalance Handling**: `balanced_subsample` weighting for 0.172% fraud rate
4. **Model Selection**: Random Forest with 500 estimators, max_depth=15

### Monitoring Features
- **Statistical Drift Detection**: Kolmogorov-Smirnov tests with configurable thresholds
- **Real-time Performance Tracking**: Fraud rate, confidence scores, alert generation
- **Population Stability Index**: Quantitative drift measurement
- **Automated Reporting**: JSON-formatted drift reports with statistical details

### Production Architecture
- **FastAPI Framework**: Modern async web framework with automatic documentation
- **Model Serving**: Joblib serialization with preprocessing pipeline
- **Error Handling**: Comprehensive exception management and logging
- **Health Monitoring**: System status endpoints and performance metrics

## Testing and Validation

### Automated Testing
- **CI/CD Pipeline**: GitHub Actions with automated test execution
- **Unit Testing**: pytest framework with comprehensive test coverage
- **Integration Testing**: End-to-end workflow validation
- **Performance Testing**: API latency and throughput validation

### Model Validation
- **Cross-Validation**: 5-fold stratified validation with stable performance
- **Hold-out Testing**: Final evaluation on unseen test data
- **Statistical Significance**: Performance metrics with confidence intervals
- **Feature Importance**: Analysis of predictive factors

## Business Impact

This system addresses critical fraud detection needs:
- **Real-time Risk Assessment**: Sub-100ms prediction latency enables instant decisions
- **High Accuracy**: 97.51% AUC meets industry standards for production deployment
- **Scalable Architecture**: Production-ready design supporting high-volume transactions
- **Operational Monitoring**: Automated drift detection ensures system reliability

## CI/CD Pipeline

- **Automated Testing**: Runs on every commit and pull request
- **Code Quality**: Style validation and dependency checking
- **Deployment Validation**: Project structure and functionality verification
- **Continuous Integration**: Ensures system stability across development cycles

## Future Enhancements

- **Cloud Deployment**: Containerization with Docker and Kubernetes orchestration
- **Advanced Monitoring**: Enhanced drift detection with multiple statistical methods
- **Model Ensemble**: Multiple algorithm combination for improved performance
- **A/B Testing Framework**: Systematic model comparison and deployment strategies

## Documentation

- **Technical Report**: Comprehensive implementation analysis and methodology
- **API Documentation**: Automatic OpenAPI documentation with interactive examples
- **Portfolio Summary**: Business impact and technical achievement overview
- **Deployment Guide**: Step-by-step production deployment instructions

## License

This project is for portfolio demonstration purposes showcasing production-level MLOps capabilities.

---

**Repository**: https://github.com/jingxuan091/ml-monitoring-system  
**Live Demo**: Run `python portfolio_demo.py` for complete system demonstration
