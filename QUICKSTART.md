# Quick Start Guide - Fraud Detection ML System

<!--
Project: Fraud Detection System using ML
Developer: Molla Samser (Founder) - https://rskworld.in
Designer & Tester: Rima Khatun
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Copyright © 2025 RSK World. All rights reserved.
-->

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/rskworld/fraud-detection-ml.git
cd fraud-detection-ml
```

2. **Create virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Generate Sample Data and Train

```bash
# Generate sample data
python generate_sample_data.py

# Train the model
python main.py
```

### Option 2: Use Jupyter Notebooks

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open and run notebooks in order:
   - `notebooks/01_data_exploration.ipynb`
   - `notebooks/02_feature_engineering.ipynb`
   - `notebooks/03_model_training.ipynb`
   - `notebooks/04_model_evaluation.ipynb`

### Option 3: Run Examples

```bash
python example_usage.py
```

## Using the API

1. **Start the Flask API server:**
```bash
python app.py
```

2. **Test the API:**

**Single Prediction:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 1500.00,
    "user_id": 123,
    "timestamp": "2024-01-15 14:30:00",
    "merchant_category": "online"
  }'
```

**Batch Prediction:**
```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"amount": 50.00, "user_id": 101, "timestamp": "2024-01-15 10:00:00"},
      {"amount": 5000.00, "user_id": 102, "timestamp": "2024-01-15 11:00:00"}
    ]
  }'
```

**Get Explanation:**
```bash
curl -X POST http://localhost:5000/explain \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 1500.00,
    "user_id": 123,
    "timestamp": "2024-01-15 14:30:00"
  }'
```

## Project Structure

```
fraud-detection-ml/
├── data/
│   ├── raw/              # Raw transaction data
│   ├── processed/         # Processed data
│   └── models/            # Saved models
├── src/
│   ├── preprocessing.py  # Data preprocessing
│   ├── feature_engineering.py  # Feature engineering
│   ├── models.py         # ML models
│   ├── evaluation.py     # Model evaluation
│   └── fraud_scorer.py   # Real-time scoring
├── notebooks/            # Jupyter notebooks
├── config/              # Configuration files
├── app.py               # Flask API
├── main.py              # Training script
└── example_usage.py     # Usage examples
```

## Features

- ✅ Transaction data preprocessing
- ✅ Advanced feature engineering
- ✅ Multiple ML algorithms (Random Forest, XGBoost, Neural Networks)
- ✅ Ensemble model support
- ✅ Real-time fraud scoring API
- ✅ Comprehensive evaluation metrics
- ✅ Batch processing support

## Contact

**Founder**: Molla Samser  
**Designer & Tester**: Rima Khatun  
**Website**: https://rskworld.in  
**Email**: help@rskworld.in, support@rskworld.in  
**Phone**: +91 93305 39277

© 2025 RSK World. All rights reserved.

