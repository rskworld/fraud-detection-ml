"""
Flask API for Real-time Fraud Detection
Developer: Molla Samser (Founder) - https://rskworld.in
Designer & Tester: Rima Khatun
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Copyright © 2025 RSK World. All rights reserved.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from src.fraud_scorer import FraudScorer
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Initialize fraud scorer
# In production, load a pre-trained model
scorer = None

def load_scorer():
    """Load the fraud detection model."""
    global scorer
    model_path = os.path.join('data', 'models', 'fraud_model.pkl')
    
    if os.path.exists(model_path):
        scorer = FraudScorer(model_path=model_path, model_type='ensemble')
    else:
        # Use default scorer (will need to be trained)
        scorer = FraudScorer(model_type='ensemble')
        print("Warning: Model not found. Please train the model first.")

# Load scorer on startup
load_scorer()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Fraud Detection API',
        'developer': 'Molla Samser (Founder) - https://rskworld.in',
        'version': '1.0.0'
    })

@app.route('/predict', methods=['POST'])
def predict_fraud():
    """
    Predict fraud for a single transaction.
    
    Expected JSON:
    {
        "amount": 100.50,
        "user_id": "user123",
        "timestamp": "2024-01-01 12:00:00",
        ...
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Predict fraud
        fraud_score = scorer.predict(data, return_probability=True)
        fraud_prediction = int(fraud_score >= scorer.threshold)
        
        # Get explanation
        explanation = scorer.get_fraud_explanation(data)
        
        return jsonify({
            'fraud_probability': float(fraud_score),
            'fraud_prediction': fraud_prediction,
            'risk_level': explanation['risk_level'],
            'threshold': scorer.threshold,
            'explanation': explanation.get('top_features', {})
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict fraud for multiple transactions.
    
    Expected JSON:
    {
        "transactions": [
            {"amount": 100.50, "user_id": "user123", ...},
            {"amount": 200.75, "user_id": "user456", ...}
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'transactions' not in data:
            return jsonify({'error': 'No transactions provided'}), 400
        
        transactions_df = pd.DataFrame(data['transactions'])
        results_df = scorer.score_batch(transactions_df)
        
        # Convert to JSON
        results = results_df.to_dict('records')
        
        return jsonify({
            'count': len(results),
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/explain', methods=['POST'])
def explain_prediction():
    """
    Get detailed explanation for a fraud prediction.
    
    Expected JSON:
    {
        "amount": 100.50,
        "user_id": "user123",
        ...
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        explanation = scorer.get_fraud_explanation(data, top_features=10)
        
        return jsonify(explanation)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/threshold', methods=['GET', 'POST'])
def manage_threshold():
    """
    Get or update the fraud detection threshold.
    
    GET: Returns current threshold
    POST: Updates threshold (expects {"threshold": 0.5})
    """
    if request.method == 'GET':
        return jsonify({
            'threshold': scorer.threshold,
            'description': 'Fraud detection threshold (0-1)'
        })
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            new_threshold = float(data.get('threshold', scorer.threshold))
            scorer.update_threshold(new_threshold)
            
            return jsonify({
                'message': 'Threshold updated successfully',
                'new_threshold': scorer.threshold
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("Starting Fraud Detection API...")
    print("Developer: Molla Samser (Founder) - https://rskworld.in")
    print("Email: help@rskworld.in")
    app.run(host='0.0.0.0', port=5000, debug=True)

