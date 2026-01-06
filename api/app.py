"""
ClinicFlow ML API - Triage Prediction Service
Flask API that serves the MIMIC-IV trained Random Forest model
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)  # Allow requests from React app

# Load model and features at startup
print("Loading MIMIC-IV model...")
with open('triage_model_mimic_v2.pkl', 'rb') as f:
    model = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

with open('model_metadata_mimic_v2.pkl', 'rb') as f:
    metadata = pickle.load(f)

print(f"✅ Model loaded! Accuracy: {metadata['accuracy']:.1%}")

def transform_patient_data(data):
    """
    Transform React form data into model features
    """
    
    # Extract basic data
    age = data.get('age', 0)
    symptom_severity = data.get('symptomSeverity', 5)
    symptom_duration_hours = data.get('symptomDuration', 1.0)
    heart_rate = data.get('heartRate', 75)
    systolic_bp = data.get('systolicBp', 120)
    diastolic_bp = data.get('diastolicBp', 80)
    temperature = data.get('temperature', 98.6)
    oxygen_saturation = data.get('oxygenSaturation', 98)
    
    # Gender encoding (0=female, 1=male, 2=other)
    gender = data.get('gender', 'other')
    gender_encoded = 0 if gender == 'female' else (1 if gender == 'male' else 2)
    
    # Onset encoding (0=gradual, 1=sudden)
    onset = data.get('symptomOnset', 'gradual')
    onset_encoded = 1 if onset == 'sudden' else 0
    
    # Previous visits
    previous_visits = data.get('previousVisits', 0)
    
    # Red flags
    red_flags = data.get('redFlagSymptoms', [])
    has_red_flag = 1 if (len(red_flags) > 0 and 'None' not in red_flags) else 0
    
    # Chronic conditions
    chronic_conditions = data.get('chronicConditions', [])
    has_chronic_condition = 1 if (len(chronic_conditions) > 0 and 'None' not in chronic_conditions) else 0
    
    # High risk chronic (heart disease, cancer, kidney disease)
    high_risk_conditions = ['Heart Disease', 'Cancer', 'Kidney Disease']
    high_risk_chronic = 1 if any(c in chronic_conditions for c in high_risk_conditions) else 0
    
    # Calculate vital abnormalities
    hr_abnormal = 1 if (heart_rate < 60 or heart_rate > 100) else 0
    bp_abnormal = 1 if (systolic_bp < 90 or systolic_bp > 140 or diastolic_bp < 60 or diastolic_bp > 90) else 0
    temp_abnormal = 1 if (temperature < 97.0 or temperature > 99.5) else 0
    spo2_abnormal = 1 if oxygen_saturation < 95 else 0
    
    vital_abnormalities = hr_abnormal + bp_abnormal + temp_abnormal + spo2_abnormal
    
    # Symptom acuity (combination of severity and red flags)
    symptom_acuity = symptom_severity + (3 if has_red_flag else 0)
    
    # Create feature dictionary
    features = {
        'age': age,
        'symptom_severity': symptom_severity,
        'symptom_duration_hours': symptom_duration_hours,
        'heart_rate': heart_rate,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'temperature': temperature,
        'oxygen_saturation': oxygen_saturation,
        'has_red_flag': has_red_flag,
        'has_chronic_condition': has_chronic_condition,
        'high_risk_chronic': high_risk_chronic,
        'hr_abnormal': hr_abnormal,
        'bp_abnormal': bp_abnormal,
        'temp_abnormal': temp_abnormal,
        'spo2_abnormal': spo2_abnormal,
        'vital_abnormalities': vital_abnormalities,
        'symptom_acuity': symptom_acuity,
        'previous_visits': previous_visits,
        'gender_encoded': gender_encoded,
        'onset_encoded': onset_encoded
    }
    
    # Create DataFrame with correct feature order
    feature_values = [features[fname] for fname in feature_names]
    return pd.DataFrame([feature_values], columns=feature_names)

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'ClinicFlow ML API',
        'model_version': metadata['model_version'],
        'accuracy': f"{metadata['accuracy']:.1%}",
        'critical_accuracy': f"{metadata['critical_accuracy']:.1%}"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict urgency level for a patient
    
    Expected JSON input:
    {
        "age": 65,
        "gender": "male",
        "symptomSeverity": 9,
        "symptomDuration": 2.0,
        "symptomOnset": "sudden",
        "heartRate": 95,
        "systolicBp": 140,
        "diastolicBp": 90,
        "temperature": 98.6,
        "oxygenSaturation": 96,
        "chronicConditions": ["Diabetes", "Hypertension"],
        "previousVisits": 3,
        "redFlagSymptoms": ["Chest pain"]
    }
    
    Returns:
    {
        "urgency_level": 2,
        "urgency_label": "High Priority",
        "probability": [0.05, 0.72, 0.18, 0.03, 0.02],
        "confidence": 0.72
    }
    """
    try:
        # Get patient data from request
        patient_data = request.json
        
        # Transform to model features
        features_df = transform_patient_data(patient_data)
        
        # Make prediction
        urgency_level = int(model.predict(features_df)[0])
        probabilities = model.predict_proba(features_df)[0].tolist()
        confidence = float(max(probabilities))
        
        # Map urgency level to label
        urgency_labels = {
            1: "Critical",
            2: "High Priority",
            3: "Moderate",
            4: "Low Priority",
            5: "Non-Urgent"
        }
        
        return jsonify({
            'urgency_level': urgency_level,
            'urgency_label': urgency_labels.get(urgency_level, "Unknown"),
            'probability': probabilities,
            'confidence': confidence,
            'model_accuracy': metadata['accuracy']
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Error processing prediction'
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
