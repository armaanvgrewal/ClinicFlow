"""
ClinicTriage Model Retraining - Original v2 Parameters
======================================================

This script retrains the model using the ORIGINAL v2 parameters that achieved
89.3% critical detection rate.

IMPORTANT: Metric Definitions
-----------------------------
There are TWO different "critical accuracy" metrics:

1. CRITICAL EXACT ACCURACY (aka "Critical Accuracy")
   Formula: Correct L1+L2 predictions / Total actual L1+L2 cases
   
   Among patients who are ACTUALLY critical (L1 or L2), what percentage 
   did we predict the EXACT correct level?
   
   Example:
   - Patient is actually L1 (Critical)
   - Model predicts L2 (High)
   - Result: WRONG (even though both are urgent categories!)
   
   This is a STRICT measure - L1 must be predicted as L1, L2 as L2.

2. CRITICAL DETECTION RATE (aka "Binary Critical Detection" or "Sensitivity")
   Formula: Detected critical cases / Total actual critical cases
           = TP / (TP + FN)
           = True Positives / (True Positives + False Negatives)
   
   Among patients who are ACTUALLY critical (L1 or L2), what percentage 
   did we correctly identify as CRITICAL (either L1 or L2)?
   
   Example:
   - Patient is actually L1 (Critical)
   - Model predicts L2 (High)
   - Result: CORRECT! (We detected them as critical)
   
   This is what clinically matters - did we FLAG the urgent patient?
   The 89.3% figure refers to THIS metric.

Why the difference matters:
- Critical Detection Rate is what matters clinically (don't miss urgent patients!)
- Critical Exact Accuracy is stricter but less clinically relevant
- Both L1 and L2 patients get urgent attention, so detection > exact level

Author: Generated for A's AI Challenge Project
Date: January 2026
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score
)
import pickle
from datetime import datetime

print("=" * 70)
print("CLINICTRIAGE MODEL RETRAINING - ORIGINAL V2 PARAMETERS")
print("=" * 70)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n📂 Loading MIMIC-IV-ED data...")

# Try multiple possible filenames
data_files = [
    'mimic_patients_10k.csv',
    'mimic_patients_10k_enhanced.csv',
    'clinictriage_data.csv'
]

df = None
for filename in data_files:
    try:
        df = pd.read_csv(filename)
        print(f"   ✅ Loaded {len(df):,} patient records from {filename}")
        break
    except FileNotFoundError:
        continue

if df is None:
    print("   ❌ ERROR: No data file found!")
    print("   Please ensure one of these files exists in the current directory:")
    for f in data_files:
        print(f"      - {f}")
    exit(1)

# ============================================================================
# PREPARE FEATURES
# ============================================================================

print("\n🔧 Preparing features...")

# Standard feature columns (adjust if your dataset has different columns)
feature_columns = [
    'age',
    'symptom_severity',
    'symptom_duration_hours',
    'heart_rate',
    'systolic_bp',
    'diastolic_bp',
    'temperature',
    'oxygen_saturation',
    'has_red_flag',
    'has_chronic_condition',
    'high_risk_chronic',
    'hr_abnormal',
    'bp_abnormal',
    'temp_abnormal',
    'spo2_abnormal',
    'vital_abnormalities',
    'symptom_acuity',
    'previous_visits',
    'gender_encoded',
    'onset_encoded'
]

# Check which features are available
available_features = [col for col in feature_columns if col in df.columns]
missing_features = [col for col in feature_columns if col not in df.columns]

if missing_features:
    print(f"   ⚠️  Missing features (will proceed without): {missing_features}")
    
print(f"   Using {len(available_features)} features")

X = df[available_features]
y = df['urgency_level']

print(f"\n   Feature matrix: {X.shape}")
print(f"   Target variable: {y.shape}")

# Check class distribution
print(f"\n   Class distribution:")
critical_count = 0
for level in sorted(y.unique()):
    count = (y == level).sum()
    pct = count / len(y) * 100
    if level in [1, 2]:
        critical_count += count
    marker = "⭐" if level in [1, 2] else "  "
    print(f"   {marker} Level {int(level)}: {count:4d} ({pct:5.1f}%)")

print(f"\n   Critical cases (L1+L2): {critical_count:,} ({critical_count/len(y)*100:.1f}%)")

# ============================================================================
# TRAIN/TEST SPLIT (SAME AS ORIGINAL)
# ============================================================================

print("\n✂️ Splitting data (stratified, random_state=42)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,  # SAME as original for reproducibility
    stratify=y
)

print(f"   Training: {len(X_train):,} records")
print(f"   Testing:  {len(X_test):,} records")

# ============================================================================
# ORIGINAL V2 MODEL - EXACT PARAMETERS
# ============================================================================

print("\n🤖 Training Random Forest with ORIGINAL v2 parameters...")
print("   (These are the parameters that achieved 89.3% critical detection)")

# ORIGINAL V2 HYPERPARAMETERS
model = RandomForestClassifier(
    n_estimators=200,           # Original v2 value
    max_depth=15,               # Original v2 value
    min_samples_split=10,       # Original v2 value
    min_samples_leaf=5,         # Original v2 value
    max_features='sqrt',        # Original v2 value
    class_weight='balanced',    # CRITICAL: This handles class imbalance
    random_state=42,            # For reproducibility
    n_jobs=-1,                  # Use all CPU cores
    oob_score=True              # Out-of-bag validation
)

print(f"\n   Hyperparameters:")
print(f"   • n_estimators:     200")
print(f"   • max_depth:        15")
print(f"   • min_samples_split: 10")
print(f"   • min_samples_leaf:  5")
print(f"   • max_features:     'sqrt'")
print(f"   • class_weight:     'balanced' ⭐ (handles imbalance)")
print(f"   • random_state:     42")

# Train model
print("\n   Training...")
model.fit(X_train, y_train)
print("   ✅ Training complete!")

# ============================================================================
# PREDICTIONS
# ============================================================================

print("\n📊 Making predictions...")

y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

# ============================================================================
# METRIC 1: OVERALL ACCURACY
# ============================================================================

overall_accuracy = accuracy_score(y_test, y_pred)
overall_accuracy_train = accuracy_score(y_train, y_train_pred)

# ============================================================================
# METRIC 2: CRITICAL EXACT ACCURACY
# ============================================================================

print("\n" + "=" * 70)
print("METRIC: CRITICAL EXACT ACCURACY")
print("=" * 70)

# Among actual critical cases (L1 or L2), what % got the EXACT level correct?
critical_mask_test = y_test.isin([1, 2])
critical_mask_train = y_train.isin([1, 2])

if critical_mask_test.sum() > 0:
    # Get predictions for critical cases only
    y_test_critical = y_test[critical_mask_test]
    y_pred_critical = y_pred[critical_mask_test.values]
    
    critical_exact_accuracy = accuracy_score(y_test_critical, y_pred_critical)
    
    print(f"\n   Formula: Correct L1+L2 predictions / Total actual L1+L2 cases")
    print(f"\n   Among {critical_mask_test.sum()} actual critical patients (L1 or L2):")
    print(f"   → Predicted exact level correctly: {(y_test_critical == y_pred_critical).sum()}")
    print(f"   → Critical Exact Accuracy: {critical_exact_accuracy:.1%}")
    print(f"\n   Note: This counts L1→L2 or L2→L1 predictions as WRONG")

# ============================================================================
# METRIC 3: CRITICAL DETECTION RATE (THE 89.3% METRIC!)
# ============================================================================

print("\n" + "=" * 70)
print("METRIC: CRITICAL DETECTION RATE (Sensitivity/Recall)")
print("=" * 70)

# Binary: critical (1,2) vs non-critical (3,4,5)
y_test_binary = (y_test <= 2).astype(int)      # 1 = critical, 0 = non-critical
y_pred_binary = (pd.Series(y_pred) <= 2).astype(int)

y_train_binary = (y_train <= 2).astype(int)
y_train_pred_binary = (pd.Series(y_train_pred) <= 2).astype(int)

# Calculate metrics
critical_detection_rate = recall_score(y_test_binary, y_pred_binary)
critical_detection_rate_train = recall_score(y_train_binary, y_train_pred_binary)
critical_precision = precision_score(y_test_binary, y_pred_binary)
critical_f1 = f1_score(y_test_binary, y_pred_binary)

# Confusion matrix for binary classification
tn, fp, fn, tp = confusion_matrix(y_test_binary, y_pred_binary).ravel()

print(f"\n   Formula: TP / (TP + FN) = Detected critical / Total actual critical")
print(f"\n   Binary Confusion Matrix (Critical vs Non-Critical):")
print(f"                          Predicted")
print(f"                      Non-Crit  Critical")
print(f"   Actual Non-Crit       {tn:4d}      {fp:4d}")
print(f"   Actual Critical       {fn:4d}      {tp:4d}")

print(f"\n   TP (True Positive - Critical detected):    {tp}")
print(f"   FN (False Negative - Critical MISSED):     {fn}")
print(f"   FP (False Positive - Over-triaged):        {fp}")
print(f"   TN (True Negative - Non-critical correct): {tn}")

print(f"\n   ⭐ CRITICAL DETECTION RATE: {tp} / ({tp} + {fn}) = {critical_detection_rate:.1%}")
print(f"   (This is the 89.3% metric from original training!)")

print(f"\n   Training Detection Rate: {critical_detection_rate_train:.1%}")
print(f"   Test Detection Rate:     {critical_detection_rate:.1%}")

# ============================================================================
# COMPARISON OF BOTH METRICS
# ============================================================================

print("\n" + "=" * 70)
print("COMPARISON: EXACT vs DETECTION")
print("=" * 70)

print(f"""
   ┌─────────────────────────────┬──────────────┬──────────────┐
   │ Metric                      │ Test Set     │ Train Set    │
   ├─────────────────────────────┼──────────────┼──────────────┤
   │ Overall Accuracy            │ {overall_accuracy:10.1%}  │ {overall_accuracy_train:10.1%}  │
   │ Critical Exact Accuracy     │ {critical_exact_accuracy:10.1%}  │      -       │
   │ ⭐ Critical Detection Rate  │ {critical_detection_rate:10.1%}  │ {critical_detection_rate_train:10.1%}  │
   │ Critical Precision          │ {critical_precision:10.1%}  │      -       │
   │ Critical F1 Score           │ {critical_f1:10.1%}  │      -       │
   │ OOB Score                   │ {model.oob_score_:10.1%}  │      -       │
   └─────────────────────────────┴──────────────┴──────────────┘
""")

print(f"   KEY INSIGHT:")
print(f"   → Critical Exact Accuracy ({critical_exact_accuracy:.1%}) < Critical Detection ({critical_detection_rate:.1%})")
print(f"   → Why? Detection allows L1↔L2 confusion, Exact does not")
print(f"   → Clinically, Detection is what matters (don't miss urgent patients!)")

# ============================================================================
# FULL CLASSIFICATION REPORT
# ============================================================================

print("\n" + "=" * 70)
print("FULL CLASSIFICATION REPORT (5-CLASS)")
print("=" * 70)

print(classification_report(y_test, y_pred, target_names=[
    'L1 (Critical)', 'L2 (High)', 'L3 (Moderate)', 'L4 (Low)', 'L5 (Non-Urgent)'
]))

# ============================================================================
# SAVE MODEL AND METADATA
# ============================================================================

print("\n💾 Saving model and metadata...")

# Save model
model_filename = 'triage_model_mimic_v2_original.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)
print(f"   ✅ Model saved as '{model_filename}'")

# Save feature names
feature_filename = 'feature_names_v2_original.pkl'
with open(feature_filename, 'wb') as f:
    pickle.dump(available_features, f)
print(f"   ✅ Features saved as '{feature_filename}'")

# Save comprehensive metadata
metadata = {
    'training_date': datetime.now().isoformat(),
    'model_version': 'v2_original',
    'data_source': 'MIMIC-IV-ED',
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    
    # Overall metrics
    'accuracy': float(overall_accuracy),
    'accuracy_train': float(overall_accuracy_train),
    'f1_score_weighted': float(f1_score(y_test, y_pred, average='weighted')),
    'oob_score': float(model.oob_score_),
    
    # Critical metrics - BOTH definitions
    'critical_exact_accuracy': float(critical_exact_accuracy),
    'critical_detection_rate': float(critical_detection_rate),  # THE 89.3% metric
    'critical_detection_rate_train': float(critical_detection_rate_train),
    'critical_precision': float(critical_precision),
    'critical_f1': float(critical_f1),
    
    # Binary confusion matrix values
    'binary_confusion': {
        'true_positive': int(tp),
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn)
    },
    
    # Feature info
    'feature_names': available_features,
    'n_features': len(available_features),
    
    # ORIGINAL V2 HYPERPARAMETERS
    'hyperparameters': {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'random_state': 42
    },
    
    # Class distribution
    'class_distribution': {
        f'level_{i}': int((y_train == i).sum()) 
        for i in sorted(y_train.unique())
    },
    
    # Metric definitions for reference
    'metric_definitions': {
        'critical_exact_accuracy': 'Among actual critical (L1+L2), % predicted exact level',
        'critical_detection_rate': 'Among actual critical (L1+L2), % detected as critical (L1 or L2) = TP/(TP+FN)'
    }
}

metadata_filename = 'model_metadata_mimic_v2_original.pkl'
with open(metadata_filename, 'wb') as f:
    pickle.dump(metadata, f)
print(f"   ✅ Metadata saved as '{metadata_filename}'")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("🎯 TRAINING COMPLETE - ORIGINAL V2 PARAMETERS")
print("=" * 70)

print(f"""
   Model Files:
   • {model_filename}
   • {metadata_filename}
   • {feature_filename}

   Original v2 Hyperparameters Used:
   • n_estimators:     200
   • max_depth:        15  
   • min_samples_split: 10
   • min_samples_leaf:  5
   • max_features:     'sqrt'
   • class_weight:     'balanced'

   Performance:
   • Overall Accuracy:         {overall_accuracy:.1%}
   • Critical Exact Accuracy:  {critical_exact_accuracy:.1%}
   • ⭐ Critical Detection:    {critical_detection_rate:.1%} (the 89.3% metric!)
   • OOB Score:                {model.oob_score_:.1%}

   METRIC FORMULAS:
   ────────────────
   Critical Exact Accuracy = (L1 predicted as L1) + (L2 predicted as L2)
                             ─────────────────────────────────────────────
                                      Total actual L1 + L2 cases

   Critical Detection Rate = True Positives
                             ────────────────────────────
                             True Positives + False Negatives
                           
                           = Correctly detected critical cases
                             ────────────────────────────────────
                             All actual critical cases
                           
                           = (L1 or L2 predicted for L1 patients) + (L1 or L2 predicted for L2 patients)
                             ─────────────────────────────────────────────────────────────────────────────
                                                 Total actual L1 + L2 cases
""")

print("=" * 70)
