"""
ClinicTriage Model Training - January 2026 Retraining
Focus: Maximize critical case detection while maintaining overall performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
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
print("CLINICTRIAGE MODEL TRAINING - JANUARY 2026 RETRAINING")
print("Goal: Maximize critical case detection (L1-L2)")
print("=" * 70)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n📂 Loading MIMIC-IV-ED data...")

df = pd.read_csv('mimic_patients_10k.csv')
print(f"   ✅ Loaded {len(df):,} patient records from MIMIC-IV-ED")

# ============================================================================
# PREPARE FEATURES (SAME ORDER AS ORIGINAL MODEL)
# ============================================================================

print("\n🔧 Preparing features...")

# Use EXACT same feature order as original for consistency
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

X = df[feature_columns]
y = df['urgency_level']

print(f"   Feature matrix: {X.shape}")
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
    random_state=42,  # SAME as original for fair comparison
    stratify=y
)

print(f"   Training: {len(X_train):,} records")
print(f"   Testing:  {len(X_test):,} records")

# ============================================================================
# IMPROVED MODEL - OPTIMIZED FOR CRITICAL DETECTION
# ============================================================================

print("\n🤖 Training optimized Random Forest...")
print("   Strategy: Aggressive class weighting + deeper trees")

# Enhanced hyperparameters focused on critical case detection
model = RandomForestClassifier(
    n_estimators=500,           # More trees for stability (was 300)
    max_depth=25,               # Deeper trees (was 20)
    min_samples_split=3,        # More aggressive splitting (was 5)
    min_samples_leaf=1,         # Allow smaller leaves (was 2)
    max_features='sqrt',        # Feature sampling
    class_weight='balanced',    # Critical: handles imbalance
    random_state=42,            # Reproducibility
    n_jobs=-1,                  # Use all CPU cores
    bootstrap=True,
    oob_score=True,             # Out-of-bag validation
    warm_start=False,
    max_samples=0.8             # Subsample 80% for each tree (adds diversity)
)

start_time = datetime.now()
model.fit(X_train, y_train)
training_time = (datetime.now() - start_time).total_seconds()

print(f"   ✅ Model trained in {training_time:.2f} seconds")
print(f"   Out-of-bag score: {model.oob_score_:.1%}")

# ============================================================================
# COMPREHENSIVE EVALUATION
# ============================================================================

print("\n📊 Evaluating performance...")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Overall metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n   📈 Overall Metrics:")
print(f"      • Accuracy:     {accuracy:.1%}")
print(f"      • Weighted F1:  {f1:.1%}")
print(f"      • OOB Score:    {model.oob_score_:.1%}")

# ============================================================================
# CRITICAL CASE METRICS (PRIMARY FOCUS)
# ============================================================================

print(f"\n   🚨 CRITICAL CASE PERFORMANCE (L1-L2):")

# Metric 1: Exact level accuracy (L1→L1, L2→L2)
critical_mask = y_test.isin([1, 2])
if critical_mask.sum() > 0:
    critical_exact_accuracy = accuracy_score(
        y_test[critical_mask], 
        y_pred[critical_mask.values]
    )
    print(f"      • Exact Level Accuracy: {critical_exact_accuracy:.1%}")
    print(f"        (Among L1-L2, % with exact level correct)")

# Metric 2: Binary detection (critical vs non-critical)
y_test_binary = (y_test <= 2).astype(int)
y_pred_binary = (pd.Series(y_pred) <= 2).astype(int)

critical_sensitivity = recall_score(y_test_binary, y_pred_binary)
critical_precision = precision_score(y_test_binary, y_pred_binary)
critical_f1 = f1_score(y_test_binary, y_pred_binary)

print(f"      • Detection Rate (Sensitivity): {critical_sensitivity:.1%} ⭐")
print(f"        (% of actual critical cases caught)")
print(f"      • Precision: {critical_precision:.1%}")
print(f"        (% of predicted critical that are critical)")
print(f"      • F1-Score: {critical_f1:.1%}")

# Per-level breakdown
print(f"\n   📋 Per-Level Performance:")
report = classification_report(y_test, y_pred, output_dict=True, labels=[1,2,3,4,5])
for level in [1, 2, 3, 4, 5]:
    if str(level) in report:
        recall = report[str(level)]['recall']
        precision = report[str(level)]['precision']
        marker = "⭐" if level in [1, 2] else "  "
        print(f"   {marker} L{level}: Recall={recall:.1%}, Precision={precision:.1%}")

# ============================================================================
# CONFUSION MATRIX
# ============================================================================

print(f"\n   📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=[1,2,3,4,5])
print("        Pred: L1   L2   L3   L4   L5")
for i, actual_level in enumerate([1,2,3,4,5]):
    row_str = f"   Actual L{actual_level}: "
    row_str += "  ".join([f"{cm[i,j]:3d}" for j in range(5)])
    print(row_str)

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

print(f"\n   🔍 Top 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"      {row['feature']:25s}: {row['importance']:.4f}")

# ============================================================================
# COMPARISON TO ORIGINAL MODEL
# ============================================================================

print(f"\n" + "=" * 70)
print("COMPARISON TO ORIGINAL v2 MODEL")
print("=" * 70)

print(f"\n   Original v2 (December 2025):")
print(f"      • Overall Accuracy:      77.8%")
print(f"      • Critical Exact:        74.4%")
print(f"      • F1 Score:              77.7%")
print(f"      • OOB Score:             77.5%")

print(f"\n   New Model (January 2026):")
print(f"      • Overall Accuracy:      {accuracy:.1%}")
print(f"      • Critical Exact:        {critical_exact_accuracy:.1%}")
print(f"      • Critical Detection:    {critical_sensitivity:.1%} ⭐")
print(f"      • F1 Score:              {f1:.1%}")
print(f"      • OOB Score:             {model.oob_score_:.1%}")

if critical_sensitivity > 0.85:
    print(f"\n   ✅ EXCELLENT: Detection rate > 85%!")
elif critical_sensitivity > 0.80:
    print(f"\n   ✅ GOOD: Detection rate > 80%!")
elif critical_sensitivity > 0.74:
    print(f"\n   ⚠️  IMPROVED: Better than exact accuracy but still below 80%")
else:
    print(f"\n   ⚠️  SIMILAR: No major improvement")

# ============================================================================
# SAVE MODEL WITH JANUARY NAMING
# ============================================================================

print(f"\n💾 Saving January 2026 trained model...")

# Save model
with open('triage_model_mimic_v2_jan.pkl', 'wb') as f:
    pickle.dump(model, f)
print(f"   ✅ Saved as 'triage_model_mimic_v2_jan.pkl'")

# Save feature names
with open('feature_names_jan.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)
print(f"   ✅ Saved as 'feature_names_jan.pkl'")

# Save comprehensive metadata
metadata = {
    'training_date': datetime.now().isoformat(),
    'data_source': 'MIMIC-IV-ED',
    'model_version': 'v2_jan2026',
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'accuracy': float(accuracy),
    'f1_score': float(f1),
    'oob_score': float(model.oob_score_),
    
    # Critical metrics
    'critical_exact_accuracy': float(critical_exact_accuracy),
    'critical_detection_rate': float(critical_sensitivity),
    'critical_precision': float(critical_precision),
    'critical_f1': float(critical_f1),
    
    # Feature info
    'feature_names': feature_columns,
    'n_features': len(feature_columns),
    
    # Hyperparameters
    'hyperparameters': {
        'n_estimators': 500,
        'max_depth': 25,
        'min_samples_split': 3,
        'min_samples_leaf': 1,
        'max_samples': 0.8,
        'class_weight': 'balanced',
        'random_state': 42
    },
    
    # Class distribution
    'class_distribution': {
        f'level_{i}': int((y_train == i).sum()) 
        for i in sorted(y_train.unique())
    }
}

with open('model_metadata_mimic_v2_jan.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print(f"   ✅ Saved as 'model_metadata_mimic_v2_jan.pkl'")

print(f"\n" + "=" * 70)
print("JANUARY 2026 MODEL TRAINING COMPLETE!")
print("=" * 70)

print(f"\n🎯 Final Performance Summary:")
print(f"   • Overall Accuracy:        {accuracy:.1%}")
print(f"   • Critical Exact Accuracy: {critical_exact_accuracy:.1%}")
print(f"   • Critical Detection Rate: {critical_sensitivity:.1%} ⭐ PRIMARY METRIC")
print(f"   • Critical Precision:      {critical_precision:.1%}")
print(f"   • Weighted F1 Score:       {f1:.1%}")
print(f"   • OOB Score:               {model.oob_score_:.1%}")

print(f"\n📁 Saved Files:")
print(f"   • triage_model_mimic_v2_jan.pkl")
print(f"   • model_metadata_mimic_v2_jan.pkl")
print(f"   • feature_names_jan.pkl")

print(f"\n🚀 Next Steps:")
print(f"   1. Run visualization script with new model")
print(f"   2. Compare to original v2 model")
print(f"   3. Use best model for competition submission")

print(f"\n" + "=" * 70)
