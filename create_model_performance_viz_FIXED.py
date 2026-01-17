"""
ClinicTriage Model Performance Visualization - FIXED VERSION
Generates comprehensive performance metrics visualization for v2 model
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    accuracy_score,
    f1_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

print("="*70)
print("CLINICTRIAGE MODEL PERFORMANCE VISUALIZATION")
print("="*70)

# ============================================================================
# STEP 1: LOAD MODEL AND METADATA
# ============================================================================

print("\n📦 Loading trained model...")

try:
    with open('triage_model_mimic_v2.pkl', 'rb') as f:
        model = pickle.load(f)
    print("   ✓ Model loaded successfully")
except FileNotFoundError:
    print("   ✗ Error: triage_model_mimic_v2.pkl not found!")
    print("   → Make sure you're in the correct directory")
    print("   → Expected location: ~/Documents/ClinicFlow/")
    exit(1)

try:
    with open('model_metadata_mimic_v2.pkl', 'rb') as f:
        metadata = pickle.load(f)
    print("   ✓ Metadata loaded successfully")
    print(f"   → Model version: {metadata.get('model_version', 'v2')}")
    print(f"   → Accuracy: {metadata.get('accuracy', 0):.1%}")
except FileNotFoundError:
    print("   ⚠ Warning: model_metadata_mimic_v2.pkl not found")
    metadata = {}

# ============================================================================
# STEP 2: LOAD AND PREPARE TEST DATA
# ============================================================================

print("\n📊 Loading test data...")

try:
    # Load the processed MIMIC-IV data
    df = pd.read_csv('mimic_patients_10k.csv')
    print(f"   ✓ Loaded {len(df):,} patient records")
except FileNotFoundError:
    print("   ✗ Error: mimic_patients_10k.csv not found!")
    print("   → This file should be in the same directory as the model")
    exit(1)

# ============================================================================
# CRITICAL FIX: Get feature names from the trained model
# ============================================================================

print("\n🔍 Extracting feature names from model...")

# Method 1: Try to get from model object (sklearn >=1.0)
if hasattr(model, 'feature_names_in_'):
    feature_columns = list(model.feature_names_in_)
    print(f"   ✓ Found {len(feature_columns)} features from model.feature_names_in_")
    print(f"   → Features: {feature_columns}")
elif 'feature_names' in metadata:
    feature_columns = metadata['feature_names']
    print(f"   ✓ Found {len(feature_columns)} features from metadata")
    print(f"   → Features: {feature_columns}")
else:
    # Fallback: Try to infer from training
    print("   ⚠ Could not find feature names in model or metadata")
    print("   → Attempting to infer from available columns...")
    
    # Get all numeric columns except target
    potential_features = [col for col in df.columns 
                         if col != 'urgency_level' and df[col].dtype in ['int64', 'float64']]
    
    # Most likely feature set based on training script
    expected_features = [
        'age', 'gender_encoded', 'symptom_severity', 'symptom_duration_hours',
        'onset_encoded', 'heart_rate', 'systolic_bp', 'diastolic_bp',
        'temperature', 'oxygen_saturation', 'has_red_flag', 'has_chronic_condition',
        'high_risk_chronic', 'hr_abnormal', 'bp_abnormal', 'temp_abnormal',
        'spo2_abnormal', 'vital_abnormalities', 'symptom_acuity', 'previous_visits'
    ]
    
    # Use intersection of expected and available
    feature_columns = [f for f in expected_features if f in df.columns]
    
    if len(feature_columns) != len(expected_features):
        missing = set(expected_features) - set(feature_columns)
        print(f"   ⚠ Warning: Missing features: {missing}")
    
    print(f"   → Using {len(feature_columns)} inferred features")

# Verify all features exist in dataframe
missing_cols = [col for col in feature_columns if col not in df.columns]
if missing_cols:
    print(f"\n   ✗ ERROR: Required features not in dataset: {missing_cols}")
    print(f"   → Available columns: {list(df.columns)}")
    exit(1)

# Prepare features and target - CRITICAL: Maintain exact order
X = df[feature_columns].copy()  # Use copy to avoid warnings
y = df['urgency_level'].copy()

print(f"   ✓ Feature matrix shape: {X.shape}")
print(f"   ✓ Target shape: {y.shape}")

# Use same train/test split (80/20 with same random state as training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   → Training set: {len(X_train):,} samples")
print(f"   → Test set: {len(X_test):,} samples")

# ============================================================================
# STEP 3: GENERATE PREDICTIONS
# ============================================================================

print("\n🔮 Generating predictions...")

# Double-check feature alignment before prediction
if hasattr(model, 'feature_names_in_'):
    if not all(X_test.columns == model.feature_names_in_):
        print("   ⚠ WARNING: Feature order mismatch detected, reordering...")
        X_test = X_test[model.feature_names_in_]
        print("   ✓ Features reordered to match model")

try:
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    print("   ✓ Predictions generated successfully")
except Exception as e:
    print(f"   ✗ ERROR during prediction: {e}")
    print(f"\n   Debug info:")
    print(f"   → X_test columns: {list(X_test.columns)}")
    if hasattr(model, 'feature_names_in_'):
        print(f"   → Model expects: {list(model.feature_names_in_)}")
    print(f"   → X_test shape: {X_test.shape}")
    print(f"   → Model n_features_in_: {model.n_features_in_}")
    exit(1)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"   ✓ Overall Accuracy: {accuracy:.1%}")
print(f"   ✓ Weighted F1 Score: {f1:.1%}")

# ============================================================================
# STEP 4: CREATE COMPREHENSIVE VISUALIZATION
# ============================================================================

print("\n🎨 Creating visualization...")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create figure with subplots
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# ============================================================================
# SUBPLOT 1: CONFUSION MATRIX (Large, top-left)
# ============================================================================

ax1 = fig.add_subplot(gs[0:2, 0:2])

cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(
    cm_normalized,
    annot=cm,  # Show counts
    fmt='d',
    cmap='Blues',
    xticklabels=['L1\nCritical', 'L2\nHigh', 'L3\nModerate', 'L4\nLow', 'L5\nMinimal'],
    yticklabels=['L1\nCritical', 'L2\nHigh', 'L3\nModerate', 'L4\nLow', 'L5\nMinimal'],
    ax=ax1,
    cbar_kws={'label': 'Normalized Frequency'}
)
ax1.set_title('Confusion Matrix\n(Numbers = Actual Counts, Colors = Normalized)', 
              fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Predicted Urgency Level', fontsize=12, fontweight='bold')
ax1.set_ylabel('Actual Urgency Level', fontsize=12, fontweight='bold')

# Add diagonal emphasis
for i in range(5):
    ax1.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=2))

# ============================================================================
# SUBPLOT 2: PER-CLASS METRICS
# ============================================================================

ax2 = fig.add_subplot(gs[0, 2])

# Get classification report as dict
report = classification_report(y_test, y_pred, output_dict=True)

# Extract per-class metrics
classes = [1, 2, 3, 4, 5]
precision = [report[str(c)]['precision'] for c in classes]
recall = [report[str(c)]['recall'] for c in classes]
f1_scores = [report[str(c)]['f1-score'] for c in classes]

x = np.arange(len(classes))
width = 0.25

bars1 = ax2.bar(x - width, precision, width, label='Precision', alpha=0.8)
bars2 = ax2.bar(x, recall, width, label='Recall', alpha=0.8)
bars3 = ax2.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)

ax2.set_xlabel('Urgency Level', fontweight='bold')
ax2.set_ylabel('Score', fontweight='bold')
ax2.set_title('Per-Class Performance Metrics', fontsize=12, fontweight='bold', pad=10)
ax2.set_xticks(x)
ax2.set_xticklabels(['L1', 'L2', 'L3', 'L4', 'L5'])
ax2.legend(loc='lower right')
ax2.set_ylim(0, 1.0)
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=8)

# ============================================================================
# SUBPLOT 3: CRITICAL CASE PERFORMANCE (Highlight)
# ============================================================================

ax3 = fig.add_subplot(gs[1, 2])

# Calculate critical case metrics (Levels 1-2)
critical_mask_true = y_test.isin([1, 2])
critical_mask_pred = pd.Series(y_pred, index=y_test.index).isin([1, 2])

# METRIC 1: Exact level accuracy (stricter - must get L1 vs L2 correct)
if critical_mask_true.sum() > 0:
    critical_exact_accuracy = accuracy_score(
        y_test[critical_mask_true], 
        y_pred[critical_mask_true.values]
    )
else:
    critical_exact_accuracy = 0.0

# METRIC 2: Binary detection rate (primary metric - did we catch critical cases?)
from sklearn.metrics import recall_score, precision_score

# Binarize: critical (1,2) vs non-critical (3,4,5)
y_test_binary = (y_test <= 2).astype(int)
y_pred_binary = (pd.Series(y_pred) <= 2).astype(int)

critical_detection_rate = recall_score(y_test_binary, y_pred_binary)  # This is the 89% metric!
critical_precision = precision_score(y_test_binary, y_pred_binary)

# Display the PRIMARY metric (binary detection) with exact accuracy as reference
metrics_data = {
    'Critical\nDetection': critical_detection_rate,  # PRIMARY: Catches L1 or L2
    'Critical\nPrecision': critical_precision,
    'Exact Level\nAccuracy': critical_exact_accuracy  # SECONDARY: Gets L1 vs L2 right
}

bars = ax3.bar(metrics_data.keys(), metrics_data.values(), 
               color=['#e74c3c', '#e67e22', '#f39c12'], alpha=0.8)
ax3.set_ylabel('Score', fontweight='bold')
ax3.set_title('Critical Case Performance\n(Detection & Precision for L1-L2)', 
              fontsize=12, fontweight='bold', pad=10, color='darkred')
ax3.set_ylim(0, 1.0)
ax3.axhline(y=0.85, color='green', linestyle='--', linewidth=1, alpha=0.5, label='85% Target')
ax3.legend(loc='lower right', fontsize=8)
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1%}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# ============================================================================
# SUBPLOT 4: FEATURE IMPORTANCE (Top 10)
# ============================================================================

ax4 = fig.add_subplot(gs[2, 0:2])

# Get feature importances
importances = model.feature_importances_
indices = np.argsort(importances)[-10:]  # Top 10

# Create readable feature names
feature_name_map = {
    'age': 'Age',
    'gender_encoded': 'Biological Sex',
    'symptom_severity': 'Symptom Severity',
    'symptom_duration_hours': 'Symptom Duration',
    'onset_encoded': 'Onset Pattern',
    'heart_rate': 'Heart Rate',
    'systolic_bp': 'Systolic BP',
    'diastolic_bp': 'Diastolic BP',
    'temperature': 'Temperature',
    'oxygen_saturation': 'O2 Saturation',
    'has_red_flag': 'Red Flag Present',
    'has_chronic_condition': 'Chronic Condition',
    'high_risk_chronic': 'High-Risk Chronic',
    'hr_abnormal': 'HR Abnormal',
    'bp_abnormal': 'BP Abnormal',
    'temp_abnormal': 'Temp Abnormal',
    'spo2_abnormal': 'SpO2 Abnormal',
    'vital_abnormalities': 'Vital Abnormalities',
    'symptom_acuity': 'Symptom Acuity',
    'previous_visits': 'Previous Visits'
}

# Get readable names
readable_names = [feature_name_map.get(feature_columns[i], feature_columns[i]) for i in indices]

ax4.barh(range(len(indices)), importances[indices], alpha=0.8, color='steelblue')
ax4.set_yticks(range(len(indices)))
ax4.set_yticklabels(readable_names)
ax4.set_xlabel('Importance Score', fontweight='bold')
ax4.set_title('Top 10 Feature Importances', fontsize=12, fontweight='bold', pad=10)
ax4.grid(axis='x', alpha=0.3)

# ============================================================================
# SUBPLOT 5: OVERALL METRICS BOX
# ============================================================================

ax5 = fig.add_subplot(gs[2, 2])
ax5.axis('off')

# Calculate additional metrics
oob_score = metadata.get('oob_score', model.oob_score_ if hasattr(model, 'oob_score_') else None)

# Create text box with key metrics
metrics_text = f"""
MODEL PERFORMANCE SUMMARY
{'='*35}

Overall Metrics:
  • Accuracy:        {accuracy:.1%}
  • Weighted F1:     {f1:.1%}
  • OOB Score:       {oob_score:.1%} (CV)

Critical Case Detection (L1-L2):
  • Detection Rate:  {critical_detection_rate:.1%} ⭐
    (Sensitivity - catches critical)
  • Precision:       {critical_precision:.1%}
    (Predicted critical are critical)
  • Exact Level:     {critical_exact_accuracy:.1%}
    (Gets L1 vs L2 distinction right)

Dataset:
  • Training Size:   {len(X_train):,} patients
  • Test Size:       {len(X_test):,} patients
  • MIMIC-IV-ED:     10,000 records

Model Details:
  • Algorithm:       Random Forest
  • Version:         v2 (Improved)
  • Features:        {len(feature_columns)}
  • Trees:           200
  • Class Weight:    Balanced
"""

ax5.text(0.1, 0.95, metrics_text, 
         transform=ax5.transAxes,
         fontsize=10,
         verticalalignment='top',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# ============================================================================
# MAIN TITLE
# ============================================================================

fig.suptitle('ClinicTriage ML Model Performance Analysis\nModel v2 (Improved) - Random Forest Classifier',
             fontsize=18, fontweight='bold', y=0.98)

# Add footer
fig.text(0.5, 0.01, 
         'Model: triage_model_mimic_v2.pkl | Dataset: MIMIC-IV-ED (10K samples) | Date: January 2026',
         ha='center', fontsize=10, style='italic', color='gray')

# ============================================================================
# SAVE FIGURE
# ============================================================================

plt.savefig('model_performance.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("   ✓ Saved: model_performance.png")

plt.savefig('model_performance_highres.png', dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("   ✓ Saved: model_performance_highres.png (high resolution)")

print("\n" + "="*70)
print("✅ VISUALIZATION COMPLETE!")
print("="*70)
print(f"\n📊 Key Findings:")
print(f"   • Overall Accuracy:        {accuracy:.1%}")
print(f"   • Critical Detection Rate: {critical_detection_rate:.1%} ⭐")
print(f"     (Sensitivity for catching L1-L2 cases)")
print(f"   • Critical Precision:      {critical_precision:.1%}")
print(f"   • Exact Level Accuracy:    {critical_exact_accuracy:.1%}")
print(f"     (Gets L1 vs L2 distinction right)")
print(f"\n💾 Output Files:")
print(f"   → model_performance.png (300 DPI - for presentations)")
print(f"   → model_performance_highres.png (600 DPI - for publication)")
print("\n💡 NOTE: The 'Critical Detection Rate' ({critical_detection_rate:.1%}) is the")
print("   primary safety metric - it measures how well we catch urgent cases.")
print("   This should match your training metrics!")
print("\n" + "="*70)
