"""
ClinicTriage Model Performance Visualization - January 2026 Model
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
    recall_score,
    precision_score
)
from sklearn.model_selection import train_test_split

print("="*70)
print("CLINICTRIAGE JANUARY 2026 MODEL - PERFORMANCE VISUALIZATION")
print("="*70)

# ============================================================================
# LOAD MODEL AND METADATA
# ============================================================================

print("\n📦 Loading January 2026 model...")

with open('triage_model_mimic_v2_jan.pkl', 'rb') as f:
    model = pickle.load(f)
print("   ✓ Model loaded")

with open('model_metadata_mimic_v2_jan.pkl', 'rb') as f:
    metadata = pickle.load(f)
print("   ✓ Metadata loaded")

print(f"\n   Model Info:")
print(f"   → Version: {metadata.get('model_version', 'N/A')}")
print(f"   → Training Date: {metadata.get('training_date', 'N/A')[:10]}")
print(f"   → Overall Accuracy: {metadata.get('accuracy', 0):.1%}")
print(f"   → Critical Detection: {metadata.get('critical_detection_rate', 0):.1%} ⭐")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n📊 Loading test data...")

df = pd.read_csv('mimic_patients_10k.csv')
feature_columns = metadata['feature_names']

X = df[feature_columns]
y = df['urgency_level']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   ✓ Test set: {len(X_test):,} samples")

# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================

print("\n🔮 Generating predictions...")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Calculate all metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

# Critical metrics
critical_mask = y_test.isin([1, 2])
critical_exact_accuracy = accuracy_score(
    y_test[critical_mask], 
    y_pred[critical_mask.values]
)

y_test_binary = (y_test <= 2).astype(int)
y_pred_binary = (pd.Series(y_pred) <= 2).astype(int)

critical_detection = recall_score(y_test_binary, y_pred_binary)
critical_precision = precision_score(y_test_binary, y_pred_binary)

print(f"   ✓ Metrics calculated")

# ============================================================================
# CREATE VISUALIZATION
# ============================================================================

print("\n🎨 Creating visualization...")

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# ============================================================================
# SUBPLOT 1: CONFUSION MATRIX
# ============================================================================

ax1 = fig.add_subplot(gs[0:2, 0:2])

cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(
    cm_normalized,
    annot=cm,
    fmt='d',
    cmap='Blues',
    xticklabels=['L1\nCritical', 'L2\nHigh', 'L3\nModerate', 'L4\nLow', 'L5\nMinimal'],
    yticklabels=['L1\nCritical', 'L2\nHigh', 'L3\nModerate', 'L4\nLow', 'L5\nMinimal'],
    ax=ax1,
    cbar_kws={'label': 'Normalized Frequency'}
)
ax1.set_title('Confusion Matrix - January 2026 Model\n(Numbers = Counts, Colors = Normalized)', 
              fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Predicted Urgency Level', fontsize=12, fontweight='bold')
ax1.set_ylabel('Actual Urgency Level', fontsize=12, fontweight='bold')

for i in range(5):
    ax1.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=2))

# ============================================================================
# SUBPLOT 2: PER-CLASS METRICS
# ============================================================================

ax2 = fig.add_subplot(gs[0, 2])

report = classification_report(y_test, y_pred, output_dict=True)

classes = [1, 2, 3, 4, 5]
precision_scores = [report[str(c)]['precision'] for c in classes]
recall_scores = [report[str(c)]['recall'] for c in classes]
f1_scores = [report[str(c)]['f1-score'] for c in classes]

x = np.arange(len(classes))
width = 0.25

bars1 = ax2.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
bars2 = ax2.bar(x, recall_scores, width, label='Recall', alpha=0.8)
bars3 = ax2.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)

ax2.set_xlabel('Urgency Level', fontweight='bold')
ax2.set_ylabel('Score', fontweight='bold')
ax2.set_title('Per-Class Performance', fontsize=12, fontweight='bold', pad=10)
ax2.set_xticks(x)
ax2.set_xticklabels(['L1', 'L2', 'L3', 'L4', 'L5'])
ax2.legend(loc='lower right')
ax2.set_ylim(0, 1.0)
ax2.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=8)

# ============================================================================
# SUBPLOT 3: CRITICAL CASE PERFORMANCE
# ============================================================================

ax3 = fig.add_subplot(gs[1, 2])

metrics_data = {
    'Detection\nRate': critical_detection,
    'Precision': critical_precision,
    'Exact Level': critical_exact_accuracy
}

colors = ['#e74c3c', '#f39c12', '#3498db']
bars = ax3.bar(metrics_data.keys(), metrics_data.values(), color=colors, alpha=0.8)

ax3.set_ylabel('Score', fontweight='bold')
ax3.set_title('Critical Case Performance (L1-L2)\nJanuary 2026 Model', 
              fontsize=12, fontweight='bold', pad=10, color='darkred')
ax3.set_ylim(0, 1.0)
ax3.axhline(y=0.85, color='green', linestyle='--', linewidth=1, alpha=0.5, label='85% Target')
ax3.legend(loc='lower right', fontsize=8)
ax3.grid(axis='y', alpha=0.3)

for i, bar in enumerate(bars):
    height = bar.get_height()
    star = " ⭐" if i == 0 and height > 0.85 else ""
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1%}{star}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# ============================================================================
# SUBPLOT 4: FEATURE IMPORTANCE
# ============================================================================

ax4 = fig.add_subplot(gs[2, 0:2])

importances = model.feature_importances_
indices = np.argsort(importances)[-10:]

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

readable_names = [feature_name_map.get(feature_columns[i], feature_columns[i]) for i in indices]

ax4.barh(range(len(indices)), importances[indices], alpha=0.8, color='steelblue')
ax4.set_yticks(range(len(indices)))
ax4.set_yticklabels(readable_names)
ax4.set_xlabel('Importance Score', fontweight='bold')
ax4.set_title('Top 10 Feature Importances', fontsize=12, fontweight='bold', pad=10)
ax4.grid(axis='x', alpha=0.3)

# ============================================================================
# SUBPLOT 5: METRICS SUMMARY
# ============================================================================

ax5 = fig.add_subplot(gs[2, 2])
ax5.axis('off')

oob_score = metadata.get('oob_score', 0)

metrics_text = f"""
JANUARY 2026 MODEL SUMMARY
{'='*35}

Overall Performance:
  • Accuracy:        {accuracy:.1%}
  • Weighted F1:     {f1:.1%}
  • OOB Score:       {oob_score:.1%}

Critical Cases (L1-L2):
  • Detection Rate:  {critical_detection:.1%} ⭐
    (Sensitivity - catches critical)
  • Precision:       {critical_precision:.1%}
    (Predicted critical accuracy)
  • Exact Level:     {critical_exact_accuracy:.1%}
    (L1 vs L2 distinction)

Dataset:
  • Training:        {len(X_train):,} patients
  • Testing:         {len(X_test):,} patients
  • Source:          MIMIC-IV-ED

Model Configuration:
  • Algorithm:       Random Forest
  • Trees:           500
  • Max Depth:       25
  • Class Weight:    Balanced
  • Random State:    42
"""

ax5.text(0.1, 0.95, metrics_text, 
         transform=ax5.transAxes,
         fontsize=10,
         verticalalignment='top',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

# ============================================================================
# TITLE AND FOOTER
# ============================================================================

fig.suptitle('ClinicTriage January 2026 Model - Performance Analysis\nOptimized for Critical Case Detection',
             fontsize=18, fontweight='bold', y=0.98)

fig.text(0.5, 0.01, 
         f'Model: triage_model_mimic_v2_jan.pkl | Trained: {metadata.get("training_date", "")[:10]} | MIMIC-IV-ED Dataset',
         ha='center', fontsize=10, style='italic', color='gray')

# ============================================================================
# SAVE
# ============================================================================

plt.savefig('model_performance_jan.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("   ✓ Saved: model_performance_jan.png")

plt.savefig('model_performance_jan_highres.png', dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("   ✓ Saved: model_performance_jan_highres.png")

print("\n" + "="*70)
print("✅ JANUARY 2026 MODEL VISUALIZATION COMPLETE!")
print("="*70)
print(f"\n📊 Performance Summary:")
print(f"   • Overall Accuracy:        {accuracy:.1%}")
print(f"   • Critical Detection Rate: {critical_detection:.1%} ⭐ PRIMARY METRIC")
print(f"   • Critical Precision:      {critical_precision:.1%}")
print(f"   • Critical Exact Level:    {critical_exact_accuracy:.1%}")
print(f"   • Weighted F1:             {f1:.1%}")
print(f"\n💾 Output Files:")
print(f"   → model_performance_jan.png (300 DPI)")
print(f"   → model_performance_jan_highres.png (600 DPI)")
print("\n" + "="*70)
