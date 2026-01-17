# ClinicTriage Model Retraining - January 2026

## 📋 Overview

This directory contains scripts to retrain the ClinicTriage ML model with improved hyperparameters focused on maximizing **critical case detection rate**.

---

## 📁 Files Provided

### Training Scripts
- **`train_model_mimic_v2_jan.py`** - Retrains model with January 2026 naming
- **`visualize_model_jan.py`** - Creates visualization for January model

### Required Input Files
- **`mimic_patients_10k.csv`** - MIMIC-IV-ED processed dataset (10,000 patients)

### Output Files (after training)
- **`triage_model_mimic_v2_jan.pkl`** - Trained model
- **`model_metadata_mimic_v2_jan.pkl`** - Model metadata with all metrics
- **`feature_names_jan.pkl`** - Feature names in correct order
- **`model_performance_jan.png`** - Performance visualization (300 DPI)
- **`model_performance_jan_highres.png`** - High-res visualization (600 DPI)

---

## 🚀 Quick Start

### Step 1: Prepare Environment

```bash
# Navigate to your project directory
cd ~/Documents/ClinicFlow

# Ensure required packages are installed
pip install pandas numpy scikit-learn matplotlib seaborn --break-system-packages

# Verify data file exists
ls -lh mimic_patients_10k.csv
```

### Step 2: Train the Model

```bash
# Run training script
python train_model_mimic_v2_jan.py
```

**Expected runtime:** 30-60 seconds

**Expected output:**
```
======================================================================
CLINICTRIAGE MODEL TRAINING - JANUARY 2026 RETRAINING
Goal: Maximize critical case detection (L1-L2)
======================================================================

📂 Loading MIMIC-IV-ED data...
   ✅ Loaded 10,000 patient records from MIMIC-IV-ED

🔧 Preparing features...
   Feature matrix: (10000, 20)
   Target variable: (10000,)

   Class distribution:
   ⭐ Level 1:  368 (  3.7%)
   ⭐ Level 2: 3355 ( 33.6%)
      Level 3: 5548 ( 55.5%)
      Level 4:  703 (  7.0%)
      Level 5:   26 (  0.3%)

   Critical cases (L1+L2): 3,723 (37.2%)

...

🎯 Final Performance Summary:
   • Overall Accuracy:        XX.X%
   • Critical Exact Accuracy: XX.X%
   • Critical Detection Rate: XX.X% ⭐ PRIMARY METRIC
   • Critical Precision:      XX.X%
   • Weighted F1 Score:       XX.X%
   • OOB Score:               XX.X%
```

### Step 3: Visualize Results

```bash
# Create performance visualization
python visualize_model_jan.py
```

**Output:** `model_performance_jan.png` and `model_performance_jan_highres.png`

---

## 📊 Model Improvements

### Hyperparameter Changes from Original v2

| Parameter | Original v2 | January 2026 | Rationale |
|-----------|-------------|--------------|-----------|
| `n_estimators` | 300 | **500** | More trees = more stable predictions |
| `max_depth` | 20 | **25** | Deeper trees capture complex patterns |
| `min_samples_split` | 5 | **3** | More aggressive splitting |
| `min_samples_leaf` | 2 | **1** | Allow finer-grained leaves |
| `max_samples` | None | **0.8** | Subsample for diversity |
| `class_weight` | balanced | **balanced** | Maintained (critical!) |
| `random_state` | 42 | **42** | Same for fair comparison |

### Focus: Critical Case Detection

The new model prioritizes **detection rate (sensitivity)** over exact level accuracy:

- **Detection Rate:** % of actual critical cases (L1 or L2) that are caught as critical
- **Exact Level Accuracy:** % of critical cases with exact L1 vs L2 prediction

**Why detection rate matters more:**
- Both L1 and L2 receive urgent priority in queue
- Missing a critical case is dangerous
- Getting L1 vs L2 distinction slightly wrong is less critical

---

## 🎯 Expected Results

### Target Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| Overall Accuracy | >78% | Maintain overall performance |
| Critical Detection | >85% | **PRIMARY GOAL** - Catch most urgent cases |
| Critical Precision | >80% | Avoid over-triage |
| Exact Level Accuracy | >74% | Secondary metric |

### Comparison to Original v2

**Original v2 Model (December 2025):**
- Overall Accuracy: 77.8%
- Critical Exact Accuracy: 74.4%
- Critical Detection: ~80% (estimated)
- OOB Score: 77.5%

**Goal for January 2026 Model:**
- Overall Accuracy: 78-80%
- Critical Detection: **85-90%** ⭐ PRIORITY
- Critical Exact Accuracy: 75-80%
- OOB Score: 78-80%

---

## 📈 Understanding the Metrics

### 1. Overall Accuracy
```
Correct predictions across all 5 urgency levels / Total predictions
```
Standard metric, but can be misleading with class imbalance.

### 2. Critical Detection Rate (Sensitivity) ⭐ PRIMARY METRIC
```
True Positives / (True Positives + False Negatives)

Where:
- True Positive: Actual L1 or L2 → Predicted L1 or L2
- False Negative: Actual L1 or L2 → Predicted L3, L4, or L5
```
**This measures patient safety** - how many urgent cases we catch.

### 3. Critical Precision
```
True Positives / (True Positives + False Positives)

Where:
- True Positive: Predicted L1 or L2, actually L1 or L2
- False Positive: Predicted L1 or L2, actually L3+
```
Measures how accurate our urgent predictions are.

### 4. Critical Exact Level Accuracy
```
Among actual L1-L2 patients:
  Correct L1→L1 + Correct L2→L2 / Total L1-L2 patients
```
Stricter metric - must get L1 vs L2 distinction right.

---

## 🔍 Interpreting Results

### Example Output Scenarios

**Scenario 1: Success ✅**
```
Critical Detection Rate: 88.5% ⭐
Critical Precision: 82.1%
Critical Exact Accuracy: 76.3%
```
**Interpretation:** Model catches 88.5% of urgent cases, with 82% accuracy when predicting urgent. Excellent for patient safety!

**Scenario 2: Mixed 🤔**
```
Critical Detection Rate: 82.0%
Critical Precision: 85.0%
Critical Exact Accuracy: 78.0%
```
**Interpretation:** Detection rate improved slightly but not dramatically. Consider if other tradeoffs are acceptable.

**Scenario 3: No Improvement ⚠️**
```
Critical Detection Rate: 80.5%
Critical Precision: 81.0%
Critical Exact Accuracy: 74.8%
```
**Interpretation:** Similar to original model. Hyperparameter changes didn't help. Stick with original v2.

---

## 🎓 For Competition Submission

### If January Model is Better (Detection >85%):

**Use this language:**

> "The ClinicTriage model achieves **88.5% sensitivity for critical case detection**, successfully identifying nearly 9 out of 10 urgent patients requiring immediate attention. Combined with 77.8% overall accuracy and 82% precision, the model demonstrates strong performance across all urgency levels while prioritizing patient safety through high detection rates for critical cases."

### If January Model is Similar:

**Use original v2 model and language:**

> "The model achieves 77.8% overall accuracy and 74.4% exact-level accuracy on critical cases, representing a dramatic improvement from an initial model that achieved 0% critical case accuracy due to severe class imbalance. By implementing balanced class weighting, we successfully transformed a broken model into one suitable for medical triage applications."

---

## 🔧 Troubleshooting

### Error: "File not found: mimic_patients_10k.csv"
```bash
# Make sure you're in the right directory
pwd
# Should show: /Users/guest2/Documents/ClinicFlow

# Check if file exists
ls -la mimic_patients_10k.csv
```

### Error: "ModuleNotFoundError: No module named 'sklearn'"
```bash
pip install scikit-learn --break-system-packages
```

### Training takes too long (>5 minutes)
```python
# In train_model_mimic_v2_jan.py, reduce:
n_estimators=500  →  n_estimators=200
```

### Poor performance on critical cases (<75% detection)
- This indicates the data or hyperparameters aren't optimal
- Stick with original v2 model (74.4% is honest and verifiable)

---

## 📊 Files Generated

After running both scripts, you'll have:

```
ClinicFlow/
├── mimic_patients_10k.csv                    # Input data
├── train_model_mimic_v2_jan.py               # Training script
├── visualize_model_jan.py                    # Visualization script
├── triage_model_mimic_v2_jan.pkl             # NEW: Trained model
├── model_metadata_mimic_v2_jan.pkl           # NEW: Metadata
├── feature_names_jan.pkl                     # NEW: Features
├── model_performance_jan.png                 # NEW: Visualization
└── model_performance_jan_highres.png         # NEW: High-res viz
```

---

## ✅ Final Checklist

Before using new model for competition:

- [ ] Training completed without errors
- [ ] Critical detection rate >85% (or comparable to original)
- [ ] Overall accuracy >77%
- [ ] Visualization generated successfully
- [ ] Compared to original v2 model
- [ ] Decided which model to use for submission
- [ ] Updated narrative with correct metrics
- [ ] Verified all numbers match metadata

---

## 🎯 Bottom Line

**Goal:** Train a model with >85% critical case detection while maintaining overall performance.

**Success Criteria:** 
- If new model achieves >85% detection → Use it ✅
- If new model similar to v2 → Use original v2 ✅

**Either way, you'll have honest, verifiable metrics for your submission!**
