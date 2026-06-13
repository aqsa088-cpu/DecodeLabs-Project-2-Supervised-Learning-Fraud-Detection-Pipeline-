# Task 2 — Supervised Learning: Fraud Detection Pipeline

## Project Overview

This project builds and tunes a **supervised classification pipeline** to identify fraudulent transactions in a highly imbalanced real-world dataset. It covers multi-file data ingestion, feature engineering, SMOTE oversampling, training multiple classifiers, hyperparameter tuning with `RandomizedSearchCV`, and strict evaluation using Precision, Recall, and ROC-AUC — deliberately discarding raw accuracy as a misleading metric on imbalanced data.

---

## Dataset

**Source:** `Fraud detection.zip`  
**Extracted Structure:**
```
fraud_detection_data/
├── Data/
│   ├── Transaction Data/
│   │   ├── transaction_records.csv      ← core transaction table
│   │   └── transaction_metadata.csv     ← timestamps & merchant IDs
│   ├── Fraudulent Patterns/
│   │   ├── fraud_indicators.csv         ← fraud labels (FraudIndicator)
│   │   └── suspicious_activity.csv
│   ├── Merchant Information/
│   ├── Customer Profiles/
│   └── Transaction Amounts/
├── Src/
└── Readme.md
```

### Final Merged Dataset (after all joins)

| Column | Type | Description |
|--------|------|-------------|
| `TransactionID` | int | Unique transaction ID (dropped before modeling) |
| `Amount` | float | Transaction amount in USD |
| `CustomerID` | int | Customer identifier (dropped before modeling) |
| `Timestamp` | datetime | Transaction timestamp |
| `MerchantID` | int | Merchant identifier (dropped before modeling) |
| `is_fraud` | int | Target variable — 0 = legitimate, 1 = fraud |
| `Hour` | int | Hour extracted from Timestamp |
| `DayOfWeek` | int | Day of week extracted from Timestamp |
| `Month` | int | Month extracted from Timestamp |

### Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| 0 — Not Fraud | 955 | 95.5% |
| 1 — Fraud | 45 | 4.5% |

---

## Requirements Checklist

| Requirement | Status |
|---|---|
| Implement SMOTE to handle class imbalance | ✅ Done |
| Train Logistic Regression (Scikit-Learn) | ✅ Done |
| Train Random Forest (Scikit-Learn) | ✅ Done |
| Hyperparameter tuning with RandomizedSearchCV | ✅ Done |
| Evaluate using Precision, Recall, ROC-AUC (no Accuracy) | ✅ Done |

---

## Notebook Structure

### Cell 0 — Extract ZIP Archive
- Extracts `Fraud detection.zip` to `/content/fraud_detection_data/`
- Lists top-level contents: `Data/`, `Src/`, `Readme.md`

### Cell 1 — Explore Data Directory
- Lists subdirectories inside `Data/`

### Cell 2 — Load Transaction Records
- Loads `transaction_records.csv` (1000 rows × 3 cols: `TransactionID`, `Amount`, `CustomerID`)

### Cell 3 — Load & Merge Transaction Metadata
- Loads `transaction_metadata.csv` (`TransactionID`, `Timestamp`, `MerchantID`)
- Merges with transaction records on `TransactionID` (left join)
- Result: 1000 rows × 5 columns

### Cell 4 — Explore Fraudulent Patterns Directory
- Lists `fraud_indicators.csv` and `suspicious_activity.csv`

### Cell 5 — Load & Merge Fraud Labels
- Loads `fraud_indicators.csv` (`TransactionID`, `FraudIndicator`)
- Merges fraud labels into main DataFrame on `TransactionID`
- Fills any NaN fraud labels with 0 (assume legitimate)

### Cell 6 — Rename Target Column
- Renames `FraudIndicator` → `is_fraud`
- Prints class distribution: **955 legitimate / 45 fraud (4.5%)**

### Cell 7 — Feature Engineering from Timestamp
- Converts `Timestamp` to datetime
- Extracts: `Hour`, `DayOfWeek`, `Month`
- Drops `Timestamp` and `TransactionID` (identifier)

### Cell 8 — Train/Test Split & Feature Scaling
- Drops `CustomerID` and `MerchantID` (high-cardinality identifiers)
- Features used: `Amount`, `Hour`, `DayOfWeek`, `Month`
- **80/20 stratified split** to preserve 4.5% fraud ratio
- `StandardScaler` applied to all 4 numerical features

### Cell 9 — SMOTE Oversampling
- Applies SMOTE on training data only
- Balances minority class synthetically

| | Not Fraud | Fraud |
|--|--|--|
| Before SMOTE | 764 | 36 |
| After SMOTE | 764 | 764 |

- Training set size grows from 800 → 1,528 samples

### Cell 10 — Model Training & Evaluation

**Logistic Regression** (`solver='liblinear'`, trained on SMOTE data):

| Metric | Score |
|--------|-------|
| Precision | 0.0455 |
| Recall | 0.5556 |
| ROC-AUC | 0.4747 |

**Random Forest** (`n_estimators=100`, trained on SMOTE data):

| Metric | Score |
|--------|-------|
| Precision | 0.0909 |
| Recall | 0.1111 |
| ROC-AUC | 0.5782 |

### Cell 11 — Hyperparameter Tuning (Random Forest)
- `RandomizedSearchCV` with 20 iterations, 5-fold CV, scoring = `roc_auc`
- Parameters tuned: `n_estimators`, `max_features`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `bootstrap`
- Best CV ROC-AUC on training: **0.9764**

**Best Parameters Found:**

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 100 |
| `max_features` | `log2` |
| `max_depth` | None (unlimited) |
| `min_samples_split` | 2 |
| `min_samples_leaf` | 1 |
| `bootstrap` | True |

**Optimized Random Forest Test Results:**

| Metric | Score |
|--------|-------|
| Precision | 0.0909 |
| Recall | 0.1111 |
| ROC-AUC | 0.5782 |

---

## Why Accuracy Was Discarded

With a 95.5%/4.5% class split, a naive model predicting **"not fraud" for every transaction** achieves **95.5% accuracy** while catching zero fraud. This is why the project uses:

- **Precision** — of all flagged fraud cases, how many are actually fraud
- **Recall** — of all actual fraud, how many were correctly identified
- **ROC-AUC** — overall ability to distinguish fraud from legitimate transactions across all thresholds

---

## Performance Notes

The low test-set metrics are expected and stem from the **very small fraud class** (only 45 fraud samples total, 9 in the test set). Despite SMOTE achieving a strong training CV ROC-AUC of 0.9764, generalization to only 9 real fraud test cases is unstable. In a production scenario with thousands of real fraud examples, these pipelines would generalize significantly better.

---

## Libraries & Key Skills

| Library | Usage |
|---------|-------|
| `pandas` | Multi-file data loading, merging, feature extraction |
| `numpy` | Numerical operations |
| `scikit-learn` | StandardScaler, train_test_split, LogisticRegression, RandomForestClassifier, RandomizedSearchCV, metrics |
| `imbalanced-learn` | SMOTE oversampling |
| `zipfile / os` | Archive extraction and path management |

**Key Skills:** Classification algorithms, Scikit-Learn pipelines, imbalanced data handling, SMOTE, hyperparameter tuning, fraud-detection metrics

---

## How to Run

1. Upload `Fraud detection.zip` to `/content/` in Google Colab
2. Open `Task2_Supervised_learning_.ipynb` in Google Colab
3. Run all cells (`Runtime → Run all`)
4. Review model metrics (Precision, Recall, ROC-AUC) printed after each evaluation step
