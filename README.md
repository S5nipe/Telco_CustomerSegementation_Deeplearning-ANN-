# Telco_CustomerSegementation_Deeplearning-ANN-
Deep learning model using Artificial Neural Networks (ANN) to predict Telco customer churn — built with TensorFlow/Keras, achieving 83% accuracy and 0.863 ROC-AUC.

# Customer Churn Prediction Using Deep Learning

A deep learning project that predicts whether a Telco customer is likely 
to leave (churn) based on their demographics, account information, and 
service usage — built using Artificial Neural Networks (ANN) with 
TensorFlow/Keras.

---

## Problem Statement
Customer churn costs telecom companies significantly more to replace 
than to retain. This project builds a binary classification ANN that 
identifies at-risk customers before they leave, enabling proactive 
retention strategies.

---

## Dataset
**Telco Customer Churn Dataset**
- 7,043 customer records · 21 features · Binary target (Churn: Yes/No)
- Class distribution: 73.5% stayed · 26.5% churned (2.77x imbalance)
- Source: [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Project Structure

telco-customer-churn-prediction-ann/
│
├── Customer_Churn_Prediction.ipynb   # Main Jupyter Notebook
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
├── Churn_Prediction_Report.pdf       # 5-page project report
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation


---

## Methodology

### 1. Exploratory Data Analysis
- Churn distribution and class imbalance analysis
- Numeric feature distributions split by churn status
- Categorical feature churn rates (Contract, PaymentMethod, InternetService)
- Correlation heatmap and outlier detection
- Churn rate by tenure band and senior citizen status

### 2. Data Preprocessing
- Converted TotalCharges from object to float (11 missing → median imputed)
- Label encoded binary columns (Yes/No → 1/0)
- One-hot encoded multi-class columns (Contract, InternetService, PaymentMethod)
- StandardScaler applied to numeric features (tenure, MonthlyCharges, TotalCharges)
- Balanced class weights computed to handle 2.77x imbalance
- Stratified 70/15/15 train/validation/test split

### 3. ANN Architecture

| Component | Model v1 (Baseline) | Model v2 (Deeper) |
|---|---|---|
| Input | 23 features | 23 features |
| Hidden layer 1 | 64 neurons, ReLU | 128 neurons, ReLU |
| Regularization 1 | BatchNorm + Dropout 0.3 | BatchNorm + Dropout 0.4 |
| Hidden layer 2 | 32 neurons, ReLU | 64 neurons, ReLU + Dropout 0.3 |
| Hidden layer 3 | — | 32 neurons, ReLU + Dropout 0.2 |
| Output | 1 neuron, Sigmoid | 1 neuron, Sigmoid |
| Parameters | 4,033 | 14,337 |

**Loss:** binary_crossentropy  
**Optimizer:** Adam (lr=0.001)  
**Callbacks:** EarlyStopping · ModelCheckpoint · ReduceLROnPlateau

### 4. Training
- Max 50 epochs · batch size 32
- Class weights applied to penalize missed churn 2.8x more
- Model v2 selected based on higher validation AUC (0.862 vs 0.853)

---

## Results

### Model Comparison

| Metric | Model v1 | Model v2 |
|---|---|---|
| Val Accuracy | 0.8021 | 0.8112 |
| Val AUC | 0.8534 | 0.8621 |
| Val Recall | 0.7123 | 0.7234 |
| Epochs trained | 28 | 32 |

### Final Evaluation (Model v2 on Test Set)

| Metric | Score |
|---|---|
| Accuracy | 83.0% |
| Precision | 65.1% |
| Recall | 77.1% |
| F1 Score | 0.706 |
| ROC-AUC | 0.863 |

> The model correctly identified **77 out of every 100 churners** on 
> unseen data — providing actionable early warning for retention teams.

### Confusion Matrix Summary
| | Predicted: Stay | Predicted: Churn |
|---|---|---|
| **Actual: Stay** | 660 ✅ | 116 ❌ |
| **Actual: Churn** | 64 ❌ | 216 ✅ |

---

## Key Findings
- **Tenure** is the strongest churn predictor — newer customers churn far more
- **Contract type** is the strongest categorical predictor — month-to-month 
  customers churn at ~43% vs under 3% for two-year contract customers
- **High monthly charges** combined with short tenure = highest risk profile
- **13 false negatives per 100 churners** — customers missed by the model

---

## Challenges and Solutions

| Challenge | Solution |
|---|---|
| Class imbalance (73.5/26.5) | Balanced class weights (2.8x penalty for churn misses) |
| TotalCharges stored as object | pd.to_numeric + median imputation |
| Boolean dtype from get_dummies | Explicit .astype(float32) conversion |
| Overfitting risk | Dropout + BatchNormalization + EarlyStopping |
| ANN black-box interpretability | Permutation importance for feature attribution |

---

## Suggested Improvements
- **SMOTE** — generate synthetic churn examples for better minority balance
- **Threshold optimization** — lower decision boundary to 0.3-0.4 for higher recall
- **Keras Tuner / Optuna** — systematic hyperparameter search
- **SHAP values** — individual-level prediction explanations for business teams
- **Ensemble with XGBoost** — combine ANN and gradient boosting for robustness

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/telco-customer-churn-prediction-ann.git
cd telco-customer-churn-prediction-ann

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook Customer_Churn_Prediction.ipynb
```

---

## Requirements

tensorflow>=2.0
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter

---

## Tech Stack
Python · TensorFlow/Keras · scikit-learn · pandas · 
numpy · matplotlib · seaborn · Jupyter Notebook

---

## Topics
`deep-learning` `neural-network` `churn-prediction` 
`binary-classification` `tensorflow` `keras` `imbalanced-data` 
`telco` `python` `jupyter` `machine-learning`
