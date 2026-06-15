# Customer Churn Prediction — Interconnect Telecom

End-to-end churn prediction pipeline. Capstone project demonstrating full ML workflow from raw relational data to business retention strategy.

## Problem
Predict which subscribers will churn so the retention team can intervene proactively. Architecture transferable to patient dropout, clinical trial adherence, and health platform retention.

## Dataset
- 4 relational tables: contract, personal, internet, phone services
- Binary target: churn yes/no
- Class imbalance: ~85% non-churn

## Methodology
1. Integrated and cleaned 4 relational data sources
2. EDA: churn rate by contract type, tenure, service bundle
3. Feature engineering: tenure, service interactions, encoded categoricals
4. Class imbalance handled via class_weight='balanced' — no artificial resampling
5. Trained 5 models: Logistic Regression, Decision Tree, Random Forest, CatBoost, XGBoost
6. Final selection via cross-validation; risk scores translated into retention framework

## Results

| Model | AUC-ROC |
|---|---|
| Logistic Regression | baseline |
| Random Forest | — |
| CatBoost | — |
| **XGBoost** | **78.4%** |

## Key Takeaways
- Contract type was the strongest churn predictor
- class_weight='balanced' outperformed resampling while preserving data integrity
- Churn risk scores map directly to intervention priority lists

## Stack
`Python` `Scikit-learn` `XGBoost` `CatBoost` `Pandas` `NumPy` `Matplotlib` `Seaborn`
