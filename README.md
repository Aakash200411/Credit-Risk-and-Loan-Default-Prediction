# Credit Risk and Credit Card Default Prediction

An end-to-end Data Science and Machine Learning project that predicts the likelihood of a credit-card customer defaulting on their payment in the following month.

The project covers exploratory data analysis, feature engineering, model comparison, hyperparameter tuning, threshold optimisation, model evaluation and explainable AI.

## Project Overview

Financial institutions need to identify customers who may fail to meet their upcoming credit-card payment obligations. Identifying potential defaults can help analysts prioritise accounts for further review and manage credit risk.

This project develops a binary classification model using historical credit-card customer data to predict whether a customer will default on their payment next month.

**Objective:** Predict the probability of credit-card payment default and identify the factors that contribute to the model's predictions.

> **Scope:** This project predicts credit-card payment default for the following month. It is not a general-purpose loan default prediction system.

## Dataset

**Dataset:** Default of Credit Card Clients  
**Source:** UCI Machine Learning Repository  
**Original study:** Yeh, I.-C. and Lien, C.-H. (2009)

- [Official UCI Dataset Page](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- [Dataset DOI](https://doi.org/10.24432/C55S3H)

### Dataset Information

| Attribute | Description |
|---|---|
| Total records | 30,000 |
| Original features | 23 predictors + ID + target |
| Total columns | 25 |
| Target variable | `default payment next month` |
| Problem type | Binary classification |
| Default class | 1 |
| Non-default class | 0 |
| Default rate | 22.12% |

The dataset contains customer demographic information, credit limits, repayment status, historical bill amounts and previous payment amounts.

The target variable indicates whether a customer defaulted on their credit-card payment in the following month.

**Note:** This is a historical dataset representing a specific population and period. Its results should not be assumed to represent current credit-card customers or other countries.

## Technologies Used

| Category | Tools and Libraries |
|---|---|
| Programming | Python |
| Data manipulation | Pandas, NumPy |
| Data visualization | Matplotlib, Seaborn |
| Machine learning | Scikit-learn |
| Gradient boosting | XGBoost, LightGBM |
| Imbalanced data | Imbalanced-learn, SMOTE |
| Explainable AI | SHAP, Permutation Importance |
| Model persistence | Joblib |
| Environment | Google Colab, Jupyter Notebook |

## Project Workflow

### 1. Data Loading and Validation
- Loaded and inspected the dataset.
- Checked data types, missing values and duplicate records.
- Examined categorical and repayment-status values.
- Handled special education and marriage codes.

### 2. Exploratory Data Analysis
- Analysed default and non-default customer distributions.
- Studied credit limits, age, bill amounts and payment amounts.
- Compared default rates across customer groups.
- Examined correlations and repayment patterns.

### 3. Feature Engineering

Created additional features to capture repayment behaviour and financial patterns:

| Feature | Description |
|---|---|
| `TOTAL_BILL_6M` | Total bill amount over six months |
| `TOTAL_PAYMENT_6M` | Total payment amount over six months |
| `AVG_BILL_6M` | Average monthly bill amount |
| `AVG_PAYMENT_6M` | Average monthly payment amount |
| `PAYMENT_TO_BILL_RATIO` | Ratio of total payments to total bills |
| `DELAY_COUNT_6M` | Number of months with delayed repayment |
| `MAX_DELAY_STATUS` | Maximum repayment delay status |
| `AVG_REPAYMENT_STATUS` | Average repayment status |
| `CREDIT_UTILIZATION` | Bill amount relative to credit limit |

The resulting modelling dataset contains 32 predictor features, excluding the customer ID and target variable.

### 4. Data Preprocessing
- Separated predictors and target variable.
- Used an 80:20 stratified train-test split.
- Created a separate validation partition within the training data.
- Applied numerical imputation and standardisation.
- Applied categorical imputation and one-hot encoding.
- Used pipelines to keep preprocessing within model training and cross-validation.

### 5. Model Comparison

Compared multiple classification algorithms using stratified 3-fold cross-validation.

Models evaluated:
- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- Linear SVM (Calibrated)
- XGBoost
- LightGBM

**Primary evaluation metric:** PR-AUC (Average Precision), because the dataset has an imbalanced target.

ROC-AUC was also used as a complementary metric.

### 6. Imbalanced Data Handling
- Evaluated class-weighted classification models.
- Experimented with SMOTE using a Logistic Regression pipeline.
- Kept resampling within training folds to prevent data leakage.
- Did not apply resampling to validation or final test data.

### 7. Hyperparameter Tuning
- Selected the candidate with the highest mean cross-validated PR-AUC.
- Used RandomizedSearchCV for hyperparameter optimisation.
- Tuned the selected XGBoost model using stratified cross-validation.
- Kept the final test set separate from model selection and tuning.

### 8. Threshold Optimisation and Evaluation
- Evaluated the model at the standard probability threshold of 0.50.
- Selected a classification threshold using validation data by maximising F2-score.
- Refit the tuned model on the full training partition.
- Evaluated the final model on the untouched test set.

### 9. Explainable AI
Used two feature-importance approaches:
- **Permutation Importance:** Measures how model performance changes when individual features are shuffled.
- **SHAP:** Explains the contribution of features to model predictions.

Also included an individual customer prediction explanation.

### 10. Interactive Prediction Demo

Implemented an interactive prediction demonstration directly inside the Jupyter Notebook using IPython widgets.

Users can modify customer information and view:
- Predicted default probability
- Predicted classification
- Risk label
- Selected probability threshold

No separate Streamlit application is required.

## Model Results

### Selected Model: XGBoost

XGBoost achieved the highest mean PR-AUC among the candidate models during the initial 3-fold cross-validation comparison.

| Model | Cross-validated PR-AUC |
|---|---:|
| XGBoost | 0.5650 |
| LightGBM | 0.5599 |
| Gradient Boosting | 0.5580 |
| Random Forest | 0.5543 |
| Extra Trees | 0.5502 |
| Decision Tree | 0.5247 |
| Logistic Regression | 0.5052 |
| Linear SVM (Calibrated) | 0.5019 |

After hyperparameter tuning, XGBoost achieved a cross-validated PR-AUC of **0.5648**.

### Final Test Performance

The final model was evaluated using two classification thresholds.

| Metric | Threshold 0.50 | Threshold 0.10 |
|---|---:|---:|
| Accuracy | 82.02% | 47.73% |
| Precision | 67.71% | 28.82% |
| Recall | 35.72% | 92.77% |
| F1-score | 0.4677 | 0.4398 |
| ROC-AUC | 0.7916 | 0.7916 |
| PR-AUC | 0.5632 | 0.5632 |

**Threshold selection:** The threshold of 0.10 was selected using the validation partition by maximising F2-score.

A lower threshold increases recall but also increases false-positive predictions.

- **Threshold 0.50:** Higher precision, but lower recall.
- **Threshold 0.10:** Higher recall, but lower precision.

The threshold should be selected according to the business objective and the relative cost of false positives and false negatives.

### Important Predictive Features

Permutation importance identified the following influential features:

1. `PAY_0` – Most recent repayment status
2. `MAX_DELAY_STATUS` – Maximum repayment delay status
3. `DELAY_COUNT_6M` – Number of delayed repayment months
4. `CREDIT_UTILIZATION` – Credit utilisation
5. `AVG_REPAYMENT_STATUS` – Average repayment status
6. `BILL_AMT1` – Most recent bill amount
7. `PAYMENT_TO_BILL_RATIO` – Payment-to-bill ratio

These findings suggest that repayment history and payment behaviour are important predictive signals in this dataset.

Feature importance indicates predictive association, not causation.

### Probability Calibration

The project also evaluates sigmoid probability calibration.

| Model probabilities | Brier Score |
|---|---:|
| Raw probabilities | 0.1332 |
| Sigmoid calibrated probabilities | 0.1339 |

In this experiment, sigmoid calibration did not improve the Brier score. The raw model probabilities performed slightly better on this metric.

## Repository Structure

```text
Credit-Risk-and-Loan-Default-Prediction/
│
├── Credit_Risk_and_Loan_Default_Prediction_FINAL.ipynb
│
├── README.md
│
├── data/
│   └── default of credit card clients.xlsx
│
└── artifacts/
    ├── credit_default_xgboost_pipeline.joblib
    ├── model_cv_comparison.csv
    ├── final_test_metrics.csv
    └── permutation_feature_importance.csv
```

The `data/` folder contains the original dataset.

The `artifacts/` folder contains the saved trained model and exported evaluation results. Include only files that have actually been generated by the notebook.

## How to Run the Project

### Google Colab (Recommended)

1. Clone or download this repository.
2. Open the Jupyter Notebook in Google Colab.
3. Upload the dataset through the Colab Files panel.
4. Make sure the dataset is available under `/content/`.
5. Run the notebook cells sequentially.

The notebook supports the original `.xls` dataset and its `.xlsx` version.

The trained model and evaluation outputs can be generated by executing the relevant notebook cells.

## Limitations and Responsible Use

- The dataset is historical and represents a specific population and period.
- Model performance may not generalise to modern credit portfolios or different geographical markets.
- Demographic variables such as age and sex can introduce fairness and legal concerns.
- The selected threshold involves a trade-off between precision and recall.
- Model predictions should not be treated as certain outcomes.
- Further validation, fairness analysis and monitoring are necessary before real-world use.

**This project is intended for educational and analytical purposes. It should support risk assessment and prioritisation for human review, not independently approve, reject or price credit.**

## Future Improvements

- Validate the model on newer and more representative credit-risk datasets.
- Conduct fairness and bias evaluation across customer groups.
- Explore additional probability calibration methods.
- Perform cost-sensitive threshold optimisation.
- Evaluate model stability and drift over time.
- Develop a deployment-ready prediction API with monitoring.

## Author

**Aakash Lodha**  
Artificial Intelligence & Data Science

- GitHub: [Aakash Lodha](https://github.com/aakash200411)
- Portfolio: [Personal Portfolio](https://aakash200411.github.io/Portfolio/)

## Acknowledgements

Yeh, I.-C. and Lien, C.-H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients*. Expert Systems with Applications.

Dataset: Default of Credit Card Clients, UCI Machine Learning Repository.

[Dataset DOI: 10.24432/C55S3H](https://doi.org/10.24432/C55S3H)