# Customer Churn Prediction

## Project Overview
This project predicts which customers are likely to leave a telecom service (churn), enabling companies to take proactive retention measures. It demonstrates the complete data science workflow, from data cleaning to model evaluation and visualization.

## Dataset
- **Source:** `TelecomCustomerChurn.csv`
- **Description:** Contains customer demographics, account information, service usage, and churn status.
- **Target Variable:** `Churn` (Yes or No)

## Features
- **Customer demographics:** gender, age, etc.
- **Account information:** tenure, contract type, payment method
- **Service usage patterns:** internet service, phone lines

## Key Steps

### 1. Data Loading & Exploration
- Analyzed data shape, types, missing values, and target distribution.

### 2. Data Preprocessing
- Handled missing values with median imputation.
- Encoded categorical features using `OrdinalEncoder`.

### 3. Feature Engineering & Balancing
- Applied SMOTE to balance the target variable.
- Encoded `Churn` numerically (Yes → 1, No → 0).

### 4. Train-Test Split
- Split dataset into 75% training and 25% testing sets with stratification.

### 5. Modeling & Evaluation
- Trained Logistic Regression and Random Forest models.
- Evaluated using Accuracy, ROC-AUC, cross-validation, confusion matrix, and classification report.

### 6. Feature Importance & Visualization
- Identified top features influencing churn using Random Forest.
- Plotted ROC curves and feature importance for interpretability.

## Tools & Technologies
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Imbalanced-learn (SMOTE)

## Results
- **Best Model:** Random Forest
- **Performance:** High accuracy and ROC-AUC
- **Business Insight:** Top features driving churn were identified to help target customer retention strategies.
