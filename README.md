#  Telecom Customer Churn Prediction

## 🧠 Overview

This project analyzes customer churn behavior in a telecom dataset and builds predictive models to identify customers likely to leave. The workflow includes **data cleaning, EDA, preprocessing, feature engineering, balancing (SMOTE)**, and **model comparison** using machine learning techniques such as **Random Forest, Logistic Regression, and Gradient Boosting**.

The goal is not only to predict churn but also to derive **business insights and actionable strategies** that can help reduce churn by up to **10–14%**.

---

## 🧩 Key Features

* Comprehensive **Exploratory Data Analysis (EDA)** with rich visualizations
* Automated **data preprocessing and encoding**
* **SMOTE balancing** to address class imbalance
* **Hyperparameter tuning** using GridSearchCV
* **Model comparison** (Logistic Regression, Random Forest, Gradient Boosting)
* **Feature importance analysis** for explainability
* **Strategic churn reduction recommendations**

---

## 📂 Project Structure

```
📁 Telecom-Churn-Prediction
│
├── TelecomCustomerChurn.csv           # Input dataset
├── Telecom_ChurnCleaned.csv           # Processed dataset (auto-generated)
├── churn_prediction.ipynb             # Main Jupyter Notebook or Python script
├── README.md                          # Project documentation (this file)
└── requirements.txt                   # Dependencies list
```

---

## ⚙️ Installation

### 1. Clone this repository

```bash
git clone https://github.com/yourusername/telecom-churn-prediction.git
cd telecom-churn-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install scikit-learn imbalanced-learn matplotlib seaborn pandas numpy
```

---

## 🧾 Dataset

**File:** `TelecomCustomerChurn.csv`
Each row represents a customer’s profile and subscription details.

### Key Columns

* `customerID` – Unique identifier
* `gender`, `SeniorCitizen`, `Partner`, `Dependents` – Demographics
* `Tenure`, `Contract`, `PaymentMethod` – Customer lifecycle
* `MonthlyCharges`, `TotalCharges` – Financial indicators
* `Churn` – Target variable (Yes/No)

---

## 🔍 Workflow

### 1. **Data Loading & Exploration**

* Reads and inspects dataset
* Handles missing values, converts data types
* Prints basic stats and churn distribution

### 2. **Exploratory Data Analysis (EDA)**

* Visualizes churn by contract type, tenure, charges, etc.
* Detects key churn drivers through visual correlations

### 3. **Data Preprocessing**

* Converts categorical to numeric using `OrdinalEncoder`
* Fills missing numeric values with median
* Saves cleaned dataset to `Telecom_ChurnCleaned.csv`

### 4. **Balancing with SMOTE**

* Handles class imbalance using **Synthetic Minority Oversampling Technique**
* Visualizes before/after distribution

### 5. **Modeling**

* Splits dataset (80/20 train-test)
* Scales features using `StandardScaler`
* Trains multiple models:

  * Logistic Regression
  * Random Forest (with GridSearchCV hyperparameter tuning)
  * Gradient Boosting

### 6. **Evaluation**

* Metrics: Accuracy, F1-score, ROC-AUC
* Confusion matrices and ROC curves for all models
* Feature importance visualization

### 7. **Business Insights**

* Identifies top churn drivers
* Proposes actionable strategies with estimated impact

---

## 📊 Results Summary

| Model               | Accuracy (%) | ROC-AUC  | F1-Score |
| ------------------- | ------------ | -------- | -------- |
| Logistic Regression | ~86          | 0.91     | 0.84     |
| Random Forest       | **~89**      | **0.93** | **0.87** |
| Gradient Boosting   | ~88          | 0.92     | 0.86     |

🏆 **Best Model:** Random Forest (Highest Accuracy & Stability)

---

## 💡 Top 5 Churn Drivers & Mitigation

| Factor           | Insight                                 | Action                          | Impact         | Priority  |
| ---------------- | --------------------------------------- | ------------------------------- | -------------- | --------- |
| Contract Type    | Month-to-month customers have 42% churn | Incentivize long-term contracts | 3–4% reduction | 🔴 High   |
| Tenure           | High churn in first 6–12 months         | Onboarding & retention program  | 2–3% reduction | 🔴 High   |
| Monthly Charges  | Higher bills increase churn             | Discounts, bundle pricing       | 2–3% reduction | 🟡 Medium |
| Tech Support     | Lack of support correlates with churn   | Promote support add-ons         | 1–2% reduction | 🟡 Medium |
| Internet Service | Fiber customers churn more              | Improve service quality         | 1–2% reduction | 🟡 Medium |

✅ **Expected overall churn reduction: 9–14%**

---

## 📈 Visualizations

The project automatically generates:

* Churn distribution pie chart
* Contract vs Churn bar plots
* Tenure & Charges histograms
* ROC curve comparison
* Confusion matrices for each model
* Feature importance bar chart

---

## 🧰 Requirements

* Python 3.8+
* Libraries:

  ```
  scikit-learn
  imbalanced-learn
  matplotlib
  seaborn
  pandas
  numpy
  ```

---

## 🚀 How to Run

1. Place `TelecomCustomerChurn.csv` in the working directory.
2. Run the notebook or script:

   ```bash
   python churn_prediction.py
   ```

   or open `churn_prediction.ipynb` in Jupyter.
3. The script will generate:

   * Visual outputs
   * Cleaned dataset
   * Performance reports

---

## 🧭 Future Improvements

* Add XGBoost and LightGBM for better performance
* Implement SHAP for interpretability
* Deploy as a Flask web app
* Automate churn risk scoring dashboard

---
