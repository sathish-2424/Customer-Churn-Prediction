---

# 📞 Customer Churn Prediction & Retention Dashboard

An end-to-end **Machine Learning + Streamlit** project that predicts **telecom customer churn**, visualizes key business metrics, and provides an **interactive churn risk simulator** for decision-making.

---

## 🚀 Project Overview

Customer churn is one of the biggest challenges in the telecom industry.
This project uses **machine learning** to identify customers likely to leave and presents insights through an **interactive Streamlit dashboard**.

### Key Goals

* Predict customer churn accurately
* Handle imbalanced data using SMOTE
* Visualize churn trends by telecom partner
* Provide real-time churn probability for individual customers

---

## 🧠 Machine Learning Approach

* **Model:** Random Forest Classifier
* **Class Imbalance Handling:** SMOTE (Synthetic Minority Over-sampling Technique)
* **Categorical Encoding:** Ordinal Encoder
* **Evaluation Metrics:**

  * Accuracy
  * Confusion Matrix
  * Classification Report

---

## 📊 Dashboard Features

### 1️⃣ Executive Overview

* Total customers
* Retention rate
* Churn rate
* Model accuracy
* Churn by telecom partner (interactive bar chart)

### 2️⃣ Business Insights

* High-level churn trends (non-technical)

### 3️⃣ Churn Simulator

* User inputs:

  * Gender
  * Telecom Partner
  * State & City (dependent dropdown)
  * Age
  * Dependents
  * Pincode
* Automatically calculates **average values** for:

  * Tenure (Days)
  * Calls Made
  * SMS Sent
  * Data Used (MB)
  * Estimated Salary
* Outputs **churn probability** with risk labels:

  * ✅ Low Risk
  * ⚠️ At Risk
  * 🚨 High Risk

---

## 🗂️ Dataset

**File:** `telecom_churn.csv`

Key columns:

* `gender`
* `telecom_partner`
* `state`, `city`, `pincode`
* `age`
* `num_dependents`
* `date_of_registration`
* `tenure_days`
* `calls_made`
* `sms_sent`
* `data_used`
* `estimated_salary`
* `churn` (target variable)

---

## 🛠️ Tech Stack

| Category      | Tools                          |
| ------------- | ------------------------------ |
| Language      | Python 3                       |
| ML            | scikit-learn, imbalanced-learn |
| Data          | pandas, numpy                  |
| Visualization | Plotly                         |
| App Framework | Streamlit                      |

---

## 📦 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/sathish-2424/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

---

## 📁 Project Structure

```
Customer-Churn-Prediction/
│
├── streamlit_app.py          # Streamlit dashboard
├── telecom_churn.csv         # Dataset
├── telecom_churn_prediction.py
├── telecom_churn_prediction.ipynb
├── requirements.txt
├── README.md
└── .devcontainer/
```

---

## 🎯 Model Performance (Current)

* **Accuracy:** ~80%
* **Balanced Prediction:** Yes (SMOTE applied)
* **Business-ready:** ✔️

---

## 💡 Business Value

* Identifies high-risk customers **before they churn**
* Helps telecom companies:

  * Reduce revenue loss
  * Design targeted retention offers
  * Make data-driven decisions

---
## 📊 Power BI Dashboard

The same telecom dataset used for machine learning is also used to build an
interactive **Power BI dashboard**.

The dashboard provides:
- Executive KPIs (Customers, Churn Rate, Retention Rate)
- Partner-wise and region-wise churn analysis
- Usage behavior insights
- Customer segmentation for business decisions

This ensures consistency between **business analytics** (Power BI) and
**predictive analytics** (Machine Learning).

