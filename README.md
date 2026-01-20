# 📞 Customer Churn Prediction & Retention Dashboard

An end-to-end **Machine Learning + Streamlit** project that predicts **telecom customer churn**, visualizes key business metrics, and provides an **interactive churn risk simulator** for data-driven decision-making.

---

## 🚀 Project Overview

Customer churn is one of the biggest challenges in the telecom industry, directly impacting revenue and growth. This project leverages **machine learning** to identify customers at risk of leaving and presents actionable insights through an **interactive Streamlit dashboard**.

### 🎯 Key Objectives

- 📈 Predict customer churn with high accuracy
- ⚖️ Handle class imbalance using SMOTE
- 📊 Visualize churn trends by telecom partner and geography
- 🎛️ Provide real-time churn probability predictions for individual customers
- 💼 Enable data-driven retention strategies

---

## 🧠 Machine Learning Approach

### Model Architecture
- **Algorithm:** Random Forest Classifier
- **Estimators:** 150 trees with max depth of 15
- **Class Imbalance Handling:** SMOTE (Synthetic Minority Over-sampling Technique)
- **Feature Encoding:** Ordinal Encoder for categorical variables
- **Pipeline:** scikit-learn ColumnTransformer + imbalanced-learn Pipeline

### Evaluation Metrics
- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)

---

## 📊 Dashboard Features

### 📋 Executive Overview
- Total active customers
- Retention rate percentage
- Churn rate percentage
- Model accuracy score
- Interactive churn distribution by telecom partner

### 💡 Business Insights
- High-level churn trends with business context
- Partner performance comparison
- Geographic churn patterns

### 🎮 Churn Risk Simulator
**Interactive tool** that predicts churn probability for custom customer profiles.

**User Input Fields:**
- Gender
- Telecom Partner
- State & City (dependent dropdown)
- Age
- Number of Dependents
- Pincode

**Auto-calculated Averages:**
- Tenure (Days)
- Calls Made
- SMS Sent
- Data Used (MB)
- Estimated Salary

**Risk Classification:**
- ✅ **Low Risk** (0-30%)
- ⚠️ **At Risk** (30-70%)
- 🚨 **High Risk** (70-100%)

---

## 🗂️ Dataset

**Location:** `data/telecom_churn.csv`

### Key Features
| Column | Description | Type |
|--------|-------------|------|
| `customer_id` | Unique customer identifier | String |
| `gender` | Customer gender | Categorical |
| `telecom_partner` | Service provider | Categorical |
| `state`, `city` | Location information | Categorical |
| `pincode` | Postal code | Numeric |
| `age` | Customer age | Numeric |
| `num_dependents` | Number of dependents | Numeric |
| `date_of_registration` | Account registration date | DateTime |
| `tenure_days` | Days as customer | Numeric |
| `calls_made` | Total calls made | Numeric |
| `sms_sent` | Total SMS sent | Numeric |
| `data_used` | Data consumption (MB) | Numeric |
| `estimated_salary` | Annual salary estimate | Numeric |
| `churn` | **Target: Customer left (1) or stayed (0)** | Binary |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.9+ |
| **ML Framework** | scikit-learn |
| **Data Processing** | pandas, numpy |
| **Class Imbalance** | imbalanced-learn (SMOTE) |
| **Visualization** | Plotly |
| **Web App** | Streamlit |

### Dependencies
See [requirements.txt](requirements.txt) for complete package list.

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Git

### Quick Start

**1️⃣ Clone the Repository**
```bash
git clone https://github.com/sathish-2424/Customer-Churn-Prediction
cd Customer-Churn-Prediction
```

**2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

**3️⃣ Run the Application**
```bash
streamlit run streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

### Running in Docker (Optional)
```bash
docker build -t churn-dashboard .
docker run -p 8501:8501 churn-dashboard
```

---

## 📁 Project Structure

```
Customer-Churn-Prediction/
├── streamlit_app.py                    # Main dashboard application
├── data/
│   └── telecom_churn.csv               # Dataset (customer data)
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
├── .devcontainer/                      # VSCode devcontainer config
└── (optional) Additional notebooks/scripts
```

### File Descriptions
- **streamlit_app.py** - Streamlit web application with interactive dashboard and churn simulator
- **data/telecom_churn.csv** - Raw telecom customer dataset
- **requirements.txt** - All required Python packages and versions

---

## 🎯 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | ~80% |
| **Precision** | High |
| **Recall** | Balanced |
| **F1-Score** | Optimized |
| **Class Balance** | ✅ SMOTE Applied |
| **Production Ready** | ✅ Yes |

*Note: Performance metrics vary based on train-test split. For detailed evaluation, run the model training pipeline.*

---

## � Business Value & Impact

### For Telecom Companies
- 🎯 **Proactive Retention** - Identify high-risk customers before they churn
- 💰 **Revenue Protection** - Reduce customer acquisition costs by retaining existing customers
- 📊 **Data-Driven Strategy** - Design targeted retention offers based on churn risk levels
- 📈 **Performance Metrics** - Track retention effectiveness over time

### Use Cases
1. **Retention Campaigns** - Target at-risk customers with special offers
2. **Resource Allocation** - Prioritize support for high-risk segments
3. **Pricing Strategy** - Adjust pricing for retention-sensitive customers
4. **Product Development** - Identify service gaps from churn patterns

---

## 📊 Power BI Integration

The same telecom dataset used for machine learning is also utilized in an interactive **Power BI dashboard**.

### Power BI Dashboard Includes
- 📋 Executive KPIs (Total Customers, Churn Rate, Retention Rate)
- 🗺️ Partner-wise and region-wise churn analysis
- 📱 Usage behavior insights and trends
- 👥 Customer segmentation for strategic decisions

### Benefits
- **Data Consistency** - Single source of truth for business analytics and ML
- **Dual Perspective** - Business analytics (Power BI) + Predictive analytics (ML)
- **Holistic Insights** - Understand both what happened and what will happen

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📧 Contact & Support

For questions, suggestions, or issues:
- Open an issue on GitHub
- Contact the project maintainer

---

## 🙏 Acknowledgments

- Dataset sourced from telecom industry
- Built with ❤️ using Python, scikit-learn, and Streamlit
- Special thanks to the open-source community

---

**⭐ If you found this project helpful, please consider giving it a star!**

