import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Churn | Retention Dashboard",
    layout="wide",
    page_icon="📉",
    initial_sidebar_state="collapsed"
)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("telecom_churn.csv")
    df["date_of_registration"] = pd.to_datetime(df["date_of_registration"])
    df["tenure_days"] = (df["date_of_registration"].max() - df["date_of_registration"]).dt.days
    df["data_used"] = df["data_used"].abs()
    return df

df = load_data()

# ================= STATE–CITY MAP =================
state_city_map = (
    df.groupby("state")["city"]
    .unique()
    .apply(list)
    .to_dict()
)

# ================= AVERAGE LOOKUP (IMPORTANT FIX) =================
avg_lookup = (
    df.groupby(["state", "city", "telecom_partner"])
    .agg({
        "tenure_days": "mean",
        "calls_made": "mean",
        "sms_sent": "mean",
        "data_used": "mean",
        "estimated_salary": "mean"
    })
    .reset_index()
)

global_avg = df[[
    "tenure_days",
    "calls_made",
    "sms_sent",
    "data_used",
    "estimated_salary"
]].mean()

# ================= MODEL PIPELINE =================
@st.cache_resource
def train_model(df):
    X = df.drop(columns=["customer_id", "churn", "date_of_registration"])
    y = df["churn"]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols)
        ],
        verbose_feature_names_out=False
    )

    pipeline = ImbPipeline(steps=[
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "conf_matrix": confusion_matrix(y_test, y_pred)
    }

    feat_imp = pd.DataFrame({
        "Feature": num_cols + cat_cols,
        "Importance": pipeline.named_steps["model"].feature_importances_
    }).sort_values("Importance", ascending=False)

    return pipeline, metrics, feat_imp, num_cols, cat_cols

pipeline, metrics, feat_imp, num_cols, cat_cols = train_model(df)

# ================= HEADER =================
st.title("📞 Telecom Churn Intelligence Dashboard")
st.markdown("---")

tab1, tab3 = st.tabs(["📊 Executive Overview", "⚡ Churn Simulator"])

# ================= TAB 1 =================
with tab1:
    churn_rate = df["churn"].mean() * 100
    retention = 100 - churn_rate

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Retention Rate", f"{retention:.1f}%")
    c3.metric("Churn Rate", f"{churn_rate:.1f}%")
    c4.metric("Model Accuracy", f"{metrics['accuracy']*100:.1f}%")

    st.subheader("Churn by Telecom Partner")
    fig = px.bar(
        df[df["churn"] == 1]["telecom_partner"].value_counts().reset_index(),
        x="telecom_partner", y="count",
        labels={"count": "Churned Customers"},
        color="count"
    )
    st.plotly_chart(fig, use_container_width=True)


# ================= TAB 3 =================
with tab3:
    st.subheader("⚡ Customer Churn Risk Simulator")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", df["gender"].unique())
        telecom_partner = st.selectbox("Telecom Partner", df["telecom_partner"].unique())
        state = st.selectbox("State", sorted(state_city_map.keys()))
        city = st.selectbox("City", sorted(state_city_map[state]))
        age = st.number_input("Age", 18, 80, 35)
        dependents = st.selectbox("Dependents", sorted(df["num_dependents"].unique()))
        pincode = st.number_input("Pincode", 100000, 999999, 400001)

    # ===== AVERAGE-BASED AUTO CALCULATION =====
    row = avg_lookup[
        (avg_lookup["state"] == state) &
        (avg_lookup["city"] == city) &
        (avg_lookup["telecom_partner"] == telecom_partner)
    ]

    if not row.empty:
        tenure_days = int(row["tenure_days"].iloc[0])
        calls = int(row["calls_made"].iloc[0])
        sms = int(row["sms_sent"].iloc[0])
        data_used = float(row["data_used"].iloc[0])
        salary = float(row["estimated_salary"].iloc[0])
    else:
        tenure_days = int(global_avg["tenure_days"])
        calls = int(global_avg["calls_made"])
        sms = int(global_avg["sms_sent"])
        data_used = float(global_avg["data_used"])
        salary = float(global_avg["estimated_salary"])

    with col2:
        st.markdown("### 📊 Auto-Calculated Average Profile")
        st.metric("Tenure (Days)", tenure_days)
        st.metric("Calls Made", calls)
        st.metric("SMS Sent", sms)
        st.metric("Data Used (MB)", f"{data_used:.0f}")
        st.metric("Estimated Salary", f"₹{salary:,.0f}")

        input_df = pd.DataFrame([{
            "gender": gender,
            "telecom_partner": telecom_partner,
            "city": city,
            "state": state,
            "age": age,
            "num_dependents": dependents,
            "tenure_days": tenure_days,
            "calls_made": calls,
            "sms_sent": sms,
            "data_used": data_used,
            "estimated_salary": salary,
            "pincode": pincode
        }])

        prob = pipeline.predict_proba(input_df)[0][1]
        st.metric("Churn Probability", f"{prob*100:.1f}%")

        if prob > 0.7:
            st.error("🚨 HIGH RISK – Immediate retention action required")
        elif prob > 0.4:
            st.warning("⚠️ AT RISK – Consider targeted offers")
        else:
            st.success("✅ LOW RISK – Customer likely to stay")
