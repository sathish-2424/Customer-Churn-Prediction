import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Telecom Churn Intelligence",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("data/telecom_churn.csv")

    # Ensure tenure_days exists
    if "tenure_days" not in df.columns:
        if "date_of_registration" in df.columns:
            df["date_of_registration"] = pd.to_datetime(df["date_of_registration"])
            df["tenure_days"] = (
                df["date_of_registration"].max() - df["date_of_registration"]
            ).dt.days
        elif "tenure" in df.columns:
            df.rename(columns={"tenure": "tenure_days"}, inplace=True)
        else:
            df["tenure_days"] = 365

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

# ================= AVERAGE LOOKUP =================
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

# ================= MODEL =================
@st.cache_resource
def train_model(df):
    X = df.drop(columns=["customer_id", "churn", "date_of_registration"], errors="ignore")
    y = df["churn"]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ), cat_cols)
        ],
        verbose_feature_names_out=False
    )

    pipeline = ImbPipeline(steps=[
        ("prep", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline.fit(Xtr, ytr)
    preds = pipeline.predict(Xte)

    metrics = {
        "accuracy": accuracy_score(yte, preds),
        "report": classification_report(yte, preds, output_dict=True),
        "confusion": confusion_matrix(yte, preds)
    }

    feat_imp = pd.DataFrame({
        "Feature": num_cols + cat_cols,
        "Importance": pipeline.named_steps["model"].feature_importances_
    }).sort_values("Importance", ascending=False)

    return pipeline, metrics, feat_imp

pipeline, metrics, feat_imp = train_model(df)

# ================= HEADER =================
st.title("📞 Telecom Customer Churn")
st.markdown("---")

# ================= KPI ROW =================
churn_rate = df["churn"].mean() * 100
retention_rate = 100 - churn_rate

k1, k2, k3, k4 = st.columns(4)
k1.metric("👥 Customers", f"{len(df):,}")
k2.metric("🟢 Retention", f"{retention_rate:.1f}%")
k3.metric("🔴 Churn", f"{churn_rate:.1f}%")
k4.metric("🎯 Model Accuracy", f"{metrics['accuracy']*100:.1f}%")

st.markdown("---")

# ================= CHURN BY PARTNER =================
st.subheader("📉 Churn Rate by Telecom Partner")

partner_churn = (
    df.groupby("telecom_partner")["churn"]
      .mean()
      .mul(100)
      .reset_index(name="churn_rate")
      .sort_values("churn_rate", ascending=False)
)

fig_partner = px.bar(
    partner_churn,
    x="telecom_partner",
    y="churn_rate",
    text="churn_rate",
    color="churn_rate",
    color_continuous_scale=["#f1c40f", "#e74c3c"],
    labels={"churn_rate": "Churn Rate (%)"}
)

fig_partner.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig_partner.update_layout(height=380, yaxis_range=[0, 100], xaxis_title="")

st.plotly_chart(fig_partner, use_container_width=True)

st.markdown("---")

# ================= SIMULATOR =================
st.subheader("⚡ Customer Churn Risk Simulator")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", df["gender"].unique())
    telecom_partner = st.selectbox("Telecom Partner", df["telecom_partner"].unique())
    state = st.selectbox("State", sorted(state_city_map.keys()))
    city = st.selectbox("City", sorted(state_city_map[state]))
    age = st.slider("Age", 18, 80, 35)
    dependents = st.selectbox("Dependents", df["num_dependents"].unique())

row = avg_lookup[
    (avg_lookup["state"] == state) &
    (avg_lookup["city"] == city) &
    (avg_lookup["telecom_partner"] == telecom_partner)
]

profile = row.iloc[0] if not row.empty else global_avg

input_df = pd.DataFrame([{
    "gender": gender,
    "telecom_partner": telecom_partner,
    "state": state,
    "city": city,
    "age": age,
    "num_dependents": dependents,
    "tenure_days": int(profile["tenure_days"]),
    "calls_made": int(profile["calls_made"]),
    "sms_sent": int(profile["sms_sent"]),
    "data_used": float(profile["data_used"]),
    "estimated_salary": float(profile["estimated_salary"]),
    "pincode": 600001
}])

prob = pipeline.predict_proba(input_df)[0][1]

with col2:
    st.metric("🎯 Churn Probability", f"{prob*100:.1f}%")

    fig_risk = px.pie(
        values=[prob, 1 - prob],
        names=["Churn Risk", "Retention"],
        hole=0.65,
        color_discrete_map={
            "Churn Risk": "#e74c3c",
            "Retention": "#2ecc71"
        }
    )
    fig_risk.update_layout(height=320)
    st.plotly_chart(fig_risk, use_container_width=True)

    if prob > 0.7:
        st.error("🚨 HIGH RISK – Immediate action required")
    elif prob > 0.4:
        st.warning("⚠️ MEDIUM RISK – Targeted offers suggested")
    else:
        st.success("✅ LOW RISK – Customer likely to stay")
