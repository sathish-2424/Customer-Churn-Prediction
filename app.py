import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt



st.set_page_config(page_title="Telecom Customer Churn — Random Forest", layout="wide", page_icon="📈")



@st.cache_data
def load_csv(path_or_buffer):
    return pd.read_csv(path_or_buffer)

@st.cache_data
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    for col in ["totalcharges", "monthlycharges", "tenure"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)
    return df

@st.cache_data
def preprocess(df: pd.DataFrame):
    if "churn" not in df.columns:
        raise ValueError("Target column 'churn' missing.")
    y = df["churn"].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)
    X = df.drop(["customerid", "churn"], axis=1, errors="ignore")
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if cat_cols:
        X[cat_cols] = enc.fit_transform(X[cat_cols].fillna("missing"))
    X = X.fillna(X.median(numeric_only=True))
    return X, y, enc, cat_cols

@st.cache_resource
def train_model(X, y, random_state=42):
    X_res, y_res = SMOTE(random_state=random_state).fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True)
    }
    return model, scaler, metrics, X.columns, y_test, y_pred


def encode_input(df, encoder, cat_cols, train_cols):
    df = df.copy()
    for col in cat_cols:
        if col not in df:
            df[col] = "missing"
    df[cat_cols] = encoder.transform(df[cat_cols].fillna("missing"))
    for col in train_cols:
        if col not in df:
            df[col] = 0
    return df[train_cols]



st.sidebar.title("⚙️ Controls")
page = st.sidebar.radio("Navigate", ["Data Overview", "Modeling", "Predictions", "Results"])

uploaded = st.sidebar.file_uploader("📁 Upload CSV", type=["csv"])
use_default = st.sidebar.checkbox("Use Default Dataset", value=False)

if use_default:
    try:
        df = load_csv("TelecomCustomerChurn.csv")
        st.sidebar.success("Loaded default dataset")
    except FileNotFoundError:
        st.sidebar.error("Default file not found.")
        st.stop()
elif uploaded is not None:
    df = load_csv(uploaded)
    st.sidebar.success(f"Loaded {uploaded.name}")
else:
    st.info("Upload a dataset or use default sample to proceed.")
    st.stop()



df = clean_data(df)
X, y, encoder, cat_cols = preprocess(df)
model, scaler, metrics, train_cols, y_test, y_pred = train_model(X, y)

st.session_state.update({"model": model, "scaler": scaler, "encoder": encoder, "cat_cols": cat_cols, "train_cols": train_cols, "metrics": metrics, "y_test": y_test, "y_pred": y_pred})


if page == "Data Overview":
    st.title("📊 Data Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Total Columns", len(df.columns))
    col3.metric("Churn Rate", f"{y.mean() * 100:.1f}%")

    st.markdown("---")
    st.dataframe(df.head(10), use_container_width=True)

    if st.checkbox("Show Descriptive Stats", value=True):
        st.dataframe(df.describe().round(2), use_container_width=True)

    if st.checkbox("Show Numeric Distributions", value=False):
        cols = df.select_dtypes(include=np.number).columns.tolist()
        selected = st.multiselect("Select Columns", cols, default=cols[:2])
        for c in selected:
            fig, ax = plt.subplots(figsize=(5, 2.5))
            ax.hist(df[c], bins=25, color="#74b9ff", edgecolor="black")
            ax.set_title(f"Distribution: {c}", fontsize=10)
            st.pyplot(fig, use_container_width=True)


elif page == "Modeling":
    st.title("🤖 Random Forest Model Performance")

    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    st.markdown("---")

    st.subheader("Confusion Matrix")
    cm = metrics["confusion_matrix"]

    # Reduced and centered confusion matrix
    fig, ax = plt.subplots(figsize=(3.5, 2.8))  # Reduced chart size

    # Display heatmap with annotations
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=9)

    # Labels and title
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title("Confusion Matrix", fontsize=11, fontweight="bold")

    # Center chart in view
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=False)
    st.markdown("</div>", unsafe_allow_html=True)



    st.markdown("---")
    st.subheader("Top 15 Feature Importances")
    feat_imp = pd.DataFrame({"Feature": train_cols, "Importance": model.feature_importances_})\
        .sort_values("Importance", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(feat_imp["Feature"], feat_imp["Importance"], color="#55efc4")
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    st.pyplot(fig, use_container_width=True)


elif page == "Predictions":
    st.title("🎯 Churn Prediction")
    mode = st.radio("Prediction Mode", ["Manual", "Batch"])

    if mode == "Manual":
        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)
            gender = c1.selectbox("Gender", df["gender"].unique())
            senior = c1.selectbox("Senior Citizen", df["seniorcitizen"].unique())
            contract = c2.selectbox("Contract", df["contract"].unique())
            payment = c2.selectbox("Payment Method", df["paymentmethod"].unique())
            paperless = c3.selectbox("Paperless Billing", df["paperlessbilling"].unique())
            monthly = c3.number_input("Monthly Charges", min_value=0.0, value=float(df["monthlycharges"].median()))
            tenure = st.number_input("Tenure (Months)", min_value=0, value=int(df["tenure"].median()))
            total = st.number_input("Total Charges", min_value=0.0, value=float(df["totalcharges"].median()))
            predict = st.form_submit_button("Predict Churn")

        if predict:
            data = pd.DataFrame([{ "gender": gender, "seniorcitizen": senior, "contract": contract, "paymentmethod": payment, "paperlessbilling": paperless, "monthlycharges": monthly, "tenure": tenure, "totalcharges": total }])
            enc_data = encode_input(data, encoder, cat_cols, train_cols)
            scaled = scaler.transform(enc_data)
            pred = model.predict(scaled)[0]
            proba = model.predict_proba(scaled)[0]

            st.markdown("---")
            col1, col2 = st.columns([1, 1])
            if pred == 1:
                col1.error(f"⚠️ High Churn Risk ({proba[1]*100:.1f}% chance)")
            else:
                col1.success(f"✅ Low Churn Risk ({proba[0]*100:.1f}% chance)")
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.pie([proba[0], proba[1]], labels=["No Churn", "Churn"], autopct="%1.1f%%", colors=["#55efc4", "#ff7675"])
            col2.pyplot(fig, use_container_width=True)

    else:
        st.subheader("📤 Batch Predictions")
        batch = st.file_uploader("Upload CSV", type=["csv"])
        if batch is not None and st.button("Run Predictions"):
            batch_df = clean_data(pd.read_csv(batch))
            preds, probs = [], []
            for _, row in batch_df.iterrows():
                r_df = encode_input(pd.DataFrame([row]), encoder, cat_cols, train_cols)
                scaled = scaler.transform(r_df)
                preds.append(model.predict(scaled)[0])
                probs.append(model.predict_proba(scaled)[0][1])
            batch_df["prediction"] = ["Churn" if p == 1 else "No Churn" for p in preds]
            batch_df["churn_probability"] = probs
            st.dataframe(batch_df, use_container_width=True)
            st.download_button("Download Results", batch_df.to_csv(index=False), "predictions.csv")


elif page == "Results":
    st.title("📈 Model Summary & Results")

    # Accuracy metric
    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    st.markdown("---")

    # Classification report
    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(metrics['report']).T.round(3), use_container_width=True)

    st.markdown("---")
    
    # Churn Distribution (reduced and centered)
    st.subheader("Churn Distribution")
    churn_counts = y.value_counts()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))  # Reduced chart size
    wedges, texts, autotexts = ax.pie(
        churn_counts,
        labels=["No Churn", "Churn"],
        autopct="%1.1f%%",
        colors=["#55efc4", "#ff7675"],
        startangle=90,
        textprops={"fontsize": 9}
    )
    ax.set_title("Customer Churn Distribution", fontsize=11, fontweight="bold")

    # Center chart
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=False)
    st.markdown("</div>", unsafe_allow_html=True)