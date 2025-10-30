import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

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
    # SMOTE for handling class imbalance
    smote = SMOTE(random_state=random_state, k_neighbors=3)
    X_res, y_res = smote.fit_resample(X, y)
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, stratify=y_res, random_state=random_state
    )
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=150, 
        max_depth=15, 
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Predictions and metrics
    y_pred = model.predict(X_test_scaled)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True)
    }
    
    return model, scaler, metrics, X.columns.tolist(), y_test, y_pred

def encode_input(df, encoder, cat_cols, train_cols):
    """Encode and prepare input data for prediction"""
    df = df.copy()
    
    # Ensure all categorical columns exist
    for col in cat_cols:
        if col not in df.columns:
            df[col] = "missing"
    
    # Encode categorical variables
    if cat_cols:
        df[cat_cols] = encoder.transform(df[cat_cols].fillna("missing"))
    
    # Ensure all training columns exist
    for col in train_cols:
        if col not in df.columns:
            df[col] = 0
    
    return df[train_cols]

# ===== SIDEBAR & DATA LOADING =====
st.sidebar.title("⚙️ Controls")
page = st.sidebar.radio("Navigate", ["Data Overview", "Modeling", "Predictions", "Results"])

uploaded = st.sidebar.file_uploader("📁 Upload CSV", type=["csv"])
use_default = st.sidebar.checkbox("Use Default Dataset", value=False)

# Load dataset
if use_default:
    try:
        df = load_csv("TelecomCustomerChurn.csv")
        st.sidebar.success("✅ Loaded default dataset")
    except FileNotFoundError:
        st.sidebar.error("❌ Default file not found.")
        st.stop()
elif uploaded is not None:
    df = load_csv(uploaded)
    st.sidebar.success(f"✅ Loaded {uploaded.name}")
else:
    st.info("📌 Upload a dataset or use default sample to proceed.")
    st.stop()

# ===== PREPROCESSING & MODEL TRAINING =====
df = clean_data(df)
X, y, encoder, cat_cols = preprocess(df)
model, scaler, metrics, train_cols, y_test, y_pred = train_model(X, y)

# Store in session state
st.session_state.update({
    "model": model,
    "scaler": scaler,
    "encoder": encoder,
    "cat_cols": cat_cols,
    "train_cols": train_cols,
    "metrics": metrics,
    "y_test": y_test,
    "y_pred": y_pred,
    "df": df,
    "y": y
})

# ===== PAGE: DATA OVERVIEW =====
if page == "Data Overview":
    st.title("📊 Data Overview")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Total Columns", len(df.columns))
    col3.metric("Churn Rate", f"{y.mean() * 100:.1f}%")
    
    st.markdown("---")
    st.subheader("Dataset Sample")
    st.dataframe(df.head(10), use_container_width=True)
    
    if st.checkbox("Show Descriptive Stats", value=True):
        st.subheader("Statistical Summary")
        st.dataframe(df.describe().round(2), use_container_width=True)
    
    if st.checkbox("Show Numeric Distributions", value=False):
        cols = df.select_dtypes(include=np.number).columns.tolist()
        selected = st.multiselect("Select Columns", cols, default=cols[:2] if len(cols) >= 2 else cols)
        for c in selected:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(df[c], bins=25, color="#74b9ff", edgecolor="black", alpha=0.7)
            ax.set_title(f"Distribution: {c}", fontsize=12, fontweight="bold")
            ax.set_xlabel(c)
            ax.set_ylabel("Frequency")
            st.pyplot(fig, use_container_width=True)

# ===== PAGE: MODELING =====
elif page == "Modeling":
    st.title("🤖 Random Forest Model Performance")
    
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    col2.metric("Test Samples", len(y_test))
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        cm = metrics["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", 
                       color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12, fontweight="bold")
        
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Churn", "Churn"])
        ax.set_yticklabels(["No Churn", "Churn"])
        ax.set_title("Confusion Matrix", fontsize=12, fontweight="bold")
        
        plt.colorbar(im, ax=ax)
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        st.subheader("Classification Metrics")
        report_df = pd.DataFrame(metrics['report']).T
        st.dataframe(report_df.round(3), use_container_width=True)
    
    st.markdown("---")
    st.subheader("Top 15 Feature Importances")
    feat_imp = pd.DataFrame({
        "Feature": train_cols,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(feat_imp["Feature"], feat_imp["Importance"], color="#55efc4", edgecolor="black")
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title("Feature Importance in Churn Prediction", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
               ha='left', va='center', fontsize=9)
    
    st.pyplot(fig, use_container_width=True)

# ===== PAGE: PREDICTIONS =====
elif page == "Predictions":
    st.title("🎯 Churn Prediction")
    
    mode = st.radio("Prediction Mode", ["Manual Input", "Batch Upload"])
    
    if mode == "Manual Input":
        st.subheader("Enter Customer Information")
        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)
            
            gender = c1.selectbox("Gender", sorted(df["gender"].unique()))
            senior = c1.selectbox("Senior Citizen", sorted(df["seniorcitizen"].unique()))
            contract = c2.selectbox("Contract Type", sorted(df["contract"].unique()))
            payment = c2.selectbox("Payment Method", sorted(df["paymentmethod"].unique()))
            paperless = c3.selectbox("Paperless Billing", sorted(df["paperlessbilling"].unique()))
            
            monthly = c1.number_input("Monthly Charges ($)", min_value=0.0, 
                                     value=float(df["monthlycharges"].median()), step=0.01)
            tenure = c2.number_input("Tenure (Months)", min_value=0, 
                                    value=int(df["tenure"].median()), step=1)
            total = c3.number_input("Total Charges ($)", min_value=0.0, 
                                   value=float(df["totalcharges"].median()), step=0.01)
            
            predict_btn = st.form_submit_button("🔮 Predict Churn", use_container_width=True)
        
        if predict_btn:
            try:
                # Prepare input data
                input_data = pd.DataFrame([{
                    "gender": gender,
                    "seniorcitizen": senior,
                    "contract": contract,
                    "paymentmethod": payment,
                    "paperlessbilling": paperless,
                    "monthlycharges": monthly,
                    "tenure": tenure,
                    "totalcharges": total
                }])
                
                # Encode and scale
                enc_data = encode_input(input_data, encoder, cat_cols, train_cols)
                scaled = scaler.transform(enc_data)
                
                # Predict
                pred = model.predict(scaled)[0]
                proba = model.predict_proba(scaled)[0]
                
                st.markdown("---")
                col1, col2 = st.columns([1.5, 1])
                
                with col1:
                    if pred == 1:
                        st.error(f"⚠️ HIGH CHURN RISK - {proba[1]*100:.1f}% probability", icon="🚨")
                    else:
                        st.success(f"✅ LOW CHURN RISK - {proba[0]*100:.1f}% probability", icon="💚")
                
                with col2:
                    fig, ax = plt.subplots(figsize=(4, 4))
                    colors = ["#55efc4", "#ff7675"]
                    wedges, texts, autotexts = ax.pie(
                        [proba[0], proba[1]],
                        labels=["No Churn", "Churn"],
                        autopct="%1.1f%%",
                        colors=colors,
                        startangle=90,
                        textprops={"fontsize": 10}
                    )
                    for autotext in autotexts:
                        autotext.set_color("white")
                        autotext.set_fontweight("bold")
                    ax.set_title("Prediction Probability", fontsize=11, fontweight="bold")
                    st.pyplot(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")
    
    else:  # Batch mode
        st.subheader("📤 Batch Predictions")
        batch_file = st.file_uploader("Upload CSV for batch predictions", type=["csv"], key="batch_uploader")
        
        if batch_file is not None:
            if st.button("🚀 Run Batch Predictions", use_container_width=True):
                try:
                    batch_df = clean_data(pd.read_csv(batch_file))
                    preds, probs = [], []
                    
                    with st.spinner("Processing predictions..."):
                        for _, row in batch_df.iterrows():
                            r_df = encode_input(pd.DataFrame([row]), encoder, cat_cols, train_cols)
                            scaled = scaler.transform(r_df)
                            preds.append(model.predict(scaled)[0])
                            probs.append(model.predict_proba(scaled)[0][1])
                    
                    batch_df["Prediction"] = ["Churn" if p == 1 else "No Churn" for p in preds]
                    batch_df["Churn_Probability"] = [f"{p:.2%}" for p in probs]
                    
                    st.success(f"✅ Processed {len(batch_df)} records successfully!")
                    st.dataframe(batch_df, use_container_width=True)
                    
                    # Download button
                    csv = batch_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name="churn_predictions.csv",
                        mime="text/csv"
                    )
                
                except Exception as e:
                    st.error(f"❌ Batch Processing Error: {str(e)}")

# ===== PAGE: RESULTS =====
elif page == "Results":
    st.title("📈 Model Summary & Results")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Accuracy", f"{metrics['accuracy']:.4f}")
    col2.metric("Total Samples", len(y_test))
    col3.metric("Churn Cases (Test)", sum(y_test))
    
    st.markdown("---")
    
    st.subheader("Classification Report")
    report_df = pd.DataFrame(metrics['report']).T.round(4)
    st.dataframe(report_df, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn Distribution (Full Dataset)")
        churn_counts = st.session_state["y"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = ["#55efc4", "#ff7675"]
        wedges, texts, autotexts = ax.pie(
            churn_counts,
            labels=["No Churn", "Churn"],
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
            textprops={"fontsize": 10}
        )
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
        ax.set_title("Customer Churn Distribution", fontsize=12, fontweight="bold")
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        st.subheader("Model Predictions (Test Set)")
        pred_counts = pd.Series(y_pred).value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = ["#55efc4", "#ff7675"]
        wedges, texts, autotexts = ax.pie(
            pred_counts,
            labels=["No Churn", "Churn"],
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
            textprops={"fontsize": 10}
        )
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
        ax.set_title("Model Predictions Distribution", fontsize=12, fontweight="bold")
        st.pyplot(fig, use_container_width=True)