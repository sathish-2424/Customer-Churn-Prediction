import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

st.set_page_config(page_title="Telecom Customer Churn Prediction", layout="wide")
st.title("Telecom Customer Churn Prediction App")

# -------------------- CSV Upload --------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()
    st.dataframe(df.head())

    # -------------------- Preprocessing --------------------
    for col in ['totalcharges', 'monthlycharges', 'tenure']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)

    if 'churn' not in df.columns:
        st.error("Target column 'churn' missing!")
        st.stop()

    y = df['churn'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    X = df.drop(['customerid', 'churn'], axis=1, errors='ignore')

    # Identify categorical columns
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X[cat_cols] = oe.fit_transform(X[cat_cols])

    # Fill numeric NaNs
    X = X.fillna(X.median(numeric_only=True))

    # Balance dataset using SMOTE
    X_bal, y_bal = SMOTE(random_state=42).fit_resample(X, y)

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------- Train Models --------------------
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred)
        }
        trained_models[name] = model

    # Save in session state
    st.session_state.trained_models = trained_models
    st.session_state.scaler = scaler
    st.session_state.encoder = oe
    st.session_state.cat_cols = cat_cols
    st.session_state.X_columns = X.columns

    # -------------------- Display Model Performance --------------------
    st.subheader("Model Performance")
    st.dataframe(pd.DataFrame(results).T.round(4))

    # -------------------- Predict Individual Customer --------------------
    st.markdown('<h2 class="section-header">Predict Individual Customer Churn</h2>', unsafe_allow_html=True)

    if 'trained_models' not in st.session_state or not st.session_state.trained_models:
        st.markdown(
            '<div class="warning-box"><strong>Please train models first</strong> to enable predictions.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown("Enter customer details below to predict churn risk:")

        # --- Helper function for encoding ---
        def encode_input(df_input, cat_cols, encoder, train_cols):
            df_encoded = df_input.copy()

            # Ensure all categorical columns exist
            for col in cat_cols:
                if col not in df_encoded.columns:
                    df_encoded[col] = 'missing'  # placeholder for missing categorical

            if cat_cols:
                df_encoded[cat_cols] = encoder.transform(df_encoded[cat_cols])

            # Ensure all training columns exist
            for col in train_cols:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0  # default numeric value

            return df_encoded[train_cols]

        # --- Prediction Form ---
        with st.form("churn_prediction_form"):
            col1, col2, col3 = st.columns(3)

            # Demographics
            with col1:
                st.subheader("Demographics")
                gender_col = 'gender' if 'gender' in df.columns else 'Gender'
                gender = st.selectbox("Gender", df[gender_col].unique(), key="gender")
                senior_citizen = st.selectbox("Senior Citizen", df['seniorcitizen'].unique(), key="senior")
                partner = st.selectbox("Partner", df['partner'].unique(), key="partner")
                dependents = st.selectbox("Dependents", df['dependents'].unique(), key="dependents")

            # Services
            with col2:
                st.subheader("Services")
                phone_service = st.selectbox("Phone Service", df['phoneservice'].unique(), key="phone")
                internet_service = st.selectbox("Internet Service", df['internetservice'].unique(), key="internet")
                online_security = st.selectbox("Online Security", df['onlinesecurity'].unique(), key="security")
                online_backup = st.selectbox("Online Backup", df['onlinebackup'].unique(), key="backup")
                tech_support = st.selectbox("Tech Support", df['techsupport'].unique(), key="tech")

            # Contract & Billing
            with col3:
                st.subheader("Contract & Billing")
                contract = st.selectbox("Contract", df['contract'].unique(), key="contract")
                paperless_billing = st.selectbox("Paperless Billing", df['paperlessbilling'].unique(), key="paperless")
                payment_method = st.selectbox("Payment Method", df['paymentmethod'].unique(), key="payment")

            # Numeric inputs
            col1, col2, col3 = st.columns(3)
            with col1:
                monthly_charges = st.number_input(
                    "Monthly Charges ($)", min_value=0.0, max_value=float(df['monthlycharges'].max()),
                    value=float(df['monthlycharges'].mean()), step=0.01)
            with col2:
                total_charges = st.number_input(
                    "Total Charges ($)", min_value=0.0, max_value=float(df['totalcharges'].max()),
                    value=float(df['totalcharges'].mean()), step=0.01)
            with col3:
                tenure = st.number_input(
                    "Tenure (Months)", min_value=0, max_value=int(df['tenure'].max()),
                    value=int(df['tenure'].mean()), step=1)

            predict_button = st.form_submit_button("Predict Churn Risk", use_container_width=True)

        if predict_button:
            input_data = {
                gender_col: gender,
                'seniorcitizen': senior_citizen,
                'partner': partner,
                'dependents': dependents,
                'tenure': tenure,
                'phoneservice': phone_service,
                'internetservice': internet_service,
                'onlinesecurity': online_security,
                'onlinebackup': online_backup,
                'techsupport': tech_support,
                'monthlycharges': monthly_charges,
                'contract': contract,
                'paperlessbilling': paperless_billing,
                'paymentmethod': payment_method,
                'totalcharges': total_charges
            }
            input_df = pd.DataFrame([input_data])

            # Encode & Scale
            input_encoded = encode_input(input_df, st.session_state.cat_cols, st.session_state.encoder, st.session_state.X_columns)
            input_scaled = st.session_state.scaler.transform(input_encoded)

            # Predict using Random Forest
            best_model = st.session_state.trained_models["Random Forest"]
            prediction = best_model.predict(input_scaled)[0]
            probability = best_model.predict_proba(input_scaled)[0]

            # Display result
            st.markdown("---")
            st.markdown('<h3 style="text-align: center;">Prediction Result</h3>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                if prediction == 1:
                    st.markdown(f'<div class="churn-risk-high"><h2>⚠️ HIGH CHURN RISK</h2><p style="font-size:1.2rem;">Probability: <strong>{probability[1]*100:.1f}%</strong></p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="churn-risk-low"><h2>✅ LOW CHURN RISK</h2><p style="font-size:1.2rem;">Probability: <strong>{probability[0]*100:.1f}%</strong></p></div>', unsafe_allow_html=True)
            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                labels = ['No Churn', 'Churn']
                sizes = [probability[0]*100, probability[1]*100]
                colors = ['#2ecc71', '#e74c3c']
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
                ax.set_title('Churn Probability Distribution', fontweight='bold')
                st.pyplot(fig, use_container_width=True)
