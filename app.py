import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="ChurnGuard | AI Retention",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM CSS (ENTERPRISE DASHBOARD) =====
st.markdown("""
    <style>
    /* Main Background */
    .main { background-color: #0E1117; }
    
    /* Card Styling */
    .metric-card {
        background-color: #1E1E1E;
        border: 1px solid #333333;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
    }
    .metric-card h3 { margin: 0; font-size: 1em; color: #aaaaaa; }
    .metric-card h2 { margin: 10px 0 0 0; font-size: 2em; color: white; font-weight: bold; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E1E1E;
        border-radius: 5px;
        color: white;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4facfe !important;
        color: white !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ===== BACKEND LOGIC (CACHED) =====

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

@st.cache_data
def clean_and_prep_data(df: pd.DataFrame):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    
    # Numeric cleanup
    for col in ["totalcharges", "monthlycharges", "tenure"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)
            
    # Target cleanup
    if "churn" in df.columns:
        y = df["churn"].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)
        X = df.drop(["customerid", "churn"], axis=1, errors="ignore")
    else:
        # For prediction only mode
        y = None
        X = df.drop(["customerid"], axis=1, errors="ignore")

    # Encoder setup
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    
    if cat_cols:
        X[cat_cols] = enc.fit_transform(X[cat_cols].fillna("missing"))
    
    X = X.fillna(X.median(numeric_only=True))
    
    return X, y, enc, cat_cols

@st.cache_resource
def train_model(X, y):
    # SMOTE
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_res, y_res = smote.fit_resample(X, y)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, stratify=y_res, random_state=42
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    model = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Metrics
    y_pred = model.predict(X_test_scaled)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "conf_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True)
    }
    
    return model, scaler, metrics, X.columns.tolist(), y_test, y_pred

# ===== SIDEBAR: DATA LOADER =====
with st.sidebar:
    st.title("🛡️ ChurnGuard")
    st.caption("AI-Powered Retention Analytics")
    st.divider()
    
    st.subheader("📂 Data Source")
    uploaded_file = st.file_uploader("Upload Telecom Data (CSV)", type=["csv"])
    
    # Fallback to demo data generator if no file
    if not uploaded_file:
        st.info("ℹ️ No file uploaded. Generating synthetic demo data for visualization.")
        # Create dummy data
        data_size = 500
        demo_data = {
            'gender': np.random.choice(['Male', 'Female'], data_size),
            'seniorcitizen': np.random.choice([0, 1], data_size),
            'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], data_size),
            'paymentmethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer'], data_size),
            'paperlessbilling': np.random.choice(['Yes', 'No'], data_size),
            'monthlycharges': np.random.uniform(20, 120, data_size),
            'tenure': np.random.randint(1, 72, data_size),
            'totalcharges': np.random.uniform(100, 8000, data_size),
            'churn': np.random.choice(['Yes', 'No'], data_size, p=[0.26, 0.74])
        }
        df = pd.DataFrame(demo_data)
    else:
        df = load_data(uploaded_file)
        st.success("Dataset Loaded Successfully")

# ===== PROCESSING =====
X, y, encoder, cat_cols = clean_and_prep_data(df)
model, scaler, model_metrics, train_cols, y_test_real, y_pred_real = train_model(X, y)

# ===== TABS UI =====
tab1, tab2, tab3 = st.tabs(["📊 Executive Overview", "🧠 Model Intelligence", "🔮 Risk Simulator"])

# ----- TAB 1: OVERVIEW -----
with tab1:
    # Key Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    total_customers = len(df)
    churn_rate = y.mean() * 100
    avg_rev = df['monthlycharges'].mean()
    risky_customers = df[df['contract'] == 'Month-to-month'].shape[0]

    with c1: st.markdown(f"<div class='metric-card'><h3>Total Customers</h3><h2>{total_customers:,}</h2></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='metric-card'><h3>Churn Rate</h3><h2 style='color: {'#ff6b6b' if churn_rate > 20 else '#51cf66'}'>{churn_rate:.1f}%</h2></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='metric-card'><h3>Avg Monthly Revenue</h3><h2>${avg_rev:.2f}</h2></div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='metric-card'><h3>High Risk (Month-to-Month)</h3><h2>{risky_customers:,}</h2></div>", unsafe_allow_html=True)

    st.write("")
    st.write("")
    
    # Charts Row
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("### 📉 Churn Distribution")
        # Interactive Pie Chart
        fig_pie = px.pie(names=['Retained', 'Churned'], values=[len(y)-sum(y), sum(y)], 
                         color_discrete_sequence=['#4facfe', '#ff6b6b'], hole=0.4)
        fig_pie.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_chart2:
        st.markdown("### 💰 Charges vs Tenure")
        # Interactive Scatter
        fig_scatter = px.scatter(df, x='tenure', y='monthlycharges', color='churn',
                                 color_discrete_map={'Yes': '#ff6b6b', 'No': '#4facfe'},
                                 opacity=0.6)
        fig_scatter.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_scatter, use_container_width=True)

# ----- TAB 2: INTELLIGENCE -----
with tab2:
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("### 🧩 Feature Importance")
        # Feature Importance Chart
        feat_imp = pd.DataFrame({
            "Feature": train_cols,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True).tail(10)
        
        fig_bar = px.bar(feat_imp, x="Importance", y="Feature", orientation='h',
                         color="Importance", color_continuous_scale="Viridis")
        fig_bar.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with c2:
        st.markdown("### 🎯 Model Accuracy")
        st.markdown(f"""
        <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px; text-align: center;">
            <h1 style="font-size: 4em; color: #4facfe; margin: 0;">{model_metrics['accuracy']*100:.1f}%</h1>
            <p>Accuracy on Test Data</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.markdown("#### Confusion Matrix")
        cm = model_metrics['conf_matrix']
        fig_hm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=['Retained', 'Churned'], y=['Retained', 'Churned'])
        fig_hm.update_layout(template="plotly_dark", width=300, height=300)
        fig_hm.update_xaxes(side="top")
        st.plotly_chart(fig_hm, use_container_width=True)

# ----- TAB 3: SIMULATOR -----
with tab3:
    st.markdown("### 🔮 Individual Customer Predictor")
    
    col_input, col_pred = st.columns([1, 1])
    
    with col_input:
        with st.form("sim_form"):
            st.markdown("#### Customer Profile")
            c1, c2 = st.columns(2)
            tenure = c1.slider("Tenure (Months)", 0, 72, 12)
            monthly = c2.slider("Monthly Charges ($)", 18.0, 120.0, 50.0)
            
            c3, c4 = st.columns(2)
            contract = c3.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            tech_support = c4.selectbox("Tech Support", ["No", "Yes", "No internet service"]) # Adding dummy logic for UI
            
            # Hidden processing for display
            total = tenure * monthly
            
            submitted = st.form_submit_button("Run Risk Analysis", use_container_width=True)
    
    with col_pred:
        if submitted:
            # Prepare single row input
            # Note: In a real scenario, we need to match ALL columns trained. 
            # Here we map the form inputs to a DataFrame and fill the rest with mode/median
            input_dict = {
                'tenure': tenure,
                'monthlycharges': monthly,
                'totalcharges': total,
                'contract': contract,
                # Add default values for other cols required by model
                'gender': 'Male', 'seniorcitizen': 0, 'partner': 'No', 'dependents': 'No',
                'phoneservice': 'Yes', 'multiplelines': 'No', 'internetservice': 'Fiber optic',
                'onlinesecurity': 'No', 'onlinebackup': 'No', 'deviceprotection': 'No',
                'techsupport': 'No', 'streamingtv': 'No', 'streamingmovies': 'No',
                'paperlessbilling': 'Yes', 'paymentmethod': 'Electronic check'
            }
            
            input_df = pd.DataFrame([input_dict])
            
            # Encoder & Scale
            # Ensure columns match training data
            for col in cat_cols:
                if col not in input_df.columns: input_df[col] = "missing"
            input_df[cat_cols] = encoder.transform(input_df[cat_cols])
            
            # Align columns
            final_input = pd.DataFrame(columns=train_cols)
            for col in train_cols:
                if col in input_df.columns:
                    final_input.loc[0, col] = input_df.iloc[0][col]
                else:
                    final_input.loc[0, col] = 0 # Fill missing numeric with 0 for demo
            
            final_input = final_input.fillna(0)
            scaled_input = scaler.transform(final_input)
            
            # Predict
            prob = model.predict_proba(scaled_input)[0][1]
            
            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                title = {'text': "Churn Probability", 'font': {'size': 24, 'color': 'white'}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#1E1E1E"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 40], 'color': "#51cf66"}, # Safe
                        {'range': [40, 70], 'color': "#fcc419"}, # Warning
                        {'range': [70, 100], 'color': "#ff6b6b"}], # Danger
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': prob * 100}}))
            
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            if prob > 0.5:
                st.error("⚠️ Customer is at High Risk of Churning")
            else:
                st.success("✅ Customer is likely to Stay")
        else:
            # Placeholder State
            st.info("👈 Enter customer details to calculate risk profile.")
            st.markdown("""
            <div style="text-align: center; color: gray; padding: 50px;">
                <h1>🛡️</h1>
                <p>Awaiting Input...</p>
            </div>
            """, unsafe_allow_html=True)