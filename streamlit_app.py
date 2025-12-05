import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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

# ===== CUSTOM CSS =====
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
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
    </style>
""", unsafe_allow_html=True)

# ===== BACKEND LOGIC =====

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

@st.cache_resource
def build_and_train_pipeline(df: pd.DataFrame):
    # 1. Cleaning & Prep
    data = df.copy()
    data.columns = data.columns.str.strip().str.lower()
    
    # Handle numeric coercion
    for col in ["totalcharges", "monthlycharges", "tenure"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
            
    # Define Target
    if "churn" not in data.columns:
        return None, None, None, None

    # Binary Target Encoding
    data['churn'] = data['churn'].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0)
    
    X = data.drop(["customerid", "churn"], axis=1, errors="ignore")
    y = data["churn"]
    
    # Identify Column Types
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 2. Split Data (BEFORE Processing to prevent Leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 3. Define Transformers
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ], verbose_feature_names_out=False)
    
    # 4. Main Pipeline with SMOTE inside (applied only to training folds)
    model_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, k_neighbors=5)),
        ('classifier', RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1))
    ])
    
    # 5. Train
    model_pipeline.fit(X_train, y_train)
    
    # 6. Evaluate
    y_pred = model_pipeline.predict(X_test)
    probs = model_pipeline.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "conf_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True)
    }
    
    # Feature Importance Extraction
    try:
        model = model_pipeline.named_steps['classifier']
        # Get feature names after transformation is tricky with pipelines, 
        # so we reconstruct a close approximation for the chart
        importances = model.feature_importances_
        # Note: This is a simplification. For exact feature names one needs to dig into the transformer.
        # We will use indices for robustness in this demo script.
        feat_names = num_cols + cat_cols
        feat_imp_df = pd.DataFrame({"Feature": feat_names[:len(importances)], "Importance": importances})
    except:
        feat_imp_df = pd.DataFrame()

    return model_pipeline, metrics, feat_imp_df, X_test, y_test, num_cols, cat_cols

# ===== SIDEBAR =====
with st.sidebar:
    st.title("🛡️ ChurnGuard")
    st.divider()
    
    uploaded_file = st.file_uploader("Upload Data (CSV)", type=["csv"])
    
    if not uploaded_file:
        st.info("ℹ️ Using synthetic demo data.")
        # Robust synthetic data generation
        data_size = 500
        df = pd.DataFrame({
            'gender': np.random.choice(['Male', 'Female'], data_size),
            'seniorcitizen': np.random.choice([0, 1], data_size),
            'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], data_size),
            'monthlycharges': np.random.uniform(20, 120, data_size),
            'tenure': np.random.randint(1, 72, data_size),
            'totalcharges': np.random.uniform(100, 8000, data_size),
            'churn': np.random.choice(['Yes', 'No'], data_size, p=[0.3, 0.7]) # 30% churn rate
        })
    else:
        df = load_data(uploaded_file)
        st.success("Dataset Loaded")

# ===== PROCESSING =====
pipeline, metrics, feat_imp, X_test, y_test, num_cols, cat_cols = build_and_train_pipeline(df)

if pipeline is None:
    st.error("Dataset must contain a 'churn' column.")
    st.stop()

# ===== TABS UI =====
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🧠 Intelligence", "🔮 Simulator"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    churn_rate = (df['churn'].apply(lambda x: 1 if str(x).lower() in ['yes','1'] else 0).mean()) * 100
    
    with c1: st.markdown(f"<div class='metric-card'><h3>Customers</h3><h2>{len(df):,}</h2></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='metric-card'><h3>Churn Rate</h3><h2 style='color: {'#ff6b6b' if churn_rate > 20 else '#51cf66'}'>{churn_rate:.1f}%</h2></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='metric-card'><h3>Avg Revenue</h3><h2>${df['monthlycharges'].mean():.2f}</h2></div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='metric-card'><h3>Accuracy</h3><h2>{metrics['accuracy']*100:.1f}%</h2></div>", unsafe_allow_html=True)

    c_chart1, c_chart2 = st.columns(2)
    with c_chart1:
        st.markdown("### 📉 Churn Composition")
        fig_pie = px.pie(names=['Retained', 'Churned'], values=[len(df)-sum(df['churn']=='Yes'), sum(df['churn']=='Yes')], 
                         color_discrete_sequence=['#4facfe', '#ff6b6b'], hole=0.4)
        fig_pie.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with c_chart2:
        st.markdown("### 💰 Risk Profile")
        fig_scatter = px.scatter(df, x='tenure', y='monthlycharges', color='churn',
                                 color_discrete_map={'Yes': '#ff6b6b', 'No': '#4facfe'}, opacity=0.6)
        fig_scatter.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    col_imp, col_cm = st.columns([2, 1])
    
    with col_imp:
        st.markdown("### 🧩 Key Drivers of Churn")
        if not feat_imp.empty:
            fig_bar = px.bar(feat_imp.sort_values("Importance", ascending=True).tail(10), 
                             x="Importance", y="Feature", orientation='h', color="Importance", color_continuous_scale="Viridis")
            fig_bar.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("Feature importance not available for this pipeline configuration.")
            
    with col_cm:
        st.markdown("### Model Evaluation")
        cm = metrics['conf_matrix']
        # Confusion Matrix Heatmap
        fig_hm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=['Retained', 'Churned'], y=['Retained', 'Churned'])
        fig_hm.update_layout(template="plotly_dark", width=300, height=300)
        st.plotly_chart(fig_hm, use_container_width=True)
        
        st.caption("Confusion Matrix Explanation:")
        st.caption("Top-Left: Correctly Retained | Bottom-Right: Correctly Predicted Churn")
        
        # We can add a conceptual diagram for confusion matrix here if the user needs education
        # 

with tab3:
    st.markdown("### 🔮 Risk Simulator")
    
    with st.form("sim_form"):
        c1, c2, c3 = st.columns(3)
        tenure = c1.slider("Tenure (Months)", 0, 72, 12)
        monthly = c2.slider("Monthly Charges ($)", 18.0, 120.0, 50.0)
        contract = c3.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        
        # Capture other features if they exist in training data
        others = {}
        if 'gender' in cat_cols: others['gender'] = c1.selectbox("Gender", ["Male", "Female"])
        if 'seniorcitizen' in num_cols: others['seniorcitizen'] = c2.selectbox("Senior Citizen", [0, 1])
        
        submitted = st.form_submit_button("Calculate Risk", type="primary")

    if submitted:
        # Construct input dataframe matching training columns
        input_data = pd.DataFrame(columns=num_cols + cat_cols)
        input_data.loc[0, 'tenure'] = tenure
        input_data.loc[0, 'monthlycharges'] = monthly
        input_data.loc[0, 'totalcharges'] = tenure * monthly
        input_data.loc[0, 'contract'] = contract
        
        for k, v in others.items():
            input_data.loc[0, k] = v
            
        # Pipeline handles imputation/encoding automatically
        prob = pipeline.predict_proba(input_data)[0][1]
        
        col_res, col_gauge = st.columns([1, 1])
        
        with col_res:
            st.markdown(f"""
            <div style="padding:20px; border-radius:10px; border:1px solid #333; background:#1E1E1E; margin-top: 20px;">
                <h3 style="color:gray; margin:0;">Churn Probability</h3>
                <h1 style="font-size:3em; margin:0; color: {'#ff6b6b' if prob > 0.5 else '#51cf66'}">{prob*100:.1f}%</h1>
                <p style="color:white; margin-top:10px;">
                    {'⚠️ High Risk Customer' if prob > 0.5 else '✅ Safe Customer'}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = prob * 100,
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#1E1E1E"},
                    'steps': [
                        {'range': [0, 40], 'color': "#51cf66"},
                        {'range': [40, 70], 'color': "#fcc419"},
                        {'range': [70, 100], 'color': "#ff6b6b"}],
                    'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': prob * 100}
                }
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=250, margin=dict(l=20,r=20,t=0,b=0))
            st.plotly_chart(fig_gauge, use_container_width=True)