import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
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
    page_title="Churn | Retention Dashboard",
    layout="wide",
    page_icon="",
    initial_sidebar_state="collapsed"
)

# ===== CUSTOM CSS (UX ENHANCEMENTS) =====
st.markdown("""
    <style>
    /* Global Font & Background */
    .main { background-color: #0f1116; font-family: 'Segoe UI', sans-serif; }
    
    /* Headers */
    h1, h2, h3 { color: #ffffff !important; font-weight: 600; }
    
    /* Custom Cards */
    .stat-card {
        background-color: #1c1f26;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #4facfe;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-bottom: 20px;
    }
    .stat-card h4 { color: #a0a0a0; margin: 0; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }
    .stat-card h2 { color: #ffffff; margin: 5px 0 0 0; font-size: 28px; }
    
    /* Warning Card Variant */
    .stat-card-risk {
        background-color: #1c1f26;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #ff6b6b;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1c1f26;
        border-radius: 8px;
        color: white;
        padding: 10px 25px;
        border: 1px solid #333;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4facfe !important;
        color: white !important;
        font-weight: bold;
        border: none;
    }
    
    /* Hide Default Elements */
    [data-testid="stSidebarCollapsedControl"] { display: none; }
    header { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

# ===== LOGIC =====
@st.cache_resource
def build_and_train_pipeline(df: pd.DataFrame):
    data = df.copy()
    data.columns = data.columns.str.strip().str.lower()
    for col in ["totalcharges", "monthlycharges", "tenure"]:
        if col in data.columns: data[col] = pd.to_numeric(data[col], errors="coerce")
    if "churn" not in data.columns: return None, None, None, None, None, None, None
    data['churn'] = data['churn'].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0)
    X = data.drop(["customerid", "churn"], axis=1, errors="ignore")
    y = data["churn"]
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
    preprocessor = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)], verbose_feature_names_out=False)
    
    model_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, k_neighbors=5)),
        ('classifier', RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1))
    ])
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "conf_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True)
    }
    try:
        model = model_pipeline.named_steps['classifier']
        importances = model.feature_importances_
        feat_names = num_cols + cat_cols
        min_len = min(len(feat_names), len(importances))
        feat_imp_df = pd.DataFrame({"Feature": feat_names[:min_len], "Importance": importances[:min_len]})
    except:
        feat_imp_df = pd.DataFrame()
    return model_pipeline, metrics, feat_imp_df, X_test, y_test, num_cols, cat_cols

# ===== DATA PREP =====
data_size = 500
df = pd.DataFrame({
    'gender': np.random.choice(['Male', 'Female'], data_size),
    'seniorcitizen': np.random.choice([0, 1], data_size),
    'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], data_size, p=[0.5, 0.3, 0.2]),
    'monthlycharges': np.random.uniform(30, 110, data_size),
    'tenure': np.random.randint(1, 72, data_size),
    'churn': np.random.choice(['Yes', 'No'], data_size, p=[0.3, 0.7])
})
df.loc[(df['monthlycharges'] > 90) & (df['tenure'] < 12), 'churn'] = np.random.choice(['Yes', 'No'], size=len(df[(df['monthlycharges'] > 90) & (df['tenure'] < 12)]), p=[0.7, 0.3])

pipeline, metrics, feat_imp, X_test, y_test, num_cols, cat_cols = build_and_train_pipeline(df)

# ===== UI HEADER (Robot Removed) =====
st.title("Churn")
st.markdown("### Customer Retention Intelligence Dashboard")
st.markdown("---")

# ===== TABS =====
tab1, tab2, tab3 = st.tabs(["📊 Executive Overview", "💡 Business Insights", "⚡ Retention Simulator"])

# --- TAB 1: EXECUTIVE OVERVIEW ---
with tab1:
    # 1. KPI Cards
    churn_rate = (df['churn'].apply(lambda x: 1 if str(x).lower() in ['yes','1'] else 0).mean()) * 100
    retention_rate = 100 - churn_rate
    at_risk_count = int(len(df) * (churn_rate/100))
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: 
        st.markdown(f"""<div class='stat-card'><h4>Active Customers</h4><h2>{len(df):,}</h2></div>""", unsafe_allow_html=True)
    with c2: 
        st.markdown(f"""<div class='stat-card'><h4>Retention Rate</h4><h2>{retention_rate:.1f}%</h2></div>""", unsafe_allow_html=True)
    with c3: 
        st.markdown(f"""<div class='stat-card-risk'><h4>Customers at Risk</h4><h2>{at_risk_count}</h2></div>""", unsafe_allow_html=True)
    with c4: 
        st.markdown(f"""<div class='stat-card'><h4>Model Reliability</h4><h2>{metrics['accuracy']*100:.0f}%</h2></div>""", unsafe_allow_html=True)

    # 2. Business Questions Charts
    st.subheader("Where are we losing customers?")
    
    c_chart1, c_chart2 = st.columns(2)
    
    with c_chart1:
        st.markdown("**1. By Contract Type**")
        contract_churn = df[df['churn']=='Yes']['contract'].value_counts().reset_index()
        contract_churn.columns = ['Contract Type', 'Churn Count']
        
        fig_bar = px.bar(contract_churn, x='Contract Type', y='Churn Count', color='Churn Count',
                         color_continuous_scale=['#ff9a9e', '#ff6b6b'], text='Churn Count')
        fig_bar.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption("Insight: Month-to-month contracts usually have the highest turnover.")

    with c_chart2:
        st.markdown("**2. Average Monthly Spend**")
        
        # 1. Aggregate Data: Calculate Average Spend for Yes vs No
        avg_spend = df.groupby('churn')['monthlycharges'].mean().reset_index()
        
        # Ensure labels are clean strings
        avg_spend['churn'] = avg_spend['churn'].apply(lambda x: 'Yes' if str(x).lower() in ['yes','1'] else 'No')
        
        # 2. Create Bar Chart
        fig_bar_avg = px.bar(avg_spend, x='churn', y='monthlycharges', color='churn',
                             color_discrete_map={'Yes': '#ff6b6b', 'No': '#4facfe'},
                             text_auto='.0f', # Automatically show value on bars (rounded)
                             labels={'monthlycharges': 'Avg Bill ($)', 'churn': 'Churn Status'})
        
        # 3. Style Updates
        fig_bar_avg.update_traces(textfont_size=14, textposition='outside', cliponaxis=False)
        fig_bar_avg.update_layout(template="plotly_dark", 
                                  plot_bgcolor='rgba(0,0,0,0)', 
                                  paper_bgcolor='rgba(0,0,0,0)',
                                  showlegend=False,
                                  yaxis=dict(showgrid=False)) # Hide grid for cleaner look
        
        st.plotly_chart(fig_bar_avg, use_container_width=True)
        st.caption("Insight: Are departing customers paying significantly more?")

# --- TAB 2: BUSINESS INSIGHTS ---
with tab2:
    col_drivers, col_explain = st.columns([2, 1])
    
    with col_drivers:
        st.subheader("What drives customer churn?")
        st.markdown("These are the top factors influencing customer decisions, ranked by impact.")
        
        if not feat_imp.empty:
            plot_imp = feat_imp.copy()
            plot_imp['Feature'] = plot_imp['Feature'].str.title()
            
            fig_bar = px.bar(plot_imp.sort_values("Importance", ascending=True).tail(8), 
                             x="Importance", y="Feature", orientation='h', 
                             color="Importance", color_continuous_scale="Blues")
            fig_bar.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("Data insufficient for feature ranking.")

# --- TAB 3: SIMULATOR ---
with tab3:
    st.markdown("### ⚡ Customer Risk Calculator")
    st.markdown("Adjust the profile below to see if a customer is likely to leave.")
    
    with st.container():
        st.markdown("""<div style="background-color:#1c1f26; padding:20px; border-radius:15px; border:1px solid #333;">""", unsafe_allow_html=True)
        
        col_input, col_result = st.columns([1, 1])
        
        with col_input:
            st.markdown("#### 👤 Customer Profile")
            tenure = st.slider("Tenure (Months)", 0, 72, 6, help="How long have they been a customer?")
            monthly = st.slider("Monthly Charges ($)", 20.0, 120.0, 85.0)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            
            # Hidden inputs (defaults) to keep UI clean
            others = {}
            if 'seniorcitizen' in num_cols: others['seniorcitizen'] = 0
            if 'gender' in cat_cols: others['gender'] = 'Male'
            
        with col_result:
            st.markdown("#### 🎯 Prediction")
            
            # Real-time calculation
            input_data = pd.DataFrame(columns=num_cols + cat_cols)
            input_data.loc[0, 'tenure'] = tenure
            input_data.loc[0, 'monthlycharges'] = monthly
            input_data.loc[0, 'totalcharges'] = tenure * monthly
            input_data.loc[0, 'contract'] = contract
            for k, v in others.items(): input_data.loc[0, k] = v
                
            prob = pipeline.predict_proba(input_data)[0][1]
            risk_pct = prob * 100
            
            # Dynamic UI based on risk
            if risk_pct < 40:
                risk_color = "#51cf66" # Green
                risk_label = "SAFE"
                action = "✅ No immediate action needed. Keep nurturing."
            elif risk_pct < 70:
                risk_color = "#fcc419" # Yellow
                risk_label = "AT RISK"
                action = "⚠️ Consider offering a small discount or check-in call."
            else:
                risk_color = "#ff6b6b" # Red
                risk_label = "HIGH CHURN RISK"
                action = "🚨 **URGENT:** Offer 12-month contract upgrade or 20% loyalty discount."

            st.markdown(f"""
                <div style="text-align:center; padding:20px;">
                    <h1 style="font-size:4em; color:{risk_color}; margin:0;">{risk_pct:.0f}%</h1>
                    <h3 style="color:{risk_color}; margin-top:0;">CHURN PROBABILITY</h3>
                    <div style="background-color:#333; height:2px; margin:20px 0;"></div>
                    <p style="font-size:1.1em; color:white;">{action}</p>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)