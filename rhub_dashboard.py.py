import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, silhouette_score, r2_score, 
                            mean_squared_error, confusion_matrix, roc_curve, auc, 
                            precision_recall_curve, accuracy_score, precision_score, 
                            recall_score, f1_score)
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from scipy import stats
from datetime import datetime, timedelta
import warnings
import json
import io
import base64
from io import BytesIO

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="ReFill Hub: Ultimate Analytics Platform",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    /* Main app background with gradient */
    .main {
        background: linear-gradient(135deg, #F0F2F6 0%, #E8EBF0 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8F9FA 100%);
        border-right: 2px solid #2E8B57;
    }
    
    /* Metric cards with enhanced styling */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
        border: 2px solid #E6E9EF;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 16px rgba(46, 139, 87, 0.1);
        transition: all 0.3s ease-in-out;
    }
    [data-testid="stMetric"]:hover {
        box-shadow: 0 12px 24px rgba(46, 139, 87, 0.2);
        transform: translateY(-5px);
        border-color: #2E8B57;
    }
    
    /* Headers with theme color */
    h1 {
        color: #2E8B57;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        font-family: 'Helvetica Neue', sans-serif;
    }
    h2 {
        color: #228B22;
        font-weight: 600;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h3 {
        color: #3CB371;
        font-weight: 500;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Enhanced containers */
    .st-emotion-cache-18ni7ap {
        background-color: #FFFFFF;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.05);
    }
    
    /* Custom button styling */
    .stButton>button {
        background: linear-gradient(135deg, #2E8B57 0%, #228B22 100%);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #228B22 0%, #2E8B57 100%);
        box-shadow: 0 6px 12px rgba(46, 139, 87, 0.3);
        transform: translateY(-2px);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-left: 5px solid #2E8B57;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    /* Warning boxes */
    .warning-box {
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
        border-left: 5px solid #FF9800;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    /* Success boxes */
    .success-box {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-left: 5px solid #4CAF50;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    /* Metric card custom */
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        box-shadow: 0 8px 20px rgba(46, 139, 87, 0.15);
        transform: translateY(-3px);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F0F2F6;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2E8B57;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_download_link(df, filename, file_label):
    """Generate a download link for a dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-btn">📥 Download {file_label}</a>'
    return href

def create_excel_download(df, filename):
    """Generate Excel download"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    output.seek(0)
    return output

def generate_synthetic_data(n_samples=600):
    """Generate synthetic survey data for ReFill Hub"""
    np.random.seed(42)
    
    data = {
        'Age_Group': np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+'], n_samples, p=[0.15, 0.35, 0.25, 0.15, 0.10]),
        'Gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.48, 0.50, 0.02]),
        'Emirate': np.random.choice(['Dubai', 'Abu Dhabi', 'Sharjah', 'Ajman', 'Ras Al Khaimah', 'Fujairah', 'Umm Al Quwain'], 
                                   n_samples, p=[0.40, 0.30, 0.15, 0.05, 0.04, 0.03, 0.03]),
        'Occupation': np.random.choice(['Private Sector', 'Government', 'Self-Employed', 'Student', 'Homemaker', 'Retired'], 
                                      n_samples, p=[0.45, 0.20, 0.15, 0.10, 0.07, 0.03]),
        'Income': np.random.choice(['Below 5000', '5000-10000', '10000-15000', '15000-20000', 'Above 20000'], 
                                  n_samples, p=[0.10, 0.25, 0.30, 0.20, 0.15]),
        'Family_Size': np.random.choice(['1-2', '3-4', '5+'], n_samples, p=[0.30, 0.50, 0.20]),
        'Purchase_Frequency': np.random.choice(['Daily', 'Weekly', 'Bi-weekly', 'Monthly'], n_samples, p=[0.10, 0.50, 0.25, 0.15]),
        'Purchase_Location': np.random.choice(['Supermarket', 'Hypermarket', 'Online', 'Local Store', 'Convenience Store'], 
                                             n_samples, p=[0.35, 0.30, 0.20, 0.10, 0.05]),
        'Uses_Eco_Products': np.random.choice(['Yes', 'No', 'Sometimes'], n_samples, p=[0.30, 0.20, 0.50]),
        'Preferred_Packaging': np.random.choice(['Plastic', 'Glass', 'Metal', 'Biodegradable', 'No Preference'], 
                                               n_samples, p=[0.25, 0.15, 0.10, 0.35, 0.15]),
        'Aware_Plastic_Ban': np.random.choice(['Yes', 'No'], n_samples, p=[0.75, 0.25]),
        'Follow_Campaigns': np.random.choice(['Yes', 'No'], n_samples, p=[0.40, 0.60]),
        'Used_Refill_Before': np.random.choice(['Yes', 'No'], n_samples, p=[0.25, 0.75]),
        'Preferred_Payment_Mode': np.random.choice(['Cash', 'Card', 'Digital Wallet', 'Mobile Banking'], 
                                                   n_samples, p=[0.10, 0.35, 0.40, 0.15]),
        'Refill_Location': np.random.choice(['Supermarket', 'Mall', 'Residential Area', 'Petrol Station', 'Standalone Kiosk'], 
                                           n_samples, p=[0.35, 0.25, 0.20, 0.10, 0.10]),
        'Container_Type': np.random.choice(['Own Container', 'Provided Container', 'Either'], n_samples, p=[0.30, 0.25, 0.45]),
        'Interest_Non_Liquids': np.random.choice(['Yes', 'No', 'Maybe'], n_samples, p=[0.40, 0.30, 0.30]),
        'Discount_Switch': np.random.choice(['5%', '10%', '15%', '20%', '25%+'], n_samples, p=[0.10, 0.20, 0.30, 0.25, 0.15]),
        'Importance_Convenience': np.random.randint(1, 6, n_samples),
        'Importance_Price': np.random.randint(1, 6, n_samples),
        'Importance_Sustainability': np.random.randint(1, 6, n_samples),
        'Reduce_Waste_Score': np.random.randint(1, 6, n_samples),
        'Eco_Brand_Preference': np.random.randint(1, 6, n_samples),
        'Social_Influence_Score': np.random.randint(1, 6, n_samples),
        'Try_Refill_Likelihood': np.random.randint(1, 6, n_samples),
        'Willingness_to_Pay_AED': np.random.uniform(20, 200, n_samples).round(2),
        'Likely_to_Use_ReFillHub': np.random.choice(['Yes', 'No'], n_samples, p=[0.65, 0.35])
    }
    
    # Generate Products_Bought
    products = ['Detergent', 'Fabric Softener', 'Dish Soap', 'Shampoo', 'Conditioner', 
                'Body Wash', 'Hand Soap', 'All-Purpose Cleaner', 'Floor Cleaner']
    data['Products_Bought'] = [', '.join(np.random.choice(products, size=np.random.randint(2, 5), replace=False)) 
                                for _ in range(n_samples)]
    
    return pd.DataFrame(data)

def calculate_clv(avg_spend, frequency_per_year, retention_rate, years=3):
    """Calculate Customer Lifetime Value"""
    clv = 0
    for year in range(1, years + 1):
        clv += (avg_spend * frequency_per_year * (retention_rate ** year))
    return clv

def perform_statistical_tests(df):
    """Perform statistical tests for insights"""
    results = {}
    
    # Chi-square test: Adoption vs Income
    contingency_table = pd.crosstab(df['Income'], df['Likely_to_Use_ReFillHub'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    results['income_adoption_chi2'] = {'chi2': chi2, 'p_value': p_value}
    
    # T-test: Spending difference between adopters and non-adopters
    adopters = df[df['Likely_to_Use_ReFillHub'] == 'Yes']['Willingness_to_Pay_AED']
    non_adopters = df[df['Likely_to_Use_ReFillHub'] == 'No']['Willingness_to_Pay_AED']
    t_stat, p_value = stats.ttest_ind(adopters, non_adopters)
    results['spending_ttest'] = {'t_stat': t_stat, 'p_value': p_value}
    
    return results

def generate_business_recommendations(df, models, metrics):
    """Generate actionable business recommendations"""
    recommendations = []
    
    # Recommendation 1: Target Segment
    cluster_adoption = df.groupby('Cluster')['Likely_to_Use_ReFillHub'].apply(lambda x: (x == 'Yes').mean())
    best_cluster = cluster_adoption.idxmax()
    recommendations.append({
        'title': '🎯 Priority Target Segment',
        'detail': f"Focus marketing on Cluster {best_cluster} with {cluster_adoption[best_cluster]*100:.1f}% adoption rate",
        'priority': 'HIGH'
    })
    
    # Recommendation 2: Location Strategy
    location_adoption = df.groupby('Refill_Location')['Likely_to_Use_ReFillHub'].apply(lambda x: (x == 'Yes').mean())
    best_location = location_adoption.idxmax()
    recommendations.append({
        'title': '📍 Optimal Kiosk Placement',
        'detail': f"Deploy first kiosks in {best_location} (highest adoption: {location_adoption[best_location]*100:.1f}%)",
        'priority': 'HIGH'
    })
    
    # Recommendation 3: Pricing Strategy
    median_wtp = df['Willingness_to_Pay_AED'].median()
    recommendations.append({
        'title': '💰 Pricing Strategy',
        'detail': f"Set initial pricing around AED {median_wtp:.2f} per visit (median willingness to pay)",
        'priority': 'MEDIUM'
    })
    
    # Recommendation 4: Product Bundle
    if 'association_rules' in models and not models['association_rules'].empty:
        top_rule = models['association_rules'].iloc[0]
        recommendations.append({
            'title': '🛒 Product Bundling',
            'detail': f"Create bundle: {top_rule['antecedents']} + {top_rule['consequents']} (Lift: {top_rule['lift']:.2f})",
            'priority': 'MEDIUM'
        })
    
    # Recommendation 5: Best Model
    best_model = max(metrics['classification'], key=lambda x: metrics['classification'][x]['Accuracy'])
    recommendations.append({
        'title': '🤖 Recommended Algorithm',
        'detail': f"Deploy {best_model} for production (Accuracy: {metrics['classification'][best_model]['Accuracy']:.2%})",
        'priority': 'HIGH'
    })
    
    return recommendations

# =============================================================================
# DATA LOADING & PREPROCESSING (CACHED)
# =============================================================================
@st.cache_data
def load_and_clean_data(filepath=None, uploaded_file=None, use_synthetic=False):
    """
    Loads and cleans the dataset from filepath, uploaded file, or generates synthetic data
    """
    if use_synthetic:
        df = generate_synthetic_data()
        st.success("✅ Synthetic data generated successfully!")
    elif uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Custom data uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading uploaded file: {str(e)}")
            return None, None, None, None
    elif filepath:
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            st.warning(f"Default data file not found. Generating synthetic data...")
            df = generate_synthetic_data()
    else:
        df = generate_synthetic_data()

    # Basic Cleaning
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].str.strip()
        
    # Feature Engineering
    def process_family_size(val):
        if '5+' in str(val): return 5
        if '1-2' in str(val): return 1.5
        if '3-4' in str(val): return 3.5
        return 3.0
    df['Family_Size_Num'] = df['Family_Size'].apply(process_family_size)
    
    # Define feature lists
    categorical_features = ['Age_Group', 'Gender', 'Emirate', 'Occupation', 'Income', 
                            'Purchase_Location', 'Purchase_Frequency', 'Uses_Eco_Products',
                            'Preferred_Packaging', 'Aware_Plastic_Ban', 'Eco_Brand_Preference', 
                            'Follow_Campaigns', 'Used_Refill_Before', 'Preferred_Payment_Mode',
                            'Refill_Location', 'Container_Type', 'Interest_Non_Liquids', 'Discount_Switch']

    numerical_features = ['Family_Size_Num', 'Importance_Convenience', 'Importance_Price', 
                          'Importance_Sustainability', 'Reduce_Waste_Score', 'Social_Influence_Score', 
                          'Try_Refill_Likelihood']

    cluster_features = ['Importance_Convenience', 'Importance_Price', 'Importance_Sustainability', 
                        'Reduce_Waste_Score', 'Family_Size_Num']

    return df, categorical_features, numerical_features, cluster_features

# =============================================================================
# COMPREHENSIVE MODEL TRAINING (CACHED)
# =============================================================================
@st.cache_resource
def train_all_models(df, categorical_features, numerical_features, cluster_features):
    """
    Trains ALL models: Multiple classifiers, regressors, clustering, and association rules
    """
    models = {}
    metrics = {'classification': {}, 'regression': {}}
    
    # --- PREPROCESSING PIPELINE ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # --- 1. CLASSIFICATION (Multiple Algorithms) ---
    X_class = df[categorical_features + numerical_features]
    y_class = df['Likely_to_Use_ReFillHub'].map({'Yes': 1, 'No': 0})
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    
    classifiers = {
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200, max_depth=15),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
    }
    
    for name, clf in classifiers.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
        pipeline.fit(X_train_c, y_train_c)
        y_pred = pipeline.predict(X_test_c)
        y_pred_proba = pipeline.predict_proba(X_test_c)[:, 1] if hasattr(pipeline, 'predict_proba') else None
        
        models[name] = pipeline
        
        # Calculate metrics
        metrics['classification'][name] = {
            'Accuracy': accuracy_score(y_test_c, y_pred),
            'Precision': precision_score(y_test_c, y_pred),
            'Recall': recall_score(y_test_c, y_pred),
            'F1 Score': f1_score(y_test_c, y_pred),
            'Confusion Matrix': confusion_matrix(y_test_c, y_pred)
        }
        
        # ROC Curve if available
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_test_c, y_pred_proba)
            metrics['classification'][name]['ROC_AUC'] = auc(fpr, tpr)
            metrics['classification'][name]['ROC_Curve'] = (fpr, tpr)
    
    # Store test data for later use
    metrics['test_data'] = {'X_test': X_test_c, 'y_test': y_test_c}
    
    # Feature importance for best model (Random Forest)
    if 'Random Forest' in models:
        rf_model = models['Random Forest']
        feature_names = (numerical_features + 
                        rf_model.named_steps['preprocessor']
                        .named_transformers_['cat']
                        .get_feature_names_out(categorical_features).tolist())
        
        importances = rf_model.named_steps['classifier'].feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(20)
        models['feature_importance_clf'] = feature_importance_df

    # --- 2. REGRESSION (Multiple Algorithms) ---
    X_reg = df[categorical_features + numerical_features]
    y_reg = df['Willingness_to_Pay_AED']
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    regressors = {
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=200, max_depth=15),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=100)
    }
    
    for name, reg in regressors.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', reg)])
        pipeline.fit(X_train_r, y_train_r)
        y_pred_r = pipeline.predict(X_test_r)
        
        models[f"{name}_Regressor"] = pipeline
        
        metrics['regression'][name] = {
            'R2': r2_score(y_test_r, y_pred_r),
            'RMSE': np.sqrt(mean_squared_error(y_test_r, y_pred_r)),
            'MAE': np.mean(np.abs(y_test_r - y_pred_r))
        }
    
    # Store regression test data
    metrics['regression_test_data'] = {'y_test': y_test_r, 'y_pred': y_pred_r}

    # --- 3. CLUSTERING ---
    X_cluster = df[cluster_features]
    cluster_scaler = StandardScaler()
    X_cluster_scaled = cluster_scaler.fit_transform(X_cluster)
    
    # Elbow method data
    inertias = []
    silhouette_scores = []
    K_range = range(2, 8)
    for k in K_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_temp.fit(X_cluster_scaled)
        inertias.append(kmeans_temp.inertia_)
        silhouette_scores.append(silhouette_score(X_cluster_scaled, kmeans_temp.labels_))
    
    metrics['elbow_data'] = (list(K_range), inertias, silhouette_scores)
    
    # Final clustering with k=4
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_cluster_scaled)
    
    models['clustering_model'] = kmeans
    models['clustering_scaler'] = cluster_scaler
    metrics['silhouette_score'] = silhouette_score(X_cluster_scaled, df['Cluster'])
    
    # Cluster profiles
    cluster_profiles = df.groupby('Cluster')[cluster_features + ['Likely_to_Use_ReFillHub', 'Willingness_to_Pay_AED']].agg({
        **{feat: 'mean' for feat in cluster_features},
        'Likely_to_Use_ReFillHub': lambda x: (x == 'Yes').mean(),
        'Willingness_to_Pay_AED': 'mean'
    })
    models['cluster_profiles'] = cluster_profiles
    models['cluster_sizes'] = df['Cluster'].value_counts().sort_index()

    # --- 4. ASSOCIATION RULES ---
    transactions = df['Products_Bought'].apply(lambda x: [item.strip() for item in str(x).split(',')]).tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)
    
    frequent_itemsets = apriori(df_trans, min_support=0.05, use_colnames=True)
    
    if len(frequent_itemsets) > 0:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        models['association_rules'] = rules.sort_values('lift', ascending=False)
    else:
        models['association_rules'] = pd.DataFrame()
    
    models['frequent_itemsets'] = frequent_itemsets
    
    return models, metrics, df

# =============================================================================
# MAIN APP EXECUTION
# =============================================================================

def main():
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.df = None
        st.session_state.models = None
        st.session_state.metrics = None
    
    # =========================================================================
    # SIDEBAR
    # =========================================================================
    st.sidebar.image("https://via.placeholder.com/400x200/2E8B57/FFFFFF?text=ReFill+Hub+Analytics", 
                     use_column_width=True)
    st.sidebar.title("🧭 Navigation Hub")
    
    # Data Source Selection
    st.sidebar.header("📊 Data Source")
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Use Default Data", "Upload Custom Data", "Generate Synthetic Data"],
        help="Select how you want to load the data"
    )
    
    uploaded_file = None
    use_synthetic = False
    
    if data_source == "Upload Custom Data":
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload your own survey data in CSV format"
        )
        if uploaded_file and st.sidebar.button("📤 Load Uploaded Data"):
            st.session_state.data_loaded = False
    
    elif data_source == "Generate Synthetic Data":
        n_samples = st.sidebar.number_input("Number of samples", 100, 2000, 600, 50)
        if st.sidebar.button("🎲 Generate Data"):
            use_synthetic = True
            st.session_state.data_loaded = False
    
    # Load data based on selection
    if not st.session_state.data_loaded:
        with st.spinner("🔄 Loading and processing data..."):
            df, cat_features, num_features, cluster_features = load_and_clean_data(
                filepath='ReFillHub_SyntheticSurvey.csv',
                uploaded_file=uploaded_file,
                use_synthetic=use_synthetic
            )
            
            if df is not None:
                models, metrics, df_clustered = train_all_models(
                    df.copy(), cat_features, num_features, cluster_features
                )
                st.session_state.df = df_clustered
                st.session_state.models = models
                st.session_state.metrics = metrics
                st.session_state.cat_features = cat_features
                st.session_state.num_features = num_features
                st.session_state.cluster_features = cluster_features
                st.session_state.data_loaded = True
    
    if not st.session_state.data_loaded or st.session_state.df is None:
        st.error("❌ Failed to load data. Please check your data source.")
        return
    
    # Retrieve from session state
    df = st.session_state.df
    models = st.session_state.models
    metrics = st.session_state.metrics
    cat_features = st.session_state.cat_features
    num_features = st.session_state.num_features
    cluster_features = st.session_state.cluster_features
    
    # Global Filters
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Global Filters")
    with st.sidebar.expander("Apply Filters", expanded=False):
        selected_emirates = st.multiselect(
            "Filter by Emirate",
            options=df['Emirate'].unique(),
            default=df['Emirate'].unique()
        )
        
        selected_income = st.multiselect(
            "Filter by Income",
            options=df['Income'].unique(),
            default=df['Income'].unique()
        )
        
        selected_age = st.multiselect(
            "Filter by Age Group",
            options=df['Age_Group'].unique(),
            default=df['Age_Group'].unique()
        )
    
    # Apply filters
    df_filtered = df[
        (df['Emirate'].isin(selected_emirates)) &
        (df['Income'].isin(selected_income)) &
        (df['Age_Group'].isin(selected_age))
    ]
    
    # Page Selection
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "📑 Select Page:",
        [
            "🏠 Executive Summary",
            "🤖 Model Performance & Comparison",
            "🔮 Predictive Simulator",
            "🧩 Customer Segmentation",
            "🛒 Market Basket Analysis",
            "📊 Data Explorer & Quality",
            "📈 Advanced Analytics",
            "💡 Business Recommendations",
            "📥 Export & Download"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Active Records:** {len(df_filtered):,} / {len(df):,}")
    st.sidebar.success(f"**Models Trained:** {len(models)} algorithms")
    
    # =========================================================================
    # PAGE 1: EXECUTIVE SUMMARY
    # =========================================================================
    if "Executive Summary" in page:
        st.title("♻️ ReFill Hub: Executive Dashboard")
        st.markdown("""
        <div class='info-box'>
        <b>🎯 Overview:</b> Comprehensive market intelligence based on {0:,} survey respondents across UAE
        </div>
        """.format(len(df_filtered)), unsafe_allow_html=True)
        
        # Top Metrics Row
        st.header("📊 Key Performance Indicators")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        adoption_rate = (df_filtered['Likely_to_Use_ReFillHub'] == 'Yes').mean() * 100
        avg_wtp = df_filtered['Willingness_to_Pay_AED'].mean()
        high_value = (df_filtered['Willingness_to_Pay_AED'] > df_filtered['Willingness_to_Pay_AED'].quantile(0.75)).sum()
        best_accuracy = max(m['Accuracy'] for m in metrics['classification'].values())
        
        with col1:
            st.metric("Total Respondents", f"{len(df_filtered):,}")
        with col2:
            st.metric("Adoption Rate", f"{adoption_rate:.1f}%", 
                     delta=f"{adoption_rate - 50:.1f}% vs. 50%")
        with col3:
            st.metric("Avg. Willingness to Pay", f"AED {avg_wtp:.2f}")
        with col4:
            st.metric("High-Value Customers", f"{high_value:,}",
                     help="Top 25% spending bracket")
        with col5:
            st.metric("Best Model Accuracy", f"{best_accuracy:.1%}",
                     help="Highest performing classifier")
        
        st.markdown("---")
        
        # Visualizations Row 1
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("📍 Adoption by Income Level")
            fig = px.histogram(
                df_filtered, 
                x="Income", 
                color="Likely_to_Use_ReFillHub",
                barmode="group",
                color_discrete_map={'Yes': '#2E8B57', 'No': '#FF6347'},
                title="Income Level Impact on Adoption"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_b:
            st.subheader("💰 Spending Distribution")
            fig = px.box(
                df_filtered,
                x="Likely_to_Use_ReFillHub",
                y="Willingness_to_Pay_AED",
                color="Likely_to_Use_ReFillHub",
                color_discrete_map={'Yes': '#2E8B57', 'No': '#FF6347'},
                title="Willingness to Pay: Adopters vs. Non-Adopters"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Visualizations Row 2
        col_c, col_d = st.columns(2)
        
        with col_c:
            st.subheader("🗺️ Geographic Distribution")
            emirate_counts = df_filtered['Emirate'].value_counts()
            fig = px.pie(
                values=emirate_counts.values,
                names=emirate_counts.index,
                title="Respondent Distribution by Emirate",
                color_discrete_sequence=px.colors.sequential.Greens
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_d:
            st.subheader("🎯 Purchase Frequency Patterns")
            freq_data = df_filtered.groupby(['Purchase_Frequency', 'Likely_to_Use_ReFillHub']).size().reset_index(name='count')
            fig = px.bar(
                freq_data,
                x='Purchase_Frequency',
                y='count',
                color='Likely_to_Use_ReFillHub',
                barmode='group',
                color_discrete_map={'Yes': '#2E8B57', 'No': '#FF6347'},
                title="Shopping Frequency vs. Adoption"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Key Insights
        st.header("🔍 Key Insights")
        col_i1, col_i2, col_i3 = st.columns(3)
        
        with col_i1:
            eco_aware = (df_filtered['Aware_Plastic_Ban'] == 'Yes').mean() * 100
            st.metric("Plastic Ban Awareness", f"{eco_aware:.0f}%")
            st.caption("High awareness presents opportunity")
        
        with col_i2:
            used_before = (df_filtered['Used_Refill_Before'] == 'Yes').mean() * 100
            st.metric("Prior Refill Experience", f"{used_before:.0f}%")
            st.caption("Educational campaigns needed")
        
        with col_i3:
            eco_users = (df_filtered['Uses_Eco_Products'] == 'Yes').mean() * 100
            st.metric("Eco-Product Users", f"{eco_users:.0f}%")
            st.caption("Target segment identified")

    # =========================================================================
    # PAGE 2: MODEL PERFORMANCE & COMPARISON
    # =========================================================================
    elif "Model Performance" in page:
        st.title("🤖 Advanced Model Performance & Benchmarking")
        
        st.markdown("""
        <div class='success-box'>
        <b>🎓 Methodology:</b> Comprehensive comparison of 4 classification algorithms and 2 regression models
        </div>
        """, unsafe_allow_html=True)
        
        # Classification Comparison
        st.header("🎯 Classification Models: Adoption Prediction")
        
        # Metrics Table
        st.subheader("📊 Performance Metrics Comparison")
        metrics_df = pd.DataFrame(metrics['classification']).T
        metrics_display = metrics_df[['Accuracy', 'Precision', 'Recall', 'F1 Score']].copy()
        
        # Highlight best values
        st.dataframe(
            metrics_display.style.highlight_max(axis=0, color='#d6f5d6').format('{:.4f}'),
            use_container_width=True
        )
        
        # Best Model Highlight
        best_model = metrics_display['Accuracy'].idxmax()
        st.success(f"🏆 **Best Performing Model:** {best_model} with {metrics_display.loc[best_model, 'Accuracy']:.2%} accuracy")
        
        # Confusion Matrices
        st.subheader("🔍 Confusion Matrix Analysis")
        
        models_list = list(metrics['classification'].keys())
        n_models = len(models_list)
        cols = st.columns(min(n_models, 4))
        
        for i, model_name in enumerate(models_list):
            with cols[i % len(cols)]:
                st.markdown(f"**{model_name}**")
                cm = metrics['classification'][model_name]['Confusion Matrix']
                
                # Create annotated heatmap
                fig = ff.create_annotated_heatmap(
                    z=cm,
                    x=['Predicted No', 'Predicted Yes'],
                    y=['Actual No', 'Actual Yes'],
                    colorscale='Greens',
                    showscale=True
                )
                fig.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate additional metrics
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                st.caption(f"Specificity: {specificity:.2%}")
        
        # ROC Curves Comparison
        st.subheader("📈 ROC Curve Comparison")
        
        fig_roc = go.Figure()
        
        for model_name in models_list:
            if 'ROC_Curve' in metrics['classification'][model_name]:
                fpr, tpr = metrics['classification'][model_name]['ROC_Curve']
                auc_score = metrics['classification'][model_name]['ROC_AUC']
                
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{model_name} (AUC = {auc_score:.3f})',
                    line=dict(width=3)
                ))
        
        # Add diagonal reference line
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', dash='dash', width=2)
        ))
        
        fig_roc.update_layout(
            title="ROC Curves: Model Comparison",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            hovermode='x',
            height=500
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        
        # Performance Summary
        st.subheader("🎯 Model Selection Recommendations")
        
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            st.markdown("""
            <div class='info-box'>
            <h4>🚀 Production Deployment</h4>
            <p><b>Recommended:</b> Random Forest or Gradient Boosting</p>
            <ul>
                <li>High accuracy with robust performance</li>
                <li>Handles non-linear relationships well</li>
                <li>Built-in feature importance</li>
                <li>Less prone to overfitting than Decision Trees</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col_rec2:
            st.markdown("""
            <div class='warning-box'>
            <h4>⚠️ Model Considerations</h4>
            <p><b>Trade-offs:</b></p>
            <ul>
                <li><b>Decision Tree:</b> Fast but may overfit</li>
                <li><b>Random Forest:</b> Slower but more accurate</li>
                <li><b>Gradient Boosting:</b> Best performance but computationally expensive</li>
                <li><b>Logistic Regression:</b> Interpretable but limited for non-linear data</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Regression Comparison
        st.header("💰 Regression Models: Spending Prediction")
        
        st.subheader("📊 Regression Performance Metrics")
        
        reg_metrics = pd.DataFrame(metrics['regression']).T
        st.dataframe(
            reg_metrics.style.highlight_max(subset=['R2'], color='#d6f5d6')
                           .highlight_min(subset=['RMSE', 'MAE'], color='#d6f5d6')
                           .format({'R2': '{:.4f}', 'RMSE': '{:.2f}', 'MAE': '{:.2f}'}),
            use_container_width=True
        )
        
        # Best Regression Model
        best_reg_model = reg_metrics['R2'].idxmax()
        st.success(f"🏆 **Best Regression Model:** {best_reg_model} with R² = {reg_metrics.loc[best_reg_model, 'R2']:.4f}")
        
        # Regression Visualization
        col_reg1, col_reg2 = st.columns(2)
        
        with col_reg1:
            st.subheader("Model Performance Comparison")
            fig_reg_comp = go.Figure(data=[
                go.Bar(name='R² Score', x=reg_metrics.index, y=reg_metrics['R2'], marker_color='#2E8B57'),
            ])
            fig_reg_comp.update_layout(
                title="R² Score Comparison",
                yaxis_title="R² Score",
                yaxis_range=[0, 1],
                height=400
            )
            st.plotly_chart(fig_reg_comp, use_container_width=True)
        
        with col_reg2:
            st.subheader("Prediction Error Comparison")
            fig_error = go.Figure(data=[
                go.Bar(name='RMSE', x=reg_metrics.index, y=reg_metrics['RMSE'], marker_color='#FF6347'),
                go.Bar(name='MAE', x=reg_metrics.index, y=reg_metrics['MAE'], marker_color='#FFA500')
            ])
            fig_error.update_layout(
                title="Error Metrics Comparison",
                yaxis_title="Error (AED)",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig_error, use_container_width=True)
        
        # Feature Importance (if available)
        if 'feature_importance_clf' in models:
            st.markdown("---")
            st.header("🔍 Feature Importance Analysis")
            
            feature_imp = models['feature_importance_clf']
            
            fig_fi = px.bar(
                feature_imp.head(15),
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 15 Most Important Features for Adoption Prediction",
                color='Importance',
                color_continuous_scale='Greens'
            )
            fig_fi.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
            st.plotly_chart(fig_fi, use_container_width=True)
            
            st.info("💡 **Insight:** Focus marketing and product development on the top-ranking features to maximize adoption.")

    # =========================================================================
    # PAGE 3: PREDICTIVE SIMULATOR
    # =========================================================================
    elif "Predictive Simulator" in page:
        st.title("🔮 AI-Powered Predictive Simulator")
        
        st.markdown("""
        <div class='info-box'>
        <b>🎯 Objective:</b> Simulate customer profiles and predict adoption likelihood + spending potential
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🎯 Single Customer Prediction", "📊 Scenario Comparison"])
        
        with tab1:
            st.subheader("Configure Customer Profile")
            
            with st.form("prediction_form"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**👤 Demographics**")
                    age = st.selectbox("Age Group", df['Age_Group'].unique())
                    gender = st.selectbox("Gender", df['Gender'].unique())
                    income = st.selectbox("Income Level (AED)", df['Income'].unique())
                    emirate = st.selectbox("Emirate", df['Emirate'].unique())
                    family_size = st.selectbox("Family Size", df['Family_Size'].unique())
                
                with col2:
                    st.markdown("**💭 Attitudes (1-5 Scale)**")
                    imp_price = st.slider("Price Importance", 1, 5, 3)
                    imp_sust = st.slider("Sustainability Importance", 1, 5, 3)
                    imp_conv = st.slider("Convenience Importance", 1, 5, 3)
                    waste_score = st.slider("Waste Reduction Effort", 1, 5, 3)
                    social_score = st.slider("Social Influence", 1, 5, 3)
                
                with col3:
                    st.markdown("**🛒 Behaviors**")
                    eco_brand = st.select_slider("Eco-Brand Preference", [1, 2, 3, 4, 5], 3)
                    follow_camp = st.selectbox("Follows Green Campaigns?", ["Yes", "No"])
                    used_before = st.selectbox("Used Refill Before?", ["Yes", "No"])
                    freq = st.selectbox("Purchase Frequency", df['Purchase_Frequency'].unique())
                    try_likelihood = st.slider("Initial Interest Level", 1, 5, 3)
                
                col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
                with col_btn2:
                    submitted = st.form_submit_button("🚀 Run Prediction", use_container_width=True)
            
            if submitted:
                # Prepare input (fill missing features with mode)
                input_row = pd.DataFrame({col: [df[col].mode()[0]] for col in cat_features + num_features})
                
                # Update with user inputs
                input_row.update({
                    'Age_Group': [age],
                    'Gender': [gender],
                    'Income': [income],
                    'Emirate': [emirate],
                    'Family_Size': [family_size],
                    'Importance_Price': [imp_price],
                    'Importance_Sustainability': [imp_sust],
                    'Importance_Convenience': [imp_conv],
                    'Reduce_Waste_Score': [waste_score],
                    'Social_Influence_Score': [social_score],
                    'Eco_Brand_Preference': [eco_brand],
                    'Follow_Campaigns': [follow_camp],
                    'Used_Refill_Before': [used_before],
                    'Purchase_Frequency': [freq],
                    'Try_Refill_Likelihood': [try_likelihood]
                })
                
                # Family size numeric
                if '5+' in family_size: input_row['Family_Size_Num'] = [5.0]
                elif '1-2' in family_size: input_row['Family_Size_Num'] = [1.5]
                elif '3-4' in family_size: input_row['Family_Size_Num'] = [3.5]
                
                # Predict with best model
                best_classifier = max(metrics['classification'], key=lambda x: metrics['classification'][x]['Accuracy'])
                best_regressor = max(metrics['regression'], key=lambda x: metrics['regression'][x]['R2'])
                
                clf_model = models[best_classifier]
                reg_model = models[f"{best_regressor}_Regressor"]
                
                adoption_prob = clf_model.predict_proba(input_row)[0][1]
                predicted_spend = reg_model.predict(input_row)[0]
                
                # Display Results
                st.markdown("---")
                st.header("🎯 Prediction Results")
                
                col_r1, col_r2 = st.columns(2)
                
                with col_r1:
                    st.subheader("📊 Adoption Likelihood")
                    
                    # Gauge Chart
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=adoption_prob * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Adoption Probability (%)", 'font': {'size': 20}},
                        delta={'reference': 50, 'increasing': {'color': "#2E8B57"}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 2},
                            'bar': {'color': "#2E8B57" if adoption_prob > 0.5 else "#FF6347"},
                            'steps': [
                                {'range': [0, 30], 'color': 'rgba(255, 99, 71, 0.3)'},
                                {'range': [30, 70], 'color': 'rgba(255, 206, 86, 0.3)'},
                                {'range': [70, 100], 'color': 'rgba(46, 139, 87, 0.3)'}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 3},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=350)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # Interpretation
                    if adoption_prob > 0.7:
                        st.success("✅ **High Probability** - Strong candidate for conversion!")
                    elif adoption_prob > 0.4:
                        st.info("ℹ️ **Moderate Probability** - Needs nurturing and value demonstration")
                    else:
                        st.warning("⚠️ **Low Probability** - Requires significant incentives")
                    
                    st.metric("Model Used", best_classifier, 
                             help=f"Accuracy: {metrics['classification'][best_classifier]['Accuracy']:.2%}")
                
                with col_r2:
                    st.subheader("💰 Spending Potential")
                    
                    avg_wtp = df['Willingness_to_Pay_AED'].mean()
                    percentile = (df['Willingness_to_Pay_AED'] < predicted_spend).mean() * 100
                    
                    st.metric(
                        "Predicted Spending per Visit",
                        f"AED {predicted_spend:.2f}",
                        delta=f"{predicted_spend - avg_wtp:+.2f} vs. average",
                        delta_color="normal" if predicted_spend > avg_wtp else "inverse"
                    )
                    
                    st.metric("Spending Percentile", f"{percentile:.0f}th")
                    
                    # Comparison Chart
                    fig_compare = go.Figure()
                    fig_compare.add_trace(go.Bar(
                        x=['Market Average', 'This Customer'],
                        y=[avg_wtp, predicted_spend],
                        marker_color=['#808080', '#2E8B57'],
                        text=[f'AED {avg_wtp:.2f}', f'AED {predicted_spend:.2f}'],
                        textposition='auto'
                    ))
                    fig_compare.update_layout(
                        title="Spending Comparison",
                        yaxis_title="Willingness to Pay (AED)",
                        showlegend=False,
                        height=300
                    )
                    st.plotly_chart(fig_compare, use_container_width=True)
                    
                    st.metric("Model Used", best_regressor,
                             help=f"R²: {metrics['regression'][best_regressor]['R2']:.4f}")
                
                # Customer Lifetime Value
                st.markdown("---")
                st.subheader("💎 Customer Lifetime Value (CLV) Analysis")
                
                col_clv1, col_clv2, col_clv3, col_clv4 = st.columns(4)
                
                with col_clv1:
                    visits_year = st.number_input("Expected Visits/Year", 4, 52, 24, 
                                                 help="Weekly ≈ 52, Bi-weekly ≈ 24")
                with col_clv2:
                    retention = st.slider("Retention Rate", 0.5, 0.95, 0.75, 0.05)
                with col_clv3:
                    years = st.number_input("Time Horizon (Years)", 1, 10, 3)
                with col_clv4:
                    margin = st.slider("Gross Margin %", 20, 60, 40)
                
                clv = calculate_clv(predicted_spend, visits_year, retention, years)
                clv_profit = clv * (margin / 100)
                
                col_clv_r1, col_clv_r2, col_clv_r3 = st.columns(3)
                
                with col_clv_r1:
                    st.metric("Total CLV (Revenue)", f"AED {clv:,.2f}")
                with col_clv_r2:
                    st.metric("Gross Profit", f"AED {clv_profit:,.2f}")
                with col_clv_r3:
                    cac = 50  # Assumed Customer Acquisition Cost
                    roi = ((clv_profit - cac) / cac) * 100
                    st.metric("ROI", f"{roi:.0f}%", help=f"Assuming CAC = AED {cac}")
        
        with tab2:
            st.subheader("📊 Scenario Comparison Tool")
            st.markdown("Compare multiple customer profiles side-by-side")
            
            st.info("🚧 **Coming Soon:** Batch scenario comparison with downloadable reports")

    # =========================================================================
    # PAGE 4: CUSTOMER SEGMENTATION
    # =========================================================================
    elif "Customer Segmentation" in page:
        st.title("🧩 Advanced Customer Segmentation")
        
        st.markdown("""
        <div class='info-box'>
        <b>🎯 Methodology:</b> K-Means Clustering with optimal k=4 based on elbow method and silhouette analysis
        </div>
        """, unsafe_allow_html=True)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.metric("Silhouette Score", f"{metrics['silhouette_score']:.3f}",
                     help="Quality of clustering (0.3-0.5 is good for behavioral data)")
        with col_m2:
            st.metric("Number of Clusters", "4")
        with col_m3:
            st.metric("Features Used", len(cluster_features))
        
        # Elbow & Silhouette Analysis
        st.subheader("🔍 Cluster Optimization Analysis")
        
        col_opt1, col_opt2 = st.columns(2)
        
        K_range, inertias, silhouette_scores = metrics['elbow_data']
        
        with col_opt1:
            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(
                x=list(K_range), y=inertias,
                mode='lines+markers',
                name='Inertia',
                line=dict(color='#2E8B57', width=3),
                marker=dict(size=10)
            ))
            fig_elbow.update_layout(
                title="Elbow Method for Optimal K",
                xaxis_title="Number of Clusters (k)",
                yaxis_title="Within-Cluster Sum of Squares (Inertia)",
                hovermode='x'
            )
            st.plotly_chart(fig_elbow, use_container_width=True)
        
        with col_opt2:
            fig_sil = go.Figure()
            fig_sil.add_trace(go.Scatter(
                x=list(K_range), y=silhouette_scores,
                mode='lines+markers',
                name='Silhouette Score',
                line=dict(color='#228B22', width=3),
                marker=dict(size=10)
            ))
            fig_sil.update_layout(
                title="Silhouette Score Analysis",
                xaxis_title="Number of Clusters (k)",
                yaxis_title="Silhouette Score",
                hovermode='x'
            )
            st.plotly_chart(fig_sil, use_container_width=True)
        
        # 3D Visualization
        st.subheader("🎨 Interactive 3D Cluster Visualization")
        
        df_filtered['Cluster_Label'] = df_filtered['Cluster'].astype(str)
        
        fig_3d = px.scatter_3d(
            df_filtered,
            x='Importance_Price',
            y='Importance_Sustainability',
            z='Reduce_Waste_Score',
            color='Cluster_Label',
            symbol='Likely_to_Use_ReFillHub',
            opacity=0.7,
            title="Customer Segments in 3D Psychographic Space",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hover_data=['Emirate', 'Income', 'Willingness_to_Pay_AED']
        )
        fig_3d.update_layout(
            scene=dict(
                xaxis_title="Price Importance",
                yaxis_title="Sustainability Importance",
                zaxis_title="Waste Reduction Score"
            ),
            height=600
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Cluster Profiles
        st.subheader("📊 Detailed Cluster Profiles")
        
        cluster_profiles = models['cluster_profiles']
        cluster_sizes = models['cluster_sizes']
        
        # Enhanced profile table
        profile_display = cluster_profiles.copy()
        profile_display['Cluster_Size'] = cluster_sizes
        profile_display['Size_Percent'] = (cluster_sizes / len(df_filtered) * 100).round(1)
        profile_display['Adoption_Rate'] = (profile_display['Likely_to_Use_ReFillHub'] * 100).round(1)
        
        st.dataframe(
            profile_display.style.background_gradient(cmap='Greens', subset=cluster_features)
                                .format(precision=2),
            use_container_width=True
        )
        
        # Radar Chart Comparison
        st.subheader("⚖️ Cluster Comparison: Radar Chart")
        
        categories = cluster_features
        
        fig_radar = go.Figure()
        
        for cluster_id in sorted(df_filtered['Cluster'].unique()):
            values = cluster_profiles.loc[cluster_id, cluster_features].values.tolist()
            values += values[:1]  # Close polygon
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=f'Cluster {cluster_id}'
            ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            showlegend=True,
            title="Cluster Profiles: Multi-Dimensional Comparison",
            height=500
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Cluster Value Matrix
        st.subheader("💎 Cluster Value Matrix")
        
        cluster_stats = df_filtered.groupby('Cluster').agg({
            'Likely_to_Use_ReFillHub': lambda x: (x == 'Yes').mean() * 100,
            'Willingness_to_Pay_AED': 'mean',
            'Cluster': 'count'
        }).rename(columns={
            'Likely_to_Use_ReFillHub': 'Adoption_Rate',
            'Willingness_to_Pay_AED': 'Avg_Spend',
            'Cluster': 'Size'
        }).reset_index()
        
        fig_bubble = px.scatter(
            cluster_stats,
            x='Adoption_Rate',
            y='Avg_Spend',
            size='Size',
            color='Cluster',
            title="Cluster Opportunity Matrix: Adoption vs. Spending",
            labels={'Adoption_Rate': 'Adoption Rate (%)', 'Avg_Spend': 'Avg. Spending (AED)'},
            color_discrete_sequence=px.colors.qualitative.Set2,
            hover_data=['Size']
        )
        fig_bubble.update_layout(height=500)
        st.plotly_chart(fig_bubble, use_container_width=True)
        
        # Persona Descriptions
        st.subheader("👥 Customer Personas & Strategic Recommendations")
        
        personas = {
            0: {
                'name': '🌱 The Eco-Warriors',
                'description': 'High sustainability focus, willing to pay premium for environmental impact',
                'strategy': [
                    '• Lead with impact metrics (e.g., "You saved 100 plastic bottles")',
                    '• Premium eco-brand partnerships and certifications',
                    '• Sustainability badges and gamification',
                    '• Community events and environmental campaigns'
                ],
                'color': 'success'
            },
            1: {
                'name': '💰 The Value Seekers',
                'description': 'Price-sensitive shoppers looking for savings and best deals',
                'strategy': [
                    '• Emphasize cost savings: "Save 20% vs. packaged products"',
                    '• Subscription plans with volume discounts',
                    '• Loyalty programs with cashback rewards',
                    '• Price comparison tools in app'
                ],
                'color': 'info'
            },
            2: {
                'name': '⚡ The Convenience Seekers',
                'description': 'Busy professionals who prioritize speed and accessibility',
                'strategy': [
                    '• Strategic locations: offices, malls, residential complexes',
                    '• Mobile app with pre-ordering and QR code payments',
                    '• Express checkout lanes',
                    '• Delivery/subscription services'
                ],
                'color': 'warning'
            },
            3: {
                'name': '😐 The Skeptics',
                'description': 'Low engagement, need strong incentives to try',
                'strategy': [
                    '• Aggressive first-time discounts (50% off)',
                    '• Influencer partnerships and testimonials',
                    '• Free trial programs',
                    '• Simplified onboarding with immediate rewards'
                ],
                'color': 'error'
            }
        }
        
        for cluster_id, persona in personas.items():
            with st.expander(f"**Cluster {cluster_id}: {persona['name']}** (Size: {cluster_sizes[cluster_id]} | {profile_display.loc[cluster_id, 'Size_Percent']:.1f}%)"):
                st.markdown(f"**Profile:** {persona['description']}")
                st.markdown("**Recommended Strategies:**")
                for strategy in persona['strategy']:
                    st.markdown(strategy)
                
                # Cluster-specific metrics
                cluster_data = df_filtered[df_filtered['Cluster'] == cluster_id]
                col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                
                with col_p1:
                    adoption = (cluster_data['Likely_to_Use_ReFillHub'] == 'Yes').mean() * 100
                    st.metric("Adoption Rate", f"{adoption:.1f}%")
                
                with col_p2:
                    avg_spend = cluster_data['Willingness_to_Pay_AED'].mean()
                    st.metric("Avg. Spending", f"AED {avg_spend:.2f}")
                
                with col_p3:
                    sustainability = cluster_data['Importance_Sustainability'].mean()
                    st.metric("Sustainability", f"{sustainability:.2f}/5")
                
                with col_p4:
                    price_sensitivity = cluster_data['Importance_Price'].mean()
                    st.metric("Price Sensitivity", f"{price_sensitivity:.2f}/5")

    # =========================================================================
    # PAGE 5: MARKET BASKET ANALYSIS
    # =========================================================================
    elif "Market Basket" in page:
        st.title("🛒 Market Basket Analysis & Product Associations")
        
        st.markdown("""
        <div class='info-box'>
        <b>🎯 Objective:</b> Discover which products are frequently purchased together using Association Rule Mining
        </div>
        """, unsafe_allow_html=True)
        
        rules_df = models['association_rules']
        
        if rules_df.empty:
            st.warning("⚠️ No association rules found. Try lowering the support/confidence thresholds.")
            return
        
        # Filters
        st.subheader("🔧 Rule Filtering Controls")
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            min_lift = st.slider(
                "Minimum Lift",
                min_value=1.0,
                max_value=float(rules_df['lift'].max()),
                value=1.2,
                step=0.1,
                help="Lift > 1 means products are bought together more than by chance"
            )
        
        with col_f2:
            min_confidence = st.slider(
                "Minimum Confidence",
                min_value=0.0,
                max_value=float(rules_df['confidence'].max()),
                value=0.1,
                step=0.05,
                help="Probability that consequent is purchased given antecedent"
            )
        
        with col_f3:
            min_support = st.slider(
                "Minimum Support",
                min_value=0.0,
                max_value=float(rules_df['support'].max()),
                value=0.05,
                step=0.01,
                help="How frequently the itemset appears in transactions"
            )
        
        filtered_rules = rules_df[
            (rules_df['lift'] > min_lift) &
            (rules_df['confidence'] > min_confidence) &
            (rules_df['support'] > min_support)
        ]
        
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.metric("Total Rules Found", len(filtered_rules))
        with col_m2:
            st.metric("Avg. Lift", f"{filtered_rules['lift'].mean():.2f}" if len(filtered_rules) > 0 else "N/A")
        with col_m3:
            st.metric("Max Lift", f"{filtered_rules['lift'].max():.2f}" if len(filtered_rules) > 0 else "N/A")
        
        # Top Rules Table
        st.subheader("🏆 Top Association Rules")
        
        display_rules = filtered_rules.head(20)[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
        st.dataframe(
            display_rules.style.background_gradient(cmap='Greens', subset=['lift', 'confidence'])
                              .format({'support': '{:.3f}', 'confidence': '{:.3f}', 'lift': '{:.2f}'}),
            use_container_width=True
        )
        
        # Visualizations
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            st.subheader("📊 Top Rules by Lift")
            top_rules = filtered_rules.sort_values('lift', ascending=False).head(15)
            top_rules['Rule'] = top_rules['antecedents'] + " ➡️ " + top_rules['consequents']
            
            fig_rules = px.bar(
                top_rules,
                x="lift",
                y="Rule",
                orientation='h',
                title="Strongest Product Associations",
                color="confidence",
                color_continuous_scale='Greens',
                hover_data=['support', 'confidence', 'lift']
            )
            fig_rules.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
            st.plotly_chart(fig_rules, use_container_width=True)
        
        with col_v2:
            st.subheader("🎯 Support vs Confidence Matrix")
            fig_scatter = px.scatter(
                filtered_rules,
                x='support',
                y='confidence',
                size='lift',
                color='lift',
                hover_data=['antecedents', 'consequents'],
                title="Rule Quality Visualization",
                color_continuous_scale='Greens'
            )
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Network Visualization
        st.subheader("🕸️ Product Association Network")
        
        import networkx as nx
        
        top_network_rules = filtered_rules.sort_values('lift', ascending=False).head(20)
        
        G = nx.DiGraph()
        for _, row in top_network_rules.iterrows():
            G.add_edge(row['antecedents'], row['consequents'], weight=row['lift'])
        
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        edge_traces = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=1, color='#888'),
                    hoverinfo='none',
                    showlegend=False
                )
            )
        
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            hoverinfo='text',
            text=[node for node in G.nodes()],
            textposition="top center",
            marker=dict(
                size=[G.degree(node) * 15 for node in G.nodes()],
                color='#2E8B57',
                line=dict(width=2, color='white')
            ),
            showlegend=False
        )
        
        fig_network = go.Figure(data=edge_traces + [node_trace])
        fig_network.update_layout(
            title="Product Co-Occurrence Network (Size = Connection Strength)",
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            plot_bgcolor='#F5F7F9'
        )
        st.plotly_chart(fig_network, use_container_width=True)
        
        # Strategic Insights
        st.subheader("💡 Strategic Business Recommendations")
        
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.markdown("""
            <div class='success-box'>
            <h4>🎁 Product Bundling Opportunities</h4>
            <p>Create attractive bundles based on high-lift associations:</p>
            <ul>
            """, unsafe_allow_html=True)
            
            for idx, row in filtered_rules.sort_values('lift', ascending=False).head(5).iterrows():
                st.markdown(f"<li><b>{row['antecedents']}</b> + <b>{row['consequents']}</b> (Lift: {row['lift']:.2f})</li>", unsafe_allow_html=True)
            
            st.markdown("</ul></div>", unsafe_allow_html=True)
        
        with col_s2:
            st.markdown("""
            <div class='info-box'>
            <h4>🏪 Kiosk Layout Optimization</h4>
            <p>Place these products adjacent to each other:</p>
            <ul>
            """, unsafe_allow_html=True)
            
            for idx, row in filtered_rules.sort_values('confidence', ascending=False).head(5).iterrows():
                st.markdown(f"<li><b>{row['antecedents']}</b> → <b>{row['consequents']}</b> ({row['confidence']*100:.0f}% confidence)</li>", unsafe_allow_html=True)
            
            st.markdown("</ul></div>", unsafe_allow_html=True)
        
        # Product Popularity
        st.subheader("📦 Individual Product Popularity")
        
        frequent_itemsets = models['frequent_itemsets']
        single_items = frequent_itemsets[frequent_itemsets['itemsets'].apply(len) == 1].copy()
        single_items['product'] = single_items['itemsets'].apply(lambda x: list(x)[0])
        single_items = single_items.sort_values('support', ascending=False).head(10)
        
        fig_items = px.bar(
            single_items,
            x='product',
            y='support',
            title="Top 10 Most Purchased Products",
            labels={'support': 'Purchase Frequency (Support)', 'product': 'Product'},
            color='support',
            color_continuous_scale='Greens'
        )
        fig_items.update_layout(height=400)
        st.plotly_chart(fig_items, use_container_width=True)

    # =========================================================================
    # PAGE 6: DATA EXPLORER & QUALITY
    # =========================================================================
    elif "Data Explorer" in page:
        st.title("📊 Data Explorer & Quality Assessment")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Raw Data", "📈 Distributions", "🔗 Correlations", "✅ Quality Checks"])
        
        with tab1:
            st.subheader("Dataset Overview")
            
            col_ov1, col_ov2, col_ov3, col_ov4 = st.columns(4)
            
            with col_ov1:
                st.metric("Total Records", len(df_filtered))
            with col_ov2:
                st.metric("Total Features", len(df_filtered.columns))
            with col_ov3:
                st.metric("Categorical Features", len(df_filtered.select_dtypes(include='object').columns))
            with col_ov4:
                st.metric("Numerical Features", len(df_filtered.select_dtypes(include=['int64', 'float64']).columns))
            
            st.dataframe(df_filtered.head(100), use_container_width=True)
            
            st.subheader("Statistical Summary")
            st.dataframe(df_filtered.describe(), use_container_width=True)
        
        with tab2:
            st.subheader("Feature Distribution Analysis")
            
            col_dist1, col_dist2 = st.columns(2)
            
            with col_dist1:
                cat_cols = df_filtered.select_dtypes(include='object').columns.tolist()
                selected_cat = st.selectbox("Select Categorical Feature", cat_cols)
                
                fig_cat = px.histogram(
                    df_filtered,
                    x=selected_cat,
                    color='Likely_to_Use_ReFillHub',
                    barmode='group',
                    title=f"Distribution of {selected_cat}",
                    color_discrete_map={'Yes': '#2E8B57', 'No': '#FF6347'}
                )
                st.plotly_chart(fig_cat, use_container_width=True)
            
            with col_dist2:
                num_cols = df_filtered.select_dtypes(include=['int64', 'float64']).columns.tolist()
                selected_num = st.selectbox("Select Numerical Feature", num_cols)
                
                fig_num = px.histogram(
                    df_filtered,
                    x=selected_num,
                    marginal='box',
                    title=f"Distribution of {selected_num}",
                    color_discrete_sequence=['#2E8B57']
                )
                st.plotly_chart(fig_num, use_container_width=True)
        
        with tab3:
            st.subheader("Correlation Analysis")
            
            numeric_cols = df_filtered.select_dtypes(include=['int64', 'float64']).columns
            corr_matrix = df_filtered[numeric_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                title="Feature Correlation Heatmap",
                color_continuous_scale='RdYlGn',
                aspect='auto',
                labels=dict(color="Correlation"),
                zmin=-1, zmax=1
            )
            fig_corr.update_layout(height=600)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            st.subheader("Strongest Correlations")
            corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
            corr_pairs = corr_pairs[corr_pairs < 1]
            top_corr = corr_pairs.head(10)
            
            st.dataframe(
                pd.DataFrame(top_corr).reset_index().rename(columns={0: 'Correlation', 'level_0': 'Feature 1', 'level_1': 'Feature 2'}),
                use_container_width=True
            )
        
        with tab4:
            st.subheader("Data Quality Report")
            
            col_q1, col_q2 = st.columns(2)
            
            with col_q1:
                st.markdown("#### Missing Values Analysis")
                missing = df_filtered.isnull().sum()
                missing_pct = (missing / len(df_filtered) * 100).round(2)
                missing_df = pd.DataFrame({
                    'Feature': missing.index,
                    'Missing Count': missing.values,
                    'Missing %': missing_pct.values
                }).sort_values('Missing Count', ascending=False)
                
                if missing_df['Missing Count'].sum() == 0:
                    st.success("✅ No missing values detected!")
                else:
                    st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
            
            with col_q2:
                st.markdown("#### Duplicate Records")
                duplicates = df_filtered.duplicated().sum()
                st.metric("Duplicate Rows", duplicates)
                
                if duplicates == 0:
                    st.success("✅ No duplicate records found!")
                else:
                    st.warning(f"⚠️ Found {duplicates} duplicate records")
            
            st.markdown("#### Data Type Summary")
            dtype_df = pd.DataFrame({
                'Feature': df_filtered.dtypes.index,
                'Data Type': df_filtered.dtypes.values,
                'Unique Values': [df_filtered[col].nunique() for col in df_filtered.columns],
                'Sample Value': [df_filtered[col].iloc[0] for col in df_filtered.columns]
            })
            st.dataframe(dtype_df, use_container_width=True)
            
            st.markdown("#### Outlier Detection (IQR Method)")
            outlier_summary = []
            numeric_cols = df_filtered.select_dtypes(include=['int64', 'float64']).columns
            
            for col in numeric_cols:
                Q1 = df_filtered[col].quantile(0.25)
                Q3 = df_filtered[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df_filtered[col] < (Q1 - 1.5 * IQR)) | (df_filtered[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_summary.append({
                    'Feature': col,
                    'Outliers': outliers,
                    'Outlier %': f"{outliers/len(df_filtered)*100:.2f}%"
                })
            
            outlier_df = pd.DataFrame(outlier_summary).sort_values('Outliers', ascending=False)
            st.dataframe(outlier_df, use_container_width=True)

    # =========================================================================
    # PAGE 7: ADVANCED ANALYTICS
    # =========================================================================
    elif "Advanced Analytics" in page:
        st.title("📈 Advanced Statistical Analytics")
        
        tab1, tab2 = st.tabs(["📊 Statistical Tests", "🔬 Hypothesis Testing"])
        
        with tab1:
            st.subheader("Comprehensive Statistical Analysis")
            
            stat_results = perform_statistical_tests(df_filtered)
            
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                st.markdown("#### Chi-Square Test: Income vs. Adoption")
                st.markdown("""
                <div class='info-box'>
                <b>Null Hypothesis:</b> Income level and adoption likelihood are independent
                </div>
                """, unsafe_allow_html=True)
                
                chi2_result = stat_results['income_adoption_chi2']
                
                col_chi1, col_chi2 = st.columns(2)
                with col_chi1:
                    st.metric("Chi-Square Statistic", f"{chi2_result['chi2']:.2f}")
                with col_chi2:
                    st.metric("P-Value", f"{chi2_result['p_value']:.4f}")
                
                if chi2_result['p_value'] < 0.05:
                    st.success("✅ **Reject null hypothesis** (p < 0.05): Income significantly affects adoption")
                else:
                    st.info("ℹ️ Cannot reject null hypothesis (p ≥ 0.05)")
                
                # Contingency table heatmap
                contingency = pd.crosstab(df_filtered['Income'], df_filtered['Likely_to_Use_ReFillHub'])
                fig_chi = px.imshow(
                    contingency,
                    title="Contingency Table: Income × Adoption",
                    labels=dict(x="Adoption Decision", y="Income Level", color="Count"),
                    color_continuous_scale='Greens',
                    aspect='auto'
                )
                st.plotly_chart(fig_chi, use_container_width=True)
            
            with col_t2:
                st.markdown("#### Independent T-Test: Spending Comparison")
                st.markdown("""
                <div class='info-box'>
                <b>Null Hypothesis:</b> Mean spending is equal for adopters and non-adopters
                </div>
                """, unsafe_allow_html=True)
                
                ttest_result = stat_results['spending_ttest']
                
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    st.metric("T-Statistic", f"{ttest_result['t_stat']:.2f}")
                with col_t2:
                    st.metric("P-Value", f"{ttest_result['p_value']:.4f}")
                
                if ttest_result['p_value'] < 0.05:
                    st.success("✅ **Reject null hypothesis** (p < 0.05): Significant spending difference")
                else:
                    st.info("ℹ️ Cannot reject null hypothesis (p ≥ 0.05)")
                
                # Distribution comparison
                fig_dist = go.Figure()
                
                adopters = df_filtered[df_filtered['Likely_to_Use_ReFillHub'] == 'Yes']['Willingness_to_Pay_AED']
                non_adopters = df_filtered[df_filtered['Likely_to_Use_ReFillHub'] == 'No']['Willingness_to_Pay_AED']
                
                fig_dist.add_trace(go.Histogram(
                    x=adopters, name='Adopters',
                    opacity=0.7, marker_color='#2E8B57'
                ))
                fig_dist.add_trace(go.Histogram(
                    x=non_adopters, name='Non-Adopters',
                    opacity=0.7, marker_color='#FF6347'
                ))
                
                fig_dist.update_layout(
                    barmode='overlay',
                    title="Spending Distribution Comparison",
                    xaxis_title="Willingness to Pay (AED)",
                    yaxis_title="Frequency"
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            # ANOVA
            st.markdown("#### One-Way ANOVA: Spending Across Clusters")
            
            cluster_groups = [df_filtered[df_filtered['Cluster'] == i]['Willingness_to_Pay_AED'].values 
                             for i in sorted(df_filtered['Cluster'].unique())]
            f_stat, p_value = stats.f_oneway(*cluster_groups)
            
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                st.metric("F-Statistic", f"{f_stat:.2f}")
            with col_a2:
                st.metric("P-Value", f"{p_value:.4f}")
            
            if p_value < 0.05:
                st.success("✅ **Significant differences** exist across clusters (p < 0.05)")
            else:
                st.info("ℹ️ No significant differences found (p ≥ 0.05)")
        
        with tab2:
            st.subheader("Statistical Significance Summary")
            
            st.markdown("""
            #### Understanding P-Values and Statistical Significance
            
            - **P-Value < 0.05:** Statistically significant result (reject null hypothesis)
            - **P-Value ≥ 0.05:** Not statistically significant (fail to reject null hypothesis)
            - **Confidence Level:** 95% (α = 0.05)
            
            #### Key Findings from Our Analysis:
            """)
            
            findings = []
            
            if chi2_result['p_value'] < 0.05:
                findings.append("✅ Income level significantly impacts adoption likelihood")
            else:
                findings.append("❌ No significant relationship between income and adoption")
            
            if ttest_result['p_value'] < 0.05:
                findings.append("✅ Adopters and non-adopters have significantly different spending patterns")
            else:
                findings.append("❌ No significant spending difference between groups")
            
            if p_value < 0.05:
                findings.append("✅ Customer clusters show significantly different spending behaviors")
            else:
                findings.append("❌ No significant spending differences across clusters")
            
            for finding in findings:
                st.markdown(f"- {finding}")

    # =========================================================================
    # PAGE 8: BUSINESS RECOMMENDATIONS
    # =========================================================================
    elif "Business Recommendations" in page:
        st.title("💡 AI-Generated Strategic Recommendations")
        
        st.markdown("""
        <div class='success-box'>
        <b>🎯 Data-Driven Insights:</b> Actionable strategies derived from comprehensive analysis of {0:,} customer profiles
        </div>
        """.format(len(df_filtered)), unsafe_allow_html=True)
        
        recommendations = generate_business_recommendations(df_filtered, models, metrics)
        
        # Priority sorting
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        recommendations.sort(key=lambda x: priority_order[x['priority']])
        
        st.header("🎯 Top Strategic Priorities")
        
        for idx, rec in enumerate(recommendations, 1):
            priority_colors = {
                'HIGH': '🔴',
                'MEDIUM': '🟡',
                'LOW': '🟢'
            }
            
            with st.expander(f"{priority_colors[rec['priority']]} **{idx}. {rec['title']}** [{rec['priority']} PRIORITY]", 
                           expanded=(rec['priority'] == 'HIGH')):
                st.markdown(f"**Recommendation:** {rec['detail']}")
                st.markdown(f"**Priority Level:** `{rec['priority']}`")
        
        st.markdown("---")
        
        # ROI Calculator
        st.header("💰 Business Case & Financial Projections")
        
        col_roi1, col_roi2 = st.columns(2)
        
        with col_roi1:
            st.subheader("📝 Input Assumptions")
            
            num_kiosks = st.number_input("Number of Kiosks to Deploy", 1, 50, 10)
            kiosk_cost = st.number_input("Cost per Kiosk (AED)", 10000, 100000, 50000, 5000)
            monthly_opex = st.number_input("Monthly Operating Cost/Kiosk (AED)", 1000, 10000, 3000, 500)
            customers_per_day = st.number_input("Expected Customers/Kiosk/Day", 10, 200, 50, 10)
            
            adoption_assumption = st.slider("Assumed Adoption Rate (%)", 30, 90, int(adoption_rate), 5)
            avg_transaction = st.number_input("Avg. Transaction (AED)", 20, 200, int(avg_wtp), 10)
            gross_margin_pct = st.slider("Gross Margin (%)", 20, 60, 40, 5)
        
        with col_roi2:
            st.subheader("📊 Financial Projections (Year 1)")
            
            # Calculations
            initial_investment = num_kiosks * kiosk_cost
            annual_opex = num_kiosks * monthly_opex * 12
            
            daily_transactions = num_kiosks * customers_per_day * (adoption_assumption / 100)
            annual_revenue = daily_transactions * avg_transaction * 365
            gross_profit = annual_revenue * (gross_margin_pct / 100)
            net_profit = gross_profit - annual_opex
            
            payback_period = initial_investment / net_profit if net_profit > 0 else float('inf')
            roi = ((net_profit - initial_investment) / initial_investment * 100) if initial_investment > 0 else 0
            
            st.metric("Initial Investment", f"AED {initial_investment:,.0f}")
            st.metric("Annual Revenue", f"AED {annual_revenue:,.0f}")
            st.metric("Gross Profit", f"AED {gross_profit:,.0f}")
            st.metric("Net Profit (Year 1)", f"AED {net_profit:,.0f}", 
                     delta_color="normal" if net_profit > 0 else "inverse")
            
            if payback_period != float('inf'):
                st.metric("Payback Period", f"{payback_period:.1f} years")
            else:
                st.metric("Payback Period", "N/A (Negative profit)")
            
            st.metric("ROI", f"{roi:.1f}%")
        
        # Sensitivity Analysis
        st.subheader("📊 Sensitivity Analysis: Impact of Adoption Rate")
        
        adoption_rates = np.arange(30, 91, 10)
        revenues = []
        profits = []
        
        for rate in adoption_rates:
            daily_trans = num_kiosks * customers_per_day * (rate / 100)
            annual_rev = daily_trans * avg_transaction * 365
            gross_prof = annual_rev * (gross_margin_pct / 100)
            net_prof = gross_prof - annual_opex
            
            revenues.append(annual_rev)
            profits.append(net_prof)
        
        fig_sensitivity = go.Figure()
        
        fig_sensitivity.add_trace(go.Scatter(
            x=adoption_rates, y=revenues,
            mode='lines+markers',
            name='Annual Revenue',
            line=dict(color='#2E8B57', width=3),
            marker=dict(size=10)
        ))
        
        fig_sensitivity.add_trace(go.Scatter(
            x=adoption_rates, y=profits,
            mode='lines+markers',
            name='Net Profit',
            line=dict(color='#228B22', width=3),
            marker=dict(size=10)
        ))
        
        fig_sensitivity.add_hline(y=0, line_dash="dash", line_color="red", 
                                 annotation_text="Break-even Line")
        
        fig_sensitivity.update_layout(
            title="Financial Sensitivity to Adoption Rate",
            xaxis_title="Adoption Rate (%)",
            yaxis_title="Amount (AED)",
            hovermode='x',
            height=400
        )
        
        st.plotly_chart(fig_sensitivity, use_container_width=True)
        
        # Implementation Roadmap
        st.markdown("---")
        st.header("🗺️ Strategic Implementation Roadmap")
        
        phases = [
            {
                'phase': 'Phase 1: Pilot Launch (Months 1-3)',
                'objectives': [
                    '🎯 Deploy 2-3 kiosks in high-traffic locations (Dubai Mall, Marina Mall)',
                    '📊 Test product mix, pricing, and UX with real customers',
                    '🔍 Gather feedback through surveys and app analytics',
                    '💰 Target: Break-even on pilot kiosks',
                    '📱 Launch MVP mobile app with basic features'
                ],
                'kpis': 'Daily transactions, Customer satisfaction, Product mix optimization'
            },
            {
                'phase': 'Phase 2: Market Validation (Months 4-6)',
                'objectives': [
                    '📈 Analyze pilot data and refine business model',
                    '🤝 Establish partnerships with 2-3 major retailers',
                    '💳 Integrate advanced payment options (Apple Pay, Google Pay)',
                    '🎁 Launch loyalty program based on cluster insights',
                    '📣 Begin targeted marketing campaigns'
                ],
                'kpis': 'Retention rate, Repeat purchase rate, Customer acquisition cost'
            },
            {
                'phase': 'Phase 3: Expansion (Months 7-12)',
                'objectives': [
                    '🚀 Scale to 15-20 kiosks across Dubai & Abu Dhabi',
                    '🛒 Implement product bundling based on basket analysis',
                    '🌱 Launch sustainability tracking feature in app',
                    '💰 Target: 30% of kiosks achieving profitability',
                    '🎯 Focus on high-value customer segments (Clusters 0 & 2)'
                ],
                'kpis': 'Revenue per kiosk, Market share, Brand awareness'
            },
            {
                'phase': 'Phase 4: Optimization & Scale (Year 2+)',
                'objectives': [
                    '📊 Data-driven product mix optimization',
                    '🌍 Expand to Sharjah and Northern Emirates',
                    '🏢 Explore B2B partnerships (corporate offices, hotels)',
                    '🌐 Consider regional expansion (Saudi Arabia, Qatar)',
                    '💼 Explore franchise opportunities'
                ],
                'kpis': 'Profitability, Market penetration, Customer lifetime value'
            }
        ]
        
        for phase in phases:
            with st.expander(f"**{phase['phase']}**"):
                st.markdown("**Key Objectives:**")
                for obj in phase['objectives']:
                    st.markdown(f"- {obj}")
                st.markdown(f"**Success Metrics:** {phase['kpis']}")

    # =========================================================================
    # PAGE 9: EXPORT & DOWNLOAD
    # =========================================================================
    elif "Export & Download" in page:
        st.title("📥 Export & Download Center")
        
        st.markdown("""
        <div class='success-box'>
        <b>📦 Complete Package:</b> Download all data, reports, predictions, and deployment code
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📊 Data Exports", "📄 Reports", "💻 Code Templates"])
        
        with tab1:
            st.subheader("📥 Downloadable Datasets")
            
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                st.markdown("#### 🗂️ Complete Dataset with Clusters")
                st.download_button(
                    label="📥 Download Full Dataset (CSV)",
                    data=df.to_csv(index=False),
                    file_name="refillhub_complete_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                st.markdown("#### 📊 Cluster Profiles")
                cluster_profiles = models['cluster_profiles']
                st.download_button(
                    label="📥 Download Cluster Profiles (CSV)",
                    data=cluster_profiles.to_csv(),
                    file_name="cluster_profiles.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                st.markdown("#### 🔗 Association Rules")
                if not models['association_rules'].empty:
                    st.download_button(
                        label="📥 Download Association Rules (CSV)",
                        data=models['association_rules'].to_csv(index=False),
                        file_name="association_rules.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col_d2:
                st.markdown("#### 🎯 Classification Feature Importance")
                if 'feature_importance_clf' in models:
                    st.download_button(
                        label="📥 Download Feature Importance (CSV)",
                        data=models['feature_importance_clf'].to_csv(index=False),
                        file_name="feature_importance_classification.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                st.markdown("#### 🔍 Filtered Dataset")
                st.download_button(
                    label="📥 Download Filtered Data (CSV)",
                    data=df_filtered.to_csv(index=False),
                    file_name="refillhub_filtered_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                st.markdown("#### 📈 Model Performance Metrics")
                metrics_export = pd.DataFrame(metrics['classification']).T
                st.download_button(
                    label="📥 Download Model Metrics (CSV)",
                    data=metrics_export.to_csv(),
                    file_name="model_performance_metrics.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with tab2:
            st.subheader("📄 Comprehensive Reports")
            
            # Executive Summary Report
            st.markdown("#### 📊 Executive Summary Report")
            
            summary_report = f"""
# ReFill Hub: Executive Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Business Overview
- **Market Size:** {len(df):,} survey respondents
- **Target Market:** UAE residents (7 Emirates)
- **Business Model:** Automated refill kiosks for household products

## Key Performance Indicators
- **Adoption Rate:** {adoption_rate:.1f}%
- **Average Willingness to Pay:** AED {avg_wtp:.2f}
- **High-Value Customer Segment:** {(df['Willingness_to_Pay_AED'] > df['Willingness_to_Pay_AED'].quantile(0.75)).sum():,} customers
- **Market Awareness (Plastic Ban):** {(df['Aware_Plastic_Ban'] == 'Yes').mean()*100:.1f}%

## Customer Segmentation
{models['cluster_sizes'].to_string()}

## Model Performance
- **Best Classification Model:** {max(metrics['classification'], key=lambda x: metrics['classification'][x]['Accuracy'])}
  - Accuracy: {max(m['Accuracy'] for m in metrics['classification'].values()):.2%}
  - Precision: {max(m['Precision'] for m in metrics['classification'].values()):.2%}
  - Recall: {max(m['Recall'] for m in metrics['classification'].values()):.2%}

- **Best Regression Model:** {max(metrics['regression'], key=lambda x: metrics['regression'][x]['R2'])}
  - R² Score: {max(m['R2'] for m in metrics['regression'].values()):.4f}
  - RMSE: AED {min(m['RMSE'] for m in metrics['regression'].values()):.2f}

## Strategic Recommendations
1. Target Cluster 0 (Eco-Warriors) with sustainability messaging
2. Deploy first kiosks in Dubai Mall and high-footfall supermarkets
3. Implement mobile-first payment experience
4. Create product bundles based on association rules
5. Set competitive pricing around AED {df['Willingness_to_Pay_AED'].median():.2f}

## Market Opportunity
- **Addressable Market:** {(df['Likely_to_Use_ReFillHub'] == 'Yes').sum():,} potential adopters
- **Estimated Year 1 Revenue:** AED {(df[df['Likely_to_Use_ReFillHub'] == 'Yes']['Willingness_to_Pay_AED'].sum() * 12):,.2f} (assuming monthly purchases)
"""
            
            st.download_button(
                label="📥 Download Executive Summary (TXT)",
                data=summary_report,
                file_name="executive_summary_report.txt",
                mime="text/plain",
                use_container_width=True
            )
            
            # Technical Report
            st.markdown("#### 🔬 Technical Model Performance Report")
            
            tech_report = f"""
# ReFill Hub: Technical Model Performance Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Classification Models Comparison

"""
            
            for model_name, model_metrics in metrics['classification'].items():
                tech_report += f"""
### {model_name}
- Accuracy: {model_metrics['Accuracy']:.4f}
- Precision: {model_metrics['Precision']:.4f}
- Recall: {model_metrics['Recall']:.4f}
- F1-Score: {model_metrics['F1 Score']:.4f}
"""
                if 'ROC_AUC' in model_metrics:
                    tech_report += f"- ROC-AUC: {model_metrics['ROC_AUC']:.4f}\n"
                tech_report += "\n"
            
            tech_report += """
## Regression Models Comparison

"""
            
            for model_name, model_metrics in metrics['regression'].items():
                tech_report += f"""
### {model_name}
- R² Score: {model_metrics['R2']:.4f}
- RMSE: {model_metrics['RMSE']:.2f} AED
- MAE: {model_metrics['MAE']:.2f} AED

"""
            
            tech_report += f"""
## Clustering Analysis
- Algorithm: K-Means (k=4)
- Silhouette Score: {metrics['silhouette_score']:.4f}
- Features: {', '.join(cluster_features)}

## Association Rules
- Total Rules Generated: {len(models['association_rules'])}
- Average Lift: {models['association_rules']['lift'].mean():.2f if not models['association_rules'].empty else 'N/A'}
- Max Lift: {models['association_rules']['lift'].max():.2f if not models['association_rules'].empty else 'N/A'}

## Methodology
- Train/Test Split: 80/20
- Cross-Validation: 5-fold
- Feature Engineering: Family size conversion, one-hot encoding for categoricals
- Scaling: StandardScaler for numerical features
"""
            
            st.download_button(
                label="📥 Download Technical Report (TXT)",
                data=tech_report,
                file_name="technical_model_report.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with tab3:
            st.subheader("💻 Production Deployment Code")
            
            # Python Prediction Template
            st.markdown("#### 🐍 Python: Prediction Script")
            
            python_code = '''
"""
ReFill Hub: Customer Prediction Script
Use this script to make predictions on new customer data
"""

import pandas as pd
import pickle
import numpy as np

# Load the trained model
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Prepare customer data
def prepare_input(customer_data):
    """
    customer_data: dict with keys matching feature names
    """
    df = pd.DataFrame([customer_data])
    
    # Feature engineering (match training pipeline)
    if 'Family_Size' in df.columns:
        def process_family(val):
            if '5+' in str(val): return 5.0
            if '1-2' in str(val): return 1.5
            if '3-4' in str(val): return 3.5
            return 3.0
        df['Family_Size_Num'] = df['Family_Size'].apply(process_family)
    
    return df

# Make prediction
def predict_customer(model, customer_data):
    input_df = prepare_input(customer_data)
    
    # Classification
    adoption_pred = model.predict(input_df)[0]
    adoption_prob = model.predict_proba(input_df)[0]
    
    return {
        'will_adopt': bool(adoption_pred),
        'adoption_probability': float(adoption_prob[1]),
        'confidence': f"{adoption_prob[1]*100:.1f}%"
    }

# Example usage
if __name__ == '__main__':
    # Load model
    classifier = load_model('refillhub_classifier.pkl')
    
    # New customer profile
    new_customer = {
        'Age_Group': '25-34',
        'Gender': 'Female',
        'Income': '10000-15000',
        'Emirate': 'Dubai',
        'Importance_Price': 3,
        'Importance_Sustainability': 4,
        'Importance_Convenience': 4,
        'Family_Size': '3-4',
        # ... add all other required features
    }
    
    # Predict
    result = predict_customer(classifier, new_customer)
    print(f"Prediction: {result}")
'''
            
            st.code(python_code, language='python')
            st.download_button(
                label="📥 Download Python Template",
                data=python_code,
                file_name="refillhub_prediction.py",
                mime="text/plain",
                use_container_width=True
            )
            
            # Flask API Template
            st.markdown("#### 🌐 Flask REST API Template")
            
            flask_code = '''
"""
ReFill Hub: REST API for Production Deployment
Run: python api.py
Endpoint: http://localhost:5000/predict
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Load models at startup
classifier = pickle.load(open('models/refillhub_classifier.pkl', 'rb'))
regressor = pickle.load(open('models/refillhub_regressor.pkl', 'rb'))

def prepare_features(data):
    """Prepare features to match training pipeline"""
    df = pd.DataFrame([data])
    
    # Feature engineering
    if 'Family_Size' in df.columns:
        def process_family(val):
            if '5+' in str(val): return 5.0
            if '1-2' in str(val): return 1.5
            if '3-4' in str(val): return 3.5
            return 3.0
        df['Family_Size_Num'] = df['Family_Size'].apply(process_family)
    
    return df

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'version': '1.0'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get request data
        data = request.json
        
        # Prepare features
        features = prepare_features(data)
        
        # Make predictions
        adoption_pred = classifier.predict(features)[0]
        adoption_prob = classifier.predict_proba(features)[0][1]
        spending_pred = regressor.predict(features)[0]
        
        # Prepare response
        response = {
            'success': True,
            'predictions': {
                'will_adopt': bool(adoption_pred),
                'adoption_probability': float(adoption_prob),
                'predicted_spending': float(spending_pred),
                'customer_segment': determine_segment(data),
                'recommendation': generate_recommendation(adoption_prob, spending_pred)
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

def determine_segment(data):
    """Determine customer segment based on attributes"""
    if data.get('Importance_Sustainability', 0) >= 4:
        return 'Eco-Warrior'
    elif data.get('Importance_Price', 0) >= 4:
        return 'Value Seeker'
    elif data.get('Importance_Convenience', 0) >= 4:
        return 'Convenience Seeker'
    else:
        return 'Skeptic'

def generate_recommendation(prob, spend):
    """Generate personalized recommendation"""
    if prob > 0.7 and spend > 100:
        return 'High-value prospect: Offer premium subscription'
    elif prob > 0.5:
        return 'Good prospect: Target with introductory offer'
    else:
        return 'Low probability: Require strong incentives'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
'''
            
            st.code(flask_code, language='python')
            st.download_button(
                label="📥 Download Flask API",
                data=flask_code,
                file_name="refillhub_api.py",
                mime="text/plain",
                use_container_width=True
            )
            
            # SQL Queries
            st.markdown("#### 🗄️ SQL Database Schema & Queries")
            
            sql_code = f'''
-- ReFill Hub Database Schema

-- Customer Profiles Table
CREATE TABLE customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    age_group VARCHAR(20),
    gender VARCHAR(20),
    emirate VARCHAR(50),
    income_level VARCHAR(30),
    family_size VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Predictions Table
CREATE TABLE predictions (
    prediction_id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id VARCHAR(50),
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    adoption_probability DECIMAL(5, 4),
    predicted_spending DECIMAL(10, 2),
    customer_segment VARCHAR(50),
    model_version VARCHAR(20),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Transactions Table
CREATE TABLE transactions (
    transaction_id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id VARCHAR(50),
    transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    amount DECIMAL(10, 2),
    products TEXT,
    kiosk_location VARCHAR(100),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Sample Queries

-- 1. Get high-value prospects
SELECT 
    c.customer_id,
    c.emirate,
    c.income_level,
    p.adoption_probability,
    p.predicted_spending,
    p.customer_segment
FROM customers c
JOIN predictions p ON c.customer_id = p.customer_id
WHERE p.adoption_probability > 0.7
    AND p.predicted_spending > {df['Willingness_to_Pay_AED'].median():.2f}
ORDER BY p.predicted_spending DESC
LIMIT 100;

-- 2. Customer segmentation distribution
SELECT 
    customer_segment,
    COUNT(*) as segment_size,
    AVG(adoption_probability) as avg_adoption,
    AVG(predicted_spending) as avg_spending
FROM predictions
WHERE prediction_date >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY customer_segment
ORDER BY avg_spending DESC;

-- 3. Revenue forecast by emirate
SELECT 
    c.emirate,
    COUNT(DISTINCT c.customer_id) as potential_customers,
    SUM(p.predicted_spending) as estimated_monthly_revenue,
    AVG(p.adoption_probability) as avg_adoption_rate
FROM customers c
JOIN predictions p ON c.customer_id = p.customer_id
WHERE p.adoption_probability > 0.5
GROUP BY c.emirate
ORDER BY estimated_monthly_revenue DESC;

-- 4. Track prediction accuracy
SELECT 
    p.model_version,
    COUNT(*) as predictions_made,
    AVG(CASE 
        WHEN t.amount IS NOT NULL AND p.adoption_probability > 0.5 THEN 1
        WHEN t.amount IS NULL AND p.adoption_probability <= 0.5 THEN 1
        ELSE 0
    END) as accuracy
FROM predictions p
LEFT JOIN transactions t ON p.customer_id = t.customer_id
    AND t.transaction_date > p.prediction_date
WHERE p.prediction_date >= DATE_SUB(NOW(), INTERVAL 90 DAY)
GROUP BY p.model_version;
'''
            
            st.code(sql_code, language='sql')
            st.download_button(
                label="📥 Download SQL Queries",
                data=sql_code,
                file_name="database_queries.sql",
                mime="text/plain",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
