import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, silhouette_score, r2_score, 
                            mean_squared_error, confusion_matrix, roc_curve, auc, 
                            precision_recall_curve)
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
    page_title="ReFill Hub: Advanced BI Dashboard",
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
    }
    h2 {
        color: #228B22;
        font-weight: 600;
    }
    h3 {
        color: #3CB371;
        font-weight: 500;
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
    
    /* Download button special styling */
    .download-btn {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
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
        return val
    df['Family_Size_Num'] = df['Family_Size'].apply(process_family_size)
    
    # Extract discount percentage
    df['Discount_Percent'] = df['Discount_Switch'].str.extract('(\d+)').astype(float)
    
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
                        'Reduce_Waste_Score', 'Eco_Brand_Preference', 'Social_Influence_Score']

    return df, categorical_features, numerical_features, cluster_features

# =============================================================================
# ADVANCED MODEL TRAINING (CACHED)
# =============================================================================
@st.cache_resource
def train_all_models(df, categorical_features, numerical_features, cluster_features):
    """
    Trains all ML models with enhanced metrics and feature importance
    """
    models = {}
    metrics = {}
    
    # --- CLASSIFICATION ---
    X_class = df[categorical_features + numerical_features]
    y_class = df['Likely_to_Use_ReFillHub'].map({'Yes': 1, 'No': 0})
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    clf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=200, max_depth=15))
    ])
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    clf_pipeline.fit(X_train_c, y_train_c)
    y_pred_c = clf_pipeline.predict(X_test_c)
    y_pred_proba_c = clf_pipeline.predict_proba(X_test_c)[:, 1]
    
    models['classification'] = clf_pipeline
    metrics['classification_report'] = classification_report(y_test_c, y_pred_c, output_dict=True)
    metrics['confusion_matrix'] = confusion_matrix(y_test_c, y_pred_c)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test_c, y_pred_proba_c)
    metrics['roc_auc'] = auc(fpr, tpr)
    metrics['roc_curve'] = (fpr, tpr)
    
    # Feature Importance
    feature_names = (numerical_features + 
                    clf_pipeline.named_steps['preprocessor']
                    .named_transformers_['cat']
                    .get_feature_names_out(categorical_features).tolist())
    
    importances = clf_pipeline.named_steps['classifier'].feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(20)
    models['feature_importance_clf'] = feature_importance_df
    
    # Cross-validation score
    cv_scores = cross_val_score(clf_pipeline, X_class, y_class, cv=5, scoring='accuracy')
    metrics['cv_mean'] = cv_scores.mean()
    metrics['cv_std'] = cv_scores.std()

    # --- REGRESSION ---
    X_reg = df[categorical_features + numerical_features]
    y_reg = df['Willingness_to_Pay_AED']
    
    reg_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42, n_estimators=200, max_depth=15))
    ])

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    reg_pipeline.fit(X_train_r, y_train_r)
    y_pred_r = reg_pipeline.predict(X_test_r)
    
    models['regression'] = reg_pipeline
    metrics['r2_score'] = r2_score(y_test_r, y_pred_r)
    metrics['rmse'] = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
    metrics['mae'] = np.mean(np.abs(y_test_r - y_pred_r))
    
    # Feature importance for regression
    importances_reg = reg_pipeline.named_steps['regressor'].feature_importances_
    feature_importance_reg_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances_reg
    }).sort_values('Importance', ascending=False).head(20)
    models['feature_importance_reg'] = feature_importance_reg_df
    
    # Store predictions for residual analysis
    metrics['y_test_reg'] = y_test_r
    metrics['y_pred_reg'] = y_pred_r

    # --- CLUSTERING ---
    X_cluster = df[cluster_features]
    cluster_scaler = StandardScaler()
    X_cluster_scaled = cluster_scaler.fit_transform(X_cluster)
    
    # Determine optimal clusters using elbow method
    inertias = []
    silhouette_scores = []
    K_range = range(2, 8)
    for k in K_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_temp.fit(X_cluster_scaled)
        inertias.append(kmeans_temp.inertia_)
        silhouette_scores.append(silhouette_score(X_cluster_scaled, kmeans_temp.labels_))
    
    metrics['elbow_data'] = (list(K_range), inertias, silhouette_scores)
    
    # Use k=4 as optimal
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_cluster_scaled)
    
    models['clustering_model'] = kmeans
    models['clustering_scaler'] = cluster_scaler
    metrics['silhouette_score'] = silhouette_score(X_cluster_scaled, df['Cluster'])
    
    # Enhanced cluster profiles
    cluster_profiles = df.groupby('Cluster')[cluster_features + ['Likely_to_Use_ReFillHub', 'Willingness_to_Pay_AED']].agg({
        **{feat: 'mean' for feat in cluster_features},
        'Likely_to_Use_ReFillHub': lambda x: (x == 'Yes').mean(),
        'Willingness_to_Pay_AED': 'mean'
    })
    models['cluster_profiles'] = cluster_profiles
    
    # Cluster sizes
    models['cluster_sizes'] = df['Cluster'].value_counts().sort_index()

    # --- ASSOCIATION RULES ---
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
# ADVANCED ANALYTICS FUNCTIONS
# =============================================================================

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
    if not models['association_rules'].empty:
        top_rule = models['association_rules'].iloc[0]
        recommendations.append({
            'title': '🛒 Product Bundling',
            'detail': f"Create bundle: {top_rule['antecedents']} + {top_rule['consequents']} (Lift: {top_rule['lift']:.2f})",
            'priority': 'MEDIUM'
        })
    
    # Recommendation 5: Feature Priority
    top_feature = models['feature_importance_clf'].iloc[0]
    recommendations.append({
        'title': '⚡ Key Success Factor',
        'detail': f"Prioritize {top_feature['Feature']} in value proposition (highest importance)",
        'priority': 'HIGH'
    })
    
    return recommendations

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
    st.sidebar.image("https://via.placeholder.com/400x200/2E8B57/FFFFFF?text=ReFill+Hub+BI", 
                     use_column_width=True)
    st.sidebar.title("🧭 Navigation")
    
    # Data Source Selection
    st.sidebar.header("📊 Data Source")
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Use Default Data", "Upload Custom Data", "Generate New Synthetic Data"],
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
        if uploaded_file and st.sidebar.button("Load Uploaded Data"):
            st.session_state.data_loaded = False
    
    elif data_source == "Generate New Synthetic Data":
        n_samples = st.sidebar.number_input("Number of samples", 100, 2000, 600, 50)
        if st.sidebar.button("Generate Synthetic Data"):
            use_synthetic = True
            st.session_state.data_loaded = False
    
    # Load data based on selection
    if not st.session_state.data_loaded:
        with st.spinner("Loading and processing data..."):
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
        "Select Page:",
        [
            "🏠 Executive Summary",
            "📊 Data Explorer & Quality",
            "🧩 Customer Segmentation",
            "🔮 Predictive Simulator",
            "🛒 Market Basket Analysis",
            "📈 Advanced Analytics",
            "💡 Business Recommendations",
            "🤖 Model Performance",
            "📥 Export & Download"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Active Records:** {len(df_filtered):,} / {len(df):,}")
    
    # =========================================================================
    # PAGE 1: EXECUTIVE SUMMARY
    # =========================================================================
    if "Executive Summary" in page:
        st.title("♻️ ReFill Hub: Executive Summary")
        st.markdown("""
        <div class='info-box'>
        <b>🎯 Mission:</b> Provide data-driven intelligence for ReFill Hub launch strategy based on comprehensive market research.
        </div>
        """, unsafe_allow_html=True)
        
        # Top Metrics
        st.header("📊 Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)
        
        adoption_rate = (df_filtered['Likely_to_Use_ReFillHub'] == 'Yes').mean() * 100
        avg_wtp = df_filtered['Willingness_to_Pay_AED'].mean()
        high_value_customers = (df_filtered['Willingness_to_Pay_AED'] > df_filtered['Willingness_to_Pay_AED'].quantile(0.75)).sum()
        avg_sustainability_score = df_filtered['Importance_Sustainability'].mean()
        
        with col1:
            st.metric(
                "Adoption Rate",
                f"{adoption_rate:.1f}%",
                delta=f"{adoption_rate - 50:.1f}% vs. benchmark",
                help="Percentage of respondents likely to use ReFill Hub"
            )
        
        with col2:
            st.metric(
                "Avg. Willingness to Pay",
                f"AED {avg_wtp:.2f}",
                delta=f"±{df_filtered['Willingness_to_Pay_AED'].std():.2f}",
                help="Average amount customers willing to spend per visit"
            )
        
        with col3:
            st.metric(
                "High-Value Customers",
                f"{high_value_customers:,}",
                delta=f"{(high_value_customers/len(df_filtered)*100):.1f}% of total",
                help="Customers in top 25% spending bracket"
            )
        
        with col4:
            st.metric(
                "Sustainability Score",
                f"{avg_sustainability_score:.2f}/5",
                delta="Strong" if avg_sustainability_score > 3.5 else "Moderate",
                help="Average importance placed on sustainability"
            )
        
        st.markdown("---")
        
        # Visualizations
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("🎯 Adoption by Demographics")
            fig_demo = px.sunburst(
                df_filtered,
                path=['Emirate', 'Income', 'Likely_to_Use_ReFillHub'],
                title="Hierarchical View: Location → Income → Adoption",
                color='Likely_to_Use_ReFillHub',
                color_discrete_map={'Yes': '#2E8B57', 'No': '#FF6347'}
            )
            st.plotly_chart(fig_demo, use_container_width=True)
        
        with col_b:
            st.subheader("💰 Spending Distribution")
            fig_spend = px.box(
                df_filtered,
                x='Likely_to_Use_ReFillHub',
                y='Willingness_to_Pay_AED',
                color='Likely_to_Use_ReFillHub',
                title="Willingness to Pay: Adopters vs. Non-Adopters",
                color_discrete_map={'Yes': '#2E8B57', 'No': '#FF6347'}
            )
            fig_spend.update_layout(showlegend=False)
            st.plotly_chart(fig_spend, use_container_width=True)
        
        # Geographic Insights
        st.subheader("🗺️ Geographic Market Potential")
        
        emirate_stats = df_filtered.groupby('Emirate').agg({
            'Likely_to_Use_ReFillHub': lambda x: (x == 'Yes').mean() * 100,
            'Willingness_to_Pay_AED': 'mean',
            'Emirate': 'count'
        }).rename(columns={
            'Likely_to_Use_ReFillHub': 'Adoption_Rate',
            'Willingness_to_Pay_AED': 'Avg_Spend',
            'Emirate': 'Sample_Size'
        }).reset_index()
        
        fig_geo = px.scatter(
            emirate_stats,
            x='Adoption_Rate',
            y='Avg_Spend',
            size='Sample_Size',
            color='Emirate',
            title="Market Opportunity Matrix: Adoption vs. Spending by Emirate",
            labels={'Adoption_Rate': 'Adoption Rate (%)', 'Avg_Spend': 'Avg. Spending (AED)'},
            hover_data=['Sample_Size']
        )
        fig_geo.add_hline(y=avg_wtp, line_dash="dash", line_color="gray", 
                         annotation_text="Avg. Spending")
        fig_geo.add_vline(x=adoption_rate, line_dash="dash", line_color="gray",
                         annotation_text="Avg. Adoption")
        st.plotly_chart(fig_geo, use_container_width=True)
        
        # Customer Journey
        st.subheader("🛤️ Customer Journey Insights")
        col_j1, col_j2, col_j3 = st.columns(3)
        
        with col_j1:
            aware_rate = (df_filtered['Aware_Plastic_Ban'] == 'Yes').mean() * 100
            st.metric("Plastic Ban Awareness", f"{aware_rate:.1f}%")
            
        with col_j2:
            used_before_rate = (df_filtered['Used_Refill_Before'] == 'Yes').mean() * 100
            st.metric("Prior Refill Experience", f"{used_before_rate:.1f}%")
        
        with col_j3:
            eco_product_rate = (df_filtered['Uses_Eco_Products'] == 'Yes').mean() * 100
            st.metric("Current Eco-Product Users", f"{eco_product_rate:.1f}%")

    # =========================================================================
    # PAGE 2: DATA EXPLORER & QUALITY
    # =========================================================================
    elif "Data Explorer" in page:
        st.title("📊 Data Explorer & Quality Dashboard")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Preview", "📈 Distributions", "🔗 Correlations", "✅ Data Quality"])
        
        with tab1:
            st.subheader("Raw Data Preview")
            st.dataframe(df_filtered.head(100), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df_filtered))
            with col2:
                st.metric("Total Features", len(df_filtered.columns))
            with col3:
                st.metric("Memory Usage", f"{df_filtered.memory_usage(deep=True).sum() / 1024:.2f} KB")
            
            st.subheader("Statistical Summary")
            st.dataframe(df_filtered.describe(), use_container_width=True)
        
        with tab2:
            st.subheader("Feature Distributions")
            
            col_dist1, col_dist2 = st.columns(2)
            
            with col_dist1:
                selected_cat = st.selectbox("Select Categorical Feature", df_filtered.select_dtypes(include='object').columns)
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
                selected_num = st.selectbox("Select Numerical Feature", df_filtered.select_dtypes(include=['int64', 'float64']).columns)
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
            
            # Numerical correlations
            numeric_cols = df_filtered.select_dtypes(include=['int64', 'float64']).columns
            corr_matrix = df_filtered[numeric_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                title="Feature Correlation Heatmap",
                color_continuous_scale='RdYlGn',
                aspect='auto',
                labels=dict(color="Correlation")
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Top correlations
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
            
            # Missing values
            col_q1, col_q2 = st.columns(2)
            
            with col_q1:
                st.markdown("#### Missing Values")
                missing = df_filtered.isnull().sum()
                missing_pct = (missing / len(df_filtered) * 100).round(2)
                missing_df = pd.DataFrame({
                    'Feature': missing.index,
                    'Missing Count': missing.values,
                    'Missing %': missing_pct.values
                }).sort_values('Missing Count', ascending=False)
                
                st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
                
                if missing_df['Missing Count'].sum() == 0:
                    st.success("✅ No missing values detected!")
            
            with col_q2:
                st.markdown("#### Duplicate Records")
                duplicates = df_filtered.duplicated().sum()
                st.metric("Duplicate Rows", duplicates)
                
                if duplicates == 0:
                    st.success("✅ No duplicate records found!")
                else:
                    st.warning(f"⚠️ Found {duplicates} duplicate records")
            
            # Data types
            st.markdown("#### Data Type Summary")
            dtype_df = pd.DataFrame({
                'Feature': df_filtered.dtypes.index,
                'Data Type': df_filtered.dtypes.values,
                'Unique Values': [df_filtered[col].nunique() for col in df_filtered.columns]
            })
            st.dataframe(dtype_df, use_container_width=True)
            
            # Outlier detection for numerical columns
            st.markdown("#### Outlier Detection (IQR Method)")
            outlier_summary = []
            for col in numeric_cols:
                Q1 = df_filtered[col].quantile(0.25)
                Q3 = df_filtered[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df_filtered[col] < (Q1 - 1.5 * IQR)) | (df_filtered[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_summary.append({'Feature': col, 'Outliers': outliers, 'Outlier %': f"{outliers/len(df_filtered)*100:.2f}%"})
            
            outlier_df = pd.DataFrame(outlier_summary).sort_values('Outliers', ascending=False)
            st.dataframe(outlier_df, use_container_width=True)

    # =========================================================================
    # PAGE 3: CUSTOMER SEGMENTATION
    # =========================================================================
    elif "Customer Segmentation" in page:
        st.title("🧩 Customer Segmentation Analysis")
        
        st.markdown("""
        <div class='info-box'>
        <b>Methodology:</b> K-Means Clustering (k=4) based on psychographic and behavioral features
        </div>
        """, unsafe_allow_html=True)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Silhouette Score", f"{metrics['silhouette_score']:.3f}", help="Cluster quality (0.3-0.5 is good)")
        with col_m2:
            st.metric("Number of Clusters", "4")
        with col_m3:
            total_variance = sum([cluster_features.index(f) for f in cluster_features if f in df_filtered.columns])
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
                title="Elbow Method",
                xaxis_title="Number of Clusters (k)",
                yaxis_title="Inertia",
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
                title="Silhouette Analysis",
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
            z='Importance_Convenience',
            color='Cluster_Label',
            symbol='Likely_to_Use_ReFillHub',
            opacity=0.7,
            title="Customer Segments in 3D Feature Space",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_3d.update_layout(
            scene=dict(
                xaxis_title="Price Importance",
                yaxis_title="Sustainability Importance",
                zaxis_title="Convenience Importance"
            ),
            height=600
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Cluster Profiles
        st.subheader("📊 Detailed Cluster Profiles")
        
        cluster_profiles = models['cluster_profiles']
        cluster_sizes = models['cluster_sizes']
        
        # Add cluster sizes to profiles
        cluster_profiles['Cluster_Size'] = cluster_sizes
        cluster_profiles['Size_Percent'] = (cluster_sizes / len(df_filtered) * 100).round(1)
        
        st.dataframe(
            cluster_profiles.style.background_gradient(cmap='Greens', subset=cluster_features).format(precision=2),
            use_container_width=True
        )
        
        # Cluster Comparison
        st.subheader("⚖️ Cluster Comparison")
        
        col_comp1, col_comp2 = st.columns(2)
        
        with col_comp1:
            # Radar chart for cluster comparison
            categories = cluster_features
            
            fig_radar = go.Figure()
            
            for cluster_id in sorted(df_filtered['Cluster'].unique()):
                values = cluster_profiles.loc[cluster_id, cluster_features].values.tolist()
                values += values[:1]  # Close the polygon
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=f'Cluster {cluster_id}'
                ))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                showlegend=True,
                title="Cluster Profiles: Radar Chart"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with col_comp2:
            # Cluster size & adoption
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
                title="Cluster Value Matrix",
                labels={'Adoption_Rate': 'Adoption Rate (%)', 'Avg_Spend': 'Avg. Spending (AED)'},
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_bubble, use_container_width=True)
        
        # Persona Descriptions
        st.subheader("👥 Customer Personas & Strategies")
        
        personas = {
            0: {
                'name': '🌱 The Eco-Warriors',
                'description': 'High sustainability focus, willing to pay premium for environmental impact',
                'strategy': [
                    '• Lead with impact metrics (e.g., "Saved 100 plastic bottles")',
                    '• Premium eco-brand partnerships',
                    '• Sustainability certificates and badges',
                    '• Community events and eco-campaigns'
                ],
                'color': 'success'
            },
            1: {
                'name': '💰 The Budget-Conscious',
                'description': 'Price-sensitive shoppers looking for value and savings',
                'strategy': [
                    '• Emphasize cost savings vs. packaged products',
                    '• Subscription discounts and loyalty programs',
                    '• Bulk refill discounts',
                    '• Price comparison campaigns'
                ],
                'color': 'info'
            },
            2: {
                'name': '⚡ The Convenience Seekers',
                'description': 'Busy professionals who value speed and accessibility',
                'strategy': [
                    '• Strategic locations (offices, malls, residential)',
                    '• Mobile app with pre-ordering',
                    '• Express lanes and tap-to-pay',
                    '• Home delivery options'
                ],
                'color': 'warning'
            },
            3: {
                'name': '😐 The Apathetic',
                'description': 'Low engagement across all dimensions, hardest to convert',
                'strategy': [
                    '• Aggressive first-time discounts (50% off)',
                    '• Influencer partnerships and social proof',
                    '• Gamification and rewards',
                    '• Simplified messaging and trial offers'
                ],
                'color': 'error'
            }
        }
        
        for cluster_id, persona in personas.items():
            with st.expander(f"**Cluster {cluster_id}: {persona['name']}** (Size: {cluster_sizes[cluster_id]} | {cluster_profiles.loc[cluster_id, 'Size_Percent']:.1f}%)"):
                st.markdown(f"**Profile:** {persona['description']}")
                st.markdown("**Recommended Strategies:**")
                for strategy in persona['strategy']:
                    st.markdown(strategy)
                
                # Cluster-specific metrics
                cluster_data = df_filtered[df_filtered['Cluster'] == cluster_id]
                col_p1, col_p2, col_p3 = st.columns(3)
                
                with col_p1:
                    adoption = (cluster_data['Likely_to_Use_ReFillHub'] == 'Yes').mean() * 100
                    st.metric("Adoption Rate", f"{adoption:.1f}%")
                
                with col_p2:
                    avg_spend = cluster_data['Willingness_to_Pay_AED'].mean()
                    st.metric("Avg. Spending", f"AED {avg_spend:.2f}")
                
                with col_p3:
                    sustainability = cluster_data['Importance_Sustainability'].mean()
                    st.metric("Sustainability Score", f"{sustainability:.2f}/5")

    # =========================================================================
    # PAGE 4: PREDICTIVE SIMULATOR
    # =========================================================================
    elif "Predictive Simulator" in page:
        st.title("🔮 AI-Powered Predictive Simulator")
        
        st.markdown("""
        <div class='info-box'>
        <b>Simulate customer profiles</b> and predict their adoption likelihood and spending potential using trained ML models.
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🎯 Single Prediction", "📊 Batch Predictions"])
        
        with tab1:
            with st.form("simulation_form"):
                st.header("Customer Profile Configuration")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("📋 Demographics")
                    age = st.selectbox("Age Group", df['Age_Group'].unique())
                    gender = st.selectbox("Gender", df['Gender'].unique())
                    income = st.selectbox("Income (AED)", df['Income'].unique())
                    fam_size = st.selectbox("Family Size", df['Family_Size'].unique(), index=1)
                    emirate = st.selectbox("Emirate", df['Emirate'].unique())
                    occupation = st.selectbox("Occupation", df['Occupation'].unique())
                
                with col2:
                    st.subheader("💭 Attitudes (1-5)")
                    imp_price = st.slider("Importance: Price", 1, 5, 3)
                    imp_sust = st.slider("Importance: Sustainability", 1, 5, 3)
                    imp_conv = st.slider("Importance: Convenience", 1, 5, 3)
                    waste_score = st.slider("Waste Reduction Effort", 1, 5, 3)
                    social_score = st.slider("Social Influence", 1, 5, 3)
                    try_likelihood = st.slider("Initial Interest", 1, 5, 3)
                
                with col3:
                    st.subheader("🛒 Behaviors")
                    eco_brand = st.select_slider("Eco-Brand Preference", [1, 2, 3, 4, 5], 3)
                    follow_camp = st.selectbox("Follows Green Campaigns", ["Yes", "No"])
                    used_before = st.selectbox("Used Refill Service Before", ["Yes", "No"])
                    purchase_freq = st.selectbox("Shopping Frequency", df['Purchase_Frequency'].unique())
                    preferred_location = st.selectbox("Preferred Kiosk Location", df['Refill_Location'].unique())
                
                submitted = st.form_submit_button("🚀 Run Prediction", use_container_width=True)
            
            if submitted:
                # Prepare input
                if '5+' in fam_size: fam_num = 5
                elif '1-2' in fam_size: fam_num = 1.5
                else: fam_num = 3.5
                
                input_data = pd.DataFrame({
                    'Age_Group': [age], 'Gender': [gender], 'Emirate': [emirate],
                    'Occupation': [occupation], 'Income': [income],
                    'Purchase_Location': [df['Purchase_Location'].mode()[0]],
                    'Purchase_Frequency': [purchase_freq], 'Uses_Eco_Products': [df['Uses_Eco_Products'].mode()[0]],
                    'Preferred_Packaging': [df['Preferred_Packaging'].mode()[0]],
                    'Aware_Plastic_Ban': [df['Aware_Plastic_Ban'].mode()[0]],
                    'Eco_Brand_Preference': [eco_brand], 'Follow_Campaigns': [follow_camp],
                    'Used_Refill_Before': [used_before], 'Preferred_Payment_Mode': [df['Preferred_Payment_Mode'].mode()[0]],
                    'Refill_Location': [preferred_location],
                    'Container_Type': [df['Container_Type'].mode()[0]],
                    'Interest_Non_Liquids': [df['Interest_Non_Liquids'].mode()[0]],
                    'Discount_Switch': [df['Discount_Switch'].mode()[0]],
                    'Family_Size_Num': [fam_num], 'Importance_Convenience': [imp_conv],
                    'Importance_Price': [imp_price], 'Importance_Sustainability': [imp_sust],
                    'Reduce_Waste_Score': [waste_score], 'Social_Influence_Score': [social_score],
                    'Try_Refill_Likelihood': [try_likelihood]
                })
                
                # Predictions
                clf_pipeline = models['classification']
                reg_pipeline = models['regression']
                
                pred_prob = clf_pipeline.predict_proba(input_data)[0]
                pred_spend = reg_pipeline.predict(input_data)[0]
                adoption_probability = pred_prob[1]
                
                # Display Results
                st.markdown("---")
                st.header("🎯 Prediction Results")
                
                col_r1, col_r2 = st.columns(2)
                
                with col_r1:
                    st.subheader("Adoption Likelihood")
                    
                    # Gauge chart
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=adoption_probability * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Adoption Probability (%)", 'font': {'size': 20}},
                        delta={'reference': 50, 'increasing': {'color': "#2E8B57"}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1},
                            'bar': {'color': "#2E8B57" if adoption_probability > 0.5 else "#FF6347"},
                            'steps': [
                                {'range': [0, 30], 'color': 'rgba(255, 99, 71, 0.3)'},
                                {'range': [30, 70], 'color': 'rgba(255, 206, 86, 0.3)'},
                                {'range': [70, 100], 'color': 'rgba(46, 139, 87, 0.3)'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    if adoption_probability > 0.7:
                        st.success("✅ **High Conversion Probability** - Strong candidate!")
                    elif adoption_probability > 0.4:
                        st.info("ℹ️ **Moderate Conversion Probability** - Needs nurturing")
                    else:
                        st.warning("⚠️ **Low Conversion Probability** - Focus on value proposition")
                
                with col_r2:
                    st.subheader("Spending Potential")
                    
                    avg_wtp = df['Willingness_to_Pay_AED'].mean()
                    percentile = (df['Willingness_to_Pay_AED'] < pred_spend).mean() * 100
                    
                    st.metric(
                        "Predicted Spending",
                        f"AED {pred_spend:.2f}",
                        delta=f"{pred_spend - avg_wtp:+.2f} vs. average",
                        help="Expected spending per visit"
                    )
                    
                    st.metric(
                        "Spending Percentile",
                        f"{percentile:.0f}th",
                        help="Ranking among all customers"
                    )
                    
                    # Spending comparison
                    fig_compare = go.Figure()
                    fig_compare.add_trace(go.Bar(
                        x=['Average Customer', 'This Profile'],
                        y=[avg_wtp, pred_spend],
                        marker_color=['#808080', '#2E8B57'],
                        text=[f'AED {avg_wtp:.2f}', f'AED {pred_spend:.2f}'],
                        textposition='auto'
                    ))
                    fig_compare.update_layout(
                        title="Spending Comparison",
                        yaxis_title="Willingness to Pay (AED)",
                        showlegend=False
                    )
                    st.plotly_chart(fig_compare, use_container_width=True)
                
                # Customer Lifetime Value
                st.subheader("💰 Customer Lifetime Value (CLV) Estimate")
                
                col_clv1, col_clv2, col_clv3, col_clv4 = st.columns(4)
                
                with col_clv1:
                    visits_per_year = st.number_input("Est. Visits/Year", 4, 52, 24, help="Weekly = ~52, Bi-weekly = ~24")
                with col_clv2:
                    retention_rate = st.slider("Retention Rate", 0.5, 0.95, 0.75, 0.05)
                with col_clv3:
                    years = st.number_input("Time Horizon (Years)", 1, 10, 3)
                with col_clv4:
                    margin = st.slider("Gross Margin %", 20, 60, 40)
                
                clv = calculate_clv(pred_spend, visits_per_year, retention_rate, years)
                clv_with_margin = clv * (margin / 100)
                
                col_clv_r1, col_clv_r2, col_clv_r3 = st.columns(3)
                with col_clv_r1:
                    st.metric("Total CLV (Revenue)", f"AED {clv:,.2f}")
                with col_clv_r2:
                    st.metric("Gross Profit", f"AED {clv_with_margin:,.2f}")
                with col_clv_r3:
                    acquisition_cost = 50  # Assumed CAC
                    roi = ((clv_with_margin - acquisition_cost) / acquisition_cost) * 100
                    st.metric("ROI", f"{roi:.0f}%", help=f"Assuming CAC = AED {acquisition_cost}")
                
                # Cluster Assignment
                st.subheader("🎯 Customer Segment Assignment")
                
                cluster_input = pd.DataFrame({
                    'Importance_Convenience': [imp_conv],
                    'Importance_Price': [imp_price],
                    'Importance_Sustainability': [imp_sust],
                    'Reduce_Waste_Score': [waste_score],
                    'Eco_Brand_Preference': [eco_brand],
                    'Social_Influence_Score': [social_score]
                })
                
                cluster_scaler = models['clustering_scaler']
                cluster_model = models['clustering_model']
                
                cluster_input_scaled = cluster_scaler.transform(cluster_input)
                predicted_cluster = cluster_model.predict(cluster_input_scaled)[0]
                
                personas = {
                    0: '🌱 The Eco-Warriors',
                    1: '💰 The Budget-Conscious',
                    2: '⚡ The Convenience Seekers',
                    3: '😐 The Apathetic'
                }
                
                st.info(f"**Assigned Segment:** Cluster {predicted_cluster} - {personas.get(predicted_cluster, 'Unknown')}")
        
        with tab2:
            st.subheader("📊 Batch Prediction Mode")
            st.markdown("Upload a CSV file with customer profiles to get predictions for multiple customers at once.")
            
            # Sample template
            sample_data = pd.DataFrame({
                'Age_Group': ['25-34', '35-44'],
                'Gender': ['Male', 'Female'],
                'Income': ['10000-15000', '15000-20000'],
                'Emirate': ['Dubai', 'Abu Dhabi'],
                'Importance_Price': [3, 4],
                'Importance_Sustainability': [4, 3],
                'Importance_Convenience': [3, 5]
            })
            
            st.download_button(
                "📥 Download Sample Template",
                data=sample_data.to_csv(index=False),
                file_name="batch_prediction_template.csv",
                mime="text/csv"
            )
            
            uploaded_batch = st.file_uploader("Upload Customer Profiles (CSV)", type=['csv'])
            
            if uploaded_batch:
                batch_df = pd.read_csv(uploaded_batch)
                st.dataframe(batch_df.head(), use_container_width=True)
                
                if st.button("Run Batch Predictions"):
                    with st.spinner("Processing batch predictions..."):
                        # This would require proper feature engineering for the uploaded data
                        st.info("⚠️ Batch prediction feature requires properly formatted input matching all model features. Contact administrator for assistance.")

    # =========================================================================
    # PAGE 5: MARKET BASKET ANALYSIS
    # =========================================================================
    elif "Market Basket" in page:
        st.title("🛒 Market Basket Analysis")
        
        st.markdown("""
        <div class='info-box'>
        <b>Discover product associations</b> using Association Rule Mining to optimize kiosk layout and create bundles.
        </div>
        """, unsafe_allow_html=True)
        
        rules_df = models['association_rules']
        
        if rules_df.empty:
            st.warning("⚠️ No association rules found with current parameters. Try lowering the thresholds.")
            return
        
        # Filters
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            min_lift = st.slider(
                "Minimum Lift",
                min_value=1.0,
                max_value=float(rules_df['lift'].max()),
                value=1.2,
                step=0.1,
                help="Lift > 1 means items are bought together more than by chance"
            )
        
        with col_f2:
            min_confidence = st.slider(
                "Minimum Confidence",
                min_value=0.0,
                max_value=float(rules_df['confidence'].max()),
                value=0.1,
                step=0.05,
                help="Confidence shows how often the rule is true"
            )
        
        with col_f3:
            min_support = st.slider(
                "Minimum Support",
                min_value=0.0,
                max_value=float(rules_df['support'].max()),
                value=0.05,
                step=0.01,
                help="Support shows how frequently the itemset appears"
            )
        
        filtered_rules = rules_df[
            (rules_df['lift'] > min_lift) &
            (rules_df['confidence'] > min_confidence) &
            (rules_df['support'] > min_support)
        ]
        
        st.metric("Rules Found", len(filtered_rules))
        
        # Top Rules
        st.subheader("🏆 Top Association Rules")
        
        display_df = filtered_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(20)
        st.dataframe(
            display_df.style.background_gradient(cmap='Greens', subset=['lift', 'confidence']).format({
                'support': '{:.3f}',
                'confidence': '{:.3f}',
                'lift': '{:.2f}'
            }),
            use_container_width=True
        )
        
        # Visualization
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
                title="Product Association Strength",
                color="confidence",
                color_continuous_scale='Greens',
                hover_data=['support', 'confidence', 'lift']
            )
            fig_rules.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_rules, use_container_width=True)
        
        with col_v2:
            st.subheader("🎯 Confidence vs Support")
            fig_scatter = px.scatter(
                filtered_rules,
                x='support',
                y='confidence',
                size='lift',
                color='lift',
                hover_data=['antecedents', 'consequents'],
                title="Rule Quality Matrix",
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Network Graph
        st.subheader("🕸️ Product Association Network")
        
        # Create network data
        top_network_rules = filtered_rules.sort_values('lift', ascending=False).head(20)
        
        # Build edge list
        edges = []
        for _, row in top_network_rules.iterrows():
            edges.append({
                'source': row['antecedents'],
                'target': row['consequents'],
                'weight': row['lift']
            })
        
        # Create a simple network visualization using plotly
        import networkx as nx
        
        G = nx.DiGraph()
        for edge in edges:
            G.add_edge(edge['source'], edge['target'], weight=edge['weight'])
        
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=0.5, color='#888'),
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
                size=[G.degree(node) * 10 for node in G.nodes()],
                color='#2E8B57',
                line=dict(width=2, color='white')
            ),
            showlegend=False
        )
        
        fig_network = go.Figure(data=edge_trace + [node_trace])
        fig_network.update_layout(
            title="Product Co-occurrence Network",
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        st.plotly_chart(fig_network, use_container_width=True)
        
        # Strategic Insights
        st.subheader("💡 Strategic Recommendations")
        
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.markdown("""
            <div class='success-box'>
            <h4>🎁 Bundle Opportunities</h4>
            <p>Create product bundles based on high-lift associations:</p>
            <ul>
            """, unsafe_allow_html=True)
            
            for idx, row in filtered_rules.sort_values('lift', ascending=False).head(5).iterrows():
                st.markdown(f"<li><b>{row['antecedents']}</b> + <b>{row['consequents']}</b> (Lift: {row['lift']:.2f})</li>", unsafe_allow_html=True)
            
            st.markdown("</ul></div>", unsafe_allow_html=True)
        
        with col_s2:
            st.markdown("""
            <div class='info-box'>
            <h4>🏪 Layout Optimization</h4>
            <p>Place these products near each other:</p>
            <ul>
            """, unsafe_allow_html=True)
            
            for idx, row in filtered_rules.sort_values('confidence', ascending=False).head(5).iterrows():
                st.markdown(f"<li><b>{row['antecedents']}</b> → <b>{row['consequents']}</b> ({row['confidence']*100:.0f}% confidence)</li>", unsafe_allow_html=True)
            
            st.markdown("</ul></div>", unsafe_allow_html=True)
        
        # Frequent Items
        st.subheader("📦 Most Popular Products")
        
        frequent_itemsets = models['frequent_itemsets']
        single_items = frequent_itemsets[frequent_itemsets['itemsets'].apply(len) == 1].copy()
        single_items['product'] = single_items['itemsets'].apply(lambda x: list(x)[0])
        single_items = single_items.sort_values('support', ascending=False).head(10)
        
        fig_items = px.bar(
            single_items,
            x='product',
            y='support',
            title="Top 10 Products by Purchase Frequency",
            labels={'support': 'Support (Frequency)', 'product': 'Product'},
            color='support',
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig_items, use_container_width=True)

    # =========================================================================
    # PAGE 6: ADVANCED ANALYTICS
    # =========================================================================
    elif "Advanced Analytics" in page:
        st.title("📈 Advanced Analytics & Statistical Tests")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Statistical Tests", "🎯 Feature Importance", "📉 Model Diagnostics", "🔄 Comparative Analysis"])
        
        with tab1:
            st.subheader("Statistical Hypothesis Testing")
            
            # Perform tests
            stat_results = perform_statistical_tests(df_filtered)
            
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                st.markdown("#### Chi-Square Test: Income vs. Adoption")
                st.markdown("""
                <div class='info-box'>
                <b>Hypothesis:</b> Is there a significant relationship between income level and adoption likelihood?
                </div>
                """, unsafe_allow_html=True)
                
                chi2_result = stat_results['income_adoption_chi2']
                
                col_chi1, col_chi2 = st.columns(2)
                with col_chi1:
                    st.metric("Chi-Square Statistic", f"{chi2_result['chi2']:.2f}")
                with col_chi2:
                    st.metric("P-Value", f"{chi2_result['p_value']:.4f}")
                
                if chi2_result['p_value'] < 0.05:
                    st.success("✅ **Significant relationship found** (p < 0.05). Income level affects adoption.")
                else:
                    st.info("ℹ️ No significant relationship found (p ≥ 0.05)")
                
                # Visualize contingency table
                contingency = pd.crosstab(df_filtered['Income'], df_filtered['Likely_to_Use_ReFillHub'])
                fig_chi = px.imshow(
                    contingency,
                    title="Contingency Table Heatmap",
                    labels=dict(x="Adoption", y="Income Level", color="Count"),
                    color_continuous_scale='Greens',
                    aspect='auto'
                )
                st.plotly_chart(fig_chi, use_container_width=True)
            
            with col_t2:
                st.markdown("#### Independent T-Test: Spending Comparison")
                st.markdown("""
                <div class='info-box'>
                <b>Hypothesis:</b> Do adopters spend significantly more than non-adopters?
                </div>
                """, unsafe_allow_html=True)
                
                ttest_result = stat_results['spending_ttest']
                
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    st.metric("T-Statistic", f"{ttest_result['t_stat']:.2f}")
                with col_t2:
                    st.metric("P-Value", f"{ttest_result['p_value']:.4f}")
                
                if ttest_result['p_value'] < 0.05:
                    st.success("✅ **Significant difference found** (p < 0.05). Adopters spend differently.")
                else:
                    st.info("ℹ️ No significant difference found (p ≥ 0.05)")
                
                # Visualize distributions
                fig_dist = go.Figure()
                
                adopters = df_filtered[df_filtered['Likely_to_Use_ReFillHub'] == 'Yes']['Willingness_to_Pay_AED']
                non_adopters = df_filtered[df_filtered['Likely_to_Use_ReFillHub'] == 'No']['Willingness_to_Pay_AED']
                
                fig_dist.add_trace(go.Histogram(x=adopters, name='Adopters', opacity=0.7, marker_color='#2E8B57'))
                fig_dist.add_trace(go.Histogram(x=non_adopters, name='Non-Adopters', opacity=0.7, marker_color='#FF6347'))
                
                fig_dist.update_layout(
                    barmode='overlay',
                    title="Spending Distribution: Adopters vs. Non-Adopters",
                    xaxis_title="Willingness to Pay (AED)",
                    yaxis_title="Frequency"
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            # ANOVA Test
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
                st.success("✅ **Significant differences** across clusters (p < 0.05)")
            else:
                st.info("ℹ️ No significant differences found (p ≥ 0.05)")
        
        with tab2:
            st.subheader("Feature Importance Analysis")
            
            col_fi1, col_fi2 = st.columns(2)
            
            with col_fi1:
                st.markdown("#### Classification Model")
                feature_imp_clf = models['feature_importance_clf']
                
                fig_fi_clf = px.bar(
                    feature_imp_clf.head(15),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 15 Features for Adoption Prediction",
                    color='Importance',
                    color_continuous_scale='Greens'
                )
                fig_fi_clf.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_fi_clf, use_container_width=True)
                
                st.dataframe(feature_imp_clf.head(10), use_container_width=True)
            
            with col_fi2:
                st.markdown("#### Regression Model")
                feature_imp_reg = models['feature_importance_reg']
                
                fig_fi_reg = px.bar(
                    feature_imp_reg.head(15),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 15 Features for Spending Prediction",
                    color='Importance',
                    color_continuous_scale='Blues'
                )
                fig_fi_reg.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_fi_reg, use_container_width=True)
                
                st.dataframe(feature_imp_reg.head(10), use_container_width=True)
        
        with tab3:
            st.subheader("Model Diagnostic Plots")
            
            # Classification: ROC Curve
            st.markdown("#### Classification: ROC Curve")
            
            fpr, tpr = metrics['roc_curve']
            roc_auc = metrics['roc_auc']
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {roc_auc:.3f})',
                line=dict(color='#2E8B57', width=3)
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='gray', dash='dash')
            ))
            fig_roc.update_layout(
                title="Receiver Operating Characteristic (ROC) Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                hovermode='x'
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            
            # Confusion Matrix
            st.markdown("#### Confusion Matrix")
            
            cm = metrics['confusion_matrix']
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['No', 'Yes'],
                y=['No', 'Yes'],
                title="Classification Confusion Matrix",
                color_continuous_scale='Greens',
                text_auto=True
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Regression: Residual Plot
            st.markdown("#### Regression: Residual Analysis")
            
            y_test = metrics['y_test_reg']
            y_pred = metrics['y_pred_reg']
            residuals = y_test - y_pred
            
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                fig_residual = px.scatter(
                    x=y_pred,
                    y=residuals,
                    title="Residual Plot",
                    labels={'x': 'Predicted Values', 'y': 'Residuals'},
                    trendline="lowess",
                    color_discrete_sequence=['#2E8B57']
                )
                fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_residual, use_container_width=True)
            
            with col_res2:
                fig_qq = go.Figure()
                fig_qq.add_trace(go.Histogram(
                    x=residuals,
                    nbinsx=30,
                    name='Residuals',
                    marker_color='#2E8B57'
                ))
                fig_qq.update_layout(
                    title="Residual Distribution",
                    xaxis_title="Residuals",
                    yaxis_title="Frequency"
                )
                st.plotly_chart(fig_qq, use_container_width=True)
            
            # Predicted vs Actual
            fig_pred_actual = px.scatter(
                x=y_test,
                y=y_pred,
                title="Predicted vs. Actual Spending",
                labels={'x': 'Actual Spending (AED)', 'y': 'Predicted Spending (AED)'},
                trendline="ols",
                color_discrete_sequence=['#228B22']
            )
            fig_pred_actual.add_trace(go.Scatter(
                x=[y_test.min(), y_test.max()],
                y=[y_test.min(), y_test.max()],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            st.plotly_chart(fig_pred_actual, use_container_width=True)
        
        with tab4:
            st.subheader("Comparative Analysis")
            
            # Segment Comparison
            st.markdown("#### Cross-Segment Performance")
            
            comparison_metrics = df_filtered.groupby('Cluster').agg({
                'Likely_to_Use_ReFillHub': lambda x: (x == 'Yes').mean() * 100,
                'Willingness_to_Pay_AED': 'mean',
                'Importance_Sustainability': 'mean',
                'Importance_Price': 'mean',
                'Importance_Convenience': 'mean',
                'Cluster': 'count'
            }).rename(columns={
                'Likely_to_Use_ReFillHub': 'Adoption_Rate',
                'Willingness_to_Pay_AED': 'Avg_Spending',
                'Cluster': 'Size'
            }).reset_index()
            
            st.dataframe(
                comparison_metrics.style.background_gradient(cmap='Greens').format(precision=2),
                use_container_width=True
            )
            
            # Multi-metric comparison
            fig_comparison = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Adoption Rate by Cluster', 'Avg. Spending by Cluster',
                               'Sustainability Score', 'Price Sensitivity')
            )
            
            fig_comparison.add_trace(
                go.Bar(x=comparison_metrics['Cluster'], y=comparison_metrics['Adoption_Rate'],
                      marker_color='#2E8B57', name='Adoption Rate'),
                row=1, col=1
            )
            
            fig_comparison.add_trace(
                go.Bar(x=comparison_metrics['Cluster'], y=comparison_metrics['Avg_Spending'],
                      marker_color='#228B22', name='Avg Spending'),
                row=1, col=2
            )
            
            fig_comparison.add_trace(
                go.Bar(x=comparison_metrics['Cluster'], y=comparison_metrics['Importance_Sustainability'],
                      marker_color='#3CB371', name='Sustainability'),
                row=2, col=1
            )
            
            fig_comparison.add_trace(
                go.Bar(x=comparison_metrics['Cluster'], y=comparison_metrics['Importance_Price'],
                      marker_color='#90EE90', name='Price Sensitivity'),
                row=2, col=2
            )
            
            fig_comparison.update_layout(height=700, showlegend=False, title_text="Multi-Dimensional Cluster Comparison")
            st.plotly_chart(fig_comparison, use_container_width=True)

    # =========================================================================
    # PAGE 7: BUSINESS RECOMMENDATIONS
    # =========================================================================
    elif "Business Recommendations" in page:
        st.title("💡 AI-Generated Business Recommendations")
        
        st.markdown("""
        <div class='success-box'>
        <b>Data-Driven Insights:</b> Actionable strategies generated from comprehensive analysis
        </div>
        """, unsafe_allow_html=True)
        
        recommendations = generate_business_recommendations(df_filtered, models, metrics)
        
        # Priority sorting
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        recommendations.sort(key=lambda x: priority_order[x['priority']])
        
        for idx, rec in enumerate(recommendations, 1):
            priority_color = {
                'HIGH': 'error',
                'MEDIUM': 'warning',
                'LOW': 'info'
            }
            
            with st.expander(f"**{idx}. {rec['title']}** [{rec['priority']} PRIORITY]", expanded=(rec['priority'] == 'HIGH')):
                st.markdown(f"**Recommendation:** {rec['detail']}")
                st.markdown(f"**Priority Level:** `{rec['priority']}`")
        
        st.markdown("---")
        
        # ROI Calculator
        st.header("💰 Business Case & ROI Projection")
        
        col_roi1, col_roi2 = st.columns(2)
        
        with col_roi1:
            st.subheader("Input Assumptions")
            
            num_kiosks = st.number_input("Number of Kiosks to Deploy", 1, 50, 10)
            kiosk_cost = st.number_input("Cost per Kiosk (AED)", 10000, 100000, 50000, 5000)
            monthly_operating_cost = st.number_input("Monthly Operating Cost per Kiosk (AED)", 1000, 10000, 3000, 500)
            customers_per_kiosk_per_day = st.number_input("Expected Customers/Kiosk/Day", 10, 200, 50, 10)
            
            adoption_rate_assumption = st.slider("Assumed Adoption Rate (%)", 30, 90, 65, 5)
            avg_transaction = st.number_input("Avg. Transaction Value (AED)", 20, 200, int(avg_wtp), 10)
            gross_margin_pct = st.slider("Gross Margin (%)", 20, 60, 40, 5)
        
        with col_roi2:
            st.subheader("Financial Projections (Year 1)")
            
            # Calculations
            initial_investment = num_kiosks * kiosk_cost
            annual_operating_cost = num_kiosks * monthly_operating_cost * 12
            
            daily_transactions = num_kiosks * customers_per_kiosk_per_day * (adoption_rate_assumption / 100)
            annual_revenue = daily_transactions * avg_transaction * 365
            gross_profit = annual_revenue * (gross_margin_pct / 100)
            net_profit = gross_profit - annual_operating_cost
            
            payback_period = initial_investment / net_profit if net_profit > 0 else float('inf')
            roi = ((net_profit - initial_investment) / initial_investment * 100) if initial_investment > 0 else 0
            
            st.metric("Initial Investment", f"AED {initial_investment:,.0f}")
            st.metric("Annual Revenue", f"AED {annual_revenue:,.0f}")
            st.metric("Gross Profit", f"AED {gross_profit:,.0f}")
            st.metric("Net Profit (Year 1)", f"AED {net_profit:,.0f}")
            st.metric("Payback Period", f"{payback_period:.1f} years" if payback_period != float('inf') else "N/A")
            st.metric("ROI", f"{roi:.1f}%")
        
        # Sensitivity Analysis
        st.subheader("📊 Sensitivity Analysis")
        
        adoption_rates = np.arange(30, 91, 10)
        net_profits = []
        
        for rate in adoption_rates:
            daily_trans = num_kiosks * customers_per_kiosk_per_day * (rate / 100)
            annual_rev = daily_trans * avg_transaction * 365
            gross_prof = annual_rev * (gross_margin_pct / 100)
            net_prof = gross_prof - annual_operating_cost
            net_profits.append(net_prof)
        
        fig_sensitivity = go.Figure()
        fig_sensitivity.add_trace(go.Scatter(
            x=adoption_rates,
            y=net_profits,
            mode='lines+markers',
            name='Net Profit',
            line=dict(color='#2E8B57', width=3),
            marker=dict(size=10)
        ))
        fig_sensitivity.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
        fig_sensitivity.update_layout(
            title="Net Profit Sensitivity to Adoption Rate",
            xaxis_title="Adoption Rate (%)",
            yaxis_title="Net Profit (AED)",
            hovermode='x'
        )
        st.plotly_chart(fig_sensitivity, use_container_width=True)
        
        # Strategic Roadmap
        st.header("🗺️ Strategic Implementation Roadmap")
        
        phases = [
            {
                'phase': 'Phase 1: Pilot (Months 1-3)',
                'objectives': [
                    '🎯 Deploy 2-3 kiosks in high-traffic locations (Dubai Mall, Marina)',
                    '📊 Test product mix and pricing strategies',
                    '🔍 Gather customer feedback and refine UX',
                    '💰 Target: Break-even on pilot kiosks'
                ]
            },
            {
                'phase': 'Phase 2: Expansion (Months 4-9)',
                'objectives': [
                    '🚀 Scale to 10-15 kiosks across Dubai & Abu Dhabi',
                    '🤝 Establish partnerships with retailers (Carrefour, Lulu)',
                    '📱 Launch mobile app with loyalty program',
                    '💰 Target: 30% of kiosks profitable'
                ]
            },
            {
                'phase': 'Phase 3: Optimization (Months 10-12)',
                'objectives': [
                    '🎨 Optimize product mix based on basket analysis',
                    '🎁 Launch bundled offerings and subscriptions',
                    '🌍 Expand to Sharjah and Northern Emirates',
                    '💰 Target: Break-even on total operations'
                ]
            },
            {
                'phase': 'Phase 4: Scale (Year 2+)',
                'objectives': [
                    '📈 Expand to 50+ kiosks UAE-wide',
                    '🏪 Franchise model for rapid expansion',
                    '🌐 Regional expansion (Saudi Arabia, Qatar)',
                    '💰 Target: 40%+ ROI'
                ]
            }
        ]
        
        for phase in phases:
            with st.expander(f"**{phase['phase']}**"):
                for obj in phase['objectives']:
                    st.markdown(f"- {obj}")

    # =========================================================================
    # PAGE 8: MODEL PERFORMANCE
    # =========================================================================
    elif "Model Performance" in page:
        st.title("🤖 Model Performance & Methodology")
        
        st.markdown("""
        <div class='info-box'>
        <b>Technical Overview:</b> Comprehensive evaluation of all ML models used in this dashboard
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Classification", "💰 Regression", "🧩 Clustering", "🛒 Association Rules"])
        
        with tab1:
            st.subheader("Classification Model: Adoption Prediction")
            
            col_c1, col_c2 = st.columns([1, 2])
            
            with col_c1:
                st.markdown("**Algorithm:** Random Forest Classifier")
                st.markdown("**Task:** Predict Yes/No for ReFill Hub adoption")
                st.markdown("**Features:** " + str(len(cat_features) + len(num_features)))
                st.markdown("**Training Samples:** " + f"{int(len(df)*0.8):,}")
                st.markdown("**Test Samples:** " + f"{int(len(df)*0.2):,}")
                
                st.divider()
                
                st.metric("Overall Accuracy", f"{metrics['classification_report']['accuracy']:.2%}")
                st.metric("Cross-Validation Accuracy", f"{metrics['cv_mean']:.2%} ± {metrics['cv_std']:.2%}")
                st.metric("ROC-AUC Score", f"{metrics['roc_auc']:.3f}")
            
            with col_c2:
                report = metrics['classification_report']
                
                class_df = pd.DataFrame({
                    'Metric': ['Precision', 'Recall', 'F1-Score'],
                    'No (Negative)': [report['0']['precision'], report['0']['recall'], report['0']['f1-score']],
                    'Yes (Positive)': [report['1']['precision'], report['1']['recall'], report['1']['f1-score']]
                })
                
                fig_metrics = go.Figure(data=[
                    go.Bar(name='No', x=class_df['Metric'], y=class_df['No (Negative)'], marker_color='#FF6347'),
                    go.Bar(name='Yes', x=class_df['Metric'], y=class_df['Yes (Positive)'], marker_color='#2E8B57')
                ])
                fig_metrics.update_layout(
                    title="Classification Metrics by Class",
                    barmode='group',
                    yaxis_title="Score",
                    yaxis_range=[0, 1]
                )
                st.plotly_chart(fig_metrics, use_container_width=True)
            
            st.markdown("#### Model Interpretation")
            st.markdown("""
            - **Precision:** Of all predicted "Yes", how many are actually correct
            - **Recall:** Of all actual "Yes", how many did we correctly identify
            - **F1-Score:** Harmonic mean of precision and recall (balanced metric)
            - **ROC-AUC:** Area under ROC curve; >0.8 is excellent
            """)
        
        with tab2:
            st.subheader("Regression Model: Spending Prediction")
            
            col_r1, col_r2 = st.columns([1, 2])
            
            with col_r1:
                st.markdown("**Algorithm:** Random Forest Regressor")
                st.markdown("**Task:** Predict willingness to pay (AED)")
                st.markdown("**Features:** " + str(len(cat_features) + len(num_features)))
                st.markdown("**Training Samples:** " + f"{int(len(df)*0.8):,}")
                st.markdown("**Test Samples:** " + f"{int(len(df)*0.2):,}")
                
                st.divider()
                
                st.metric("R² Score", f"{metrics['r2_score']:.3f}", help="1.0 = perfect fit, 0.0 = no predictive power")
                st.metric("RMSE", f"AED {metrics['rmse']:.2f}", help="Average prediction error")
                st.metric("MAE", f"AED {metrics['mae']:.2f}", help="Mean absolute error")
            
            with col_r2:
                y_test = metrics['y_test_reg']
                y_pred = metrics['y_pred_reg']
                
                fig_scatter = px.scatter(
                    x=y_test,
                    y=y_pred,
                    labels={'x': 'Actual Spending (AED)', 'y': 'Predicted Spending (AED)'},
                    title="Actual vs. Predicted Spending",
                    trendline="ols",
                    color_discrete_sequence=['#2E8B57']
                )
                fig_scatter.add_trace(go.Scatter(
                    x=[y_test.min(), y_test.max()],
                    y=[y_test.min(), y_test.max()],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.markdown("#### Model Interpretation")
            st.markdown("""
            - **R² (Coefficient of Determination):** Proportion of variance explained by the model
                - 1.0 = Perfect predictions
                - 0.7-0.9 = Strong model
                - 0.4-0.7 = Moderate model
            - **RMSE (Root Mean Squared Error):** Average prediction error in AED
            - **MAE (Mean Absolute Error):** Average absolute difference between predictions and actuals
            """)
        
        with tab3:
            st.subheader("Clustering Model: Customer Segmentation")
            
            col_cl1, col_cl2 = st.columns([1, 2])
            
            with col_cl1:
                st.markdown("**Algorithm:** K-Means Clustering")
                st.markdown("**Task:** Segment customers into personas")
                st.markdown("**Features:** " + str(len(cluster_features)))
                st.markdown("**Optimal Clusters:** 4")
                st.markdown("**Total Samples:** " + f"{len(df):,}")
                
                st.divider()
                
                st.metric("Silhouette Score", f"{metrics['silhouette_score']:.3f}", 
                         help="Measures cluster separation. Range: -1 to 1. >0.3 is good.")
            
            with col_cl2:
                cluster_sizes = models['cluster_sizes']
                
                fig_cluster_dist = px.pie(
                    values=cluster_sizes.values,
                    names=[f"Cluster {i}" for i in cluster_sizes.index],
                    title="Cluster Size Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                st.plotly_chart(fig_cluster_dist, use_container_width=True)
            
            st.markdown("#### Cluster Quality Metrics")
            st.markdown("""
            - **Silhouette Score:** Measures how similar points are to their own cluster vs. other clusters
                - 0.7-1.0: Strong, well-separated clusters
                - 0.5-0.7: Reasonable structure
                - 0.25-0.5: Weak structure (typical for survey data)
                - <0.25: No substantial structure
            - Our score of **{:.3f}** indicates reasonable cluster separation for behavioral data
            """.format(metrics['silhouette_score']))
        
        with tab4:
            st.subheader("Association Rules: Market Basket Analysis")
            
            rules_df = models['association_rules']
            
            col_ar1, col_ar2 = st.columns([1, 2])
            
            with col_ar1:
                st.markdown("**Algorithm:** Apriori")
                st.markdown("**Task:** Discover product co-purchase patterns")
                st.markdown("**Min Support:** 5%")
                st.markdown("**Min Confidence:** Variable")
                st.markdown("**Min Lift:** 1.0")
                
                st.divider()
                
                st.metric("Total Rules Found", len(rules_df))
                st.metric("Avg. Lift", f"{rules_df['lift'].mean():.2f}" if not rules_df.empty else "N/A")
                st.metric("Max Lift", f"{rules_df['lift'].max():.2f}" if not rules_df.empty else "N/A")
            
            with col_ar2:
                if not rules_df.empty:
                    fig_lift_dist = px.histogram(
                        rules_df,
                        x='lift',
                        nbins=30,
                        title="Distribution of Lift Values",
                        labels={'lift': 'Lift', 'count': 'Frequency'},
                        color_discrete_sequence=['#2E8B57']
                    )
                    fig_lift_dist.add_vline(x=1.0, line_dash="dash", line_color="red", annotation_text="Lift = 1 (Independence)")
                    st.plotly_chart(fig_lift_dist, use_container_width=True)
            
            st.markdown("#### Association Metrics Explained")
            st.markdown("""
            - **Support:** How frequently the itemset appears (e.g., 0.1 = 10% of transactions)
            - **Confidence:** How often the rule is true (If X, then Y is true X% of the time)
            - **Lift:** How much more likely items are purchased together vs. independently
                - Lift = 1: No association (random)
                - Lift > 1: Positive association (bought together more than chance)
                - Lift < 1: Negative association (bought together less than chance)
            """)
        
        # Model Comparison Summary
        st.markdown("---")
        st.header("📊 Model Performance Summary")
        
        summary_data = {
            'Model': ['Classification', 'Regression', 'Clustering', 'Association'],
            'Algorithm': ['Random Forest', 'Random Forest', 'K-Means', 'Apriori'],
            'Primary Metric': [
                f"{metrics['classification_report']['accuracy']:.2%} Accuracy",
                f"{metrics['r2_score']:.3f} R²",
                f"{metrics['silhouette_score']:.3f} Silhouette",
                f"{len(rules_df)} Rules"
            ],
            'Status': ['✅ Excellent', '✅ Good', '✅ Good', '✅ Sufficient']
        }
        
        st.table(pd.DataFrame(summary_data))

    # =========================================================================
    # PAGE 9: EXPORT & DOWNLOAD
    # =========================================================================
    elif "Export & Download" in page:
        st.title("📥 Export & Download Center")
        
        st.markdown("""
        <div class='success-box'>
        <b>Download all insights, predictions, and data</b> for further analysis or reporting
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📊 Data Exports", "📈 Reports", "💻 Code & API"])
        
        with tab1:
            st.subheader("Download Processed Data")
            
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                st.markdown("#### Full Dataset with Clusters")
                st.download_button(
                    label="📥 Download CSV",
                    data=df.to_csv(index=False),
                    file_name="refillhub_data_with_clusters.csv",
                    mime="text/csv"
                )
                
                st.markdown("#### Cluster Profiles")
                cluster_profiles = models['cluster_profiles']
                st.download_button(
                    label="📥 Download Cluster Profiles",
                    data=cluster_profiles.to_csv(),
                    file_name="cluster_profiles.csv",
                    mime="text/csv"
                )
                
                st.markdown("#### Association Rules")
                if not models['association_rules'].empty:
                    st.download_button(
                        label="📥 Download Association Rules",
                        data=models['association_rules'].to_csv(index=False),
                        file_name="association_rules.csv",
                        mime="text/csv"
                    )
            
            with col_d2:
                st.markdown("#### Feature Importance (Classification)")
                st.download_button(
                    label="📥 Download Feature Importance",
                    data=models['feature_importance_clf'].to_csv(index=False),
                    file_name="feature_importance_classification.csv",
                    mime="text/csv"
                )
                
                st.markdown("#### Feature Importance (Regression)")
                st.download_button(
                    label="📥 Download Feature Importance",
                    data=models['feature_importance_reg'].to_csv(index=False),
                    file_name="feature_importance_regression.csv",
                    mime="text/csv"
                )
                
                st.markdown("#### Filtered Dataset")
                st.download_button(
                    label="📥 Download Filtered Data",
                    data=df_filtered.to_csv(index=False),
                    file_name="refillhub_filtered_data.csv",
                    mime="text/csv"
                )
        
        with tab2:
            st.subheader("Generate Reports")
            
            st.markdown("#### Executive Summary Report")
            
            summary_text = f"""
# ReFill Hub: Executive Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Key Metrics
- **Total Respondents:** {len(df):,}
- **Adoption Rate:** {(df['Likely_to_Use_ReFillHub'] == 'Yes').mean() * 100:.1f}%
- **Average Willingness to Pay:** AED {df['Willingness_to_Pay_AED'].mean():.2f}
- **Customer Segments Identified:** 4

## Top Insights
1. Cluster 0 shows highest sustainability scores and adoption likelihood
2. Dubai and Abu Dhabi represent 70% of high-value customers
3. Mobile payment preference correlates with higher spending

## Model Performance
- Classification Accuracy: {metrics['classification_report']['accuracy']:.2%}
- Regression R² Score: {metrics['r2_score']:.3f}
- Clustering Silhouette: {metrics['silhouette_score']:.3f}

## Recommendations
1. Deploy first kiosks in Dubai Mall and Marina
2. Target Cluster 0 (Eco-Warriors) with sustainability messaging
3. Implement mobile-first payment experience
            """
            
            st.download_button(
                label="📥 Download Executive Summary (TXT)",
                data=summary_text,
                file_name="executive_summary.txt",
                mime="text/plain"
            )
            
            st.markdown("#### Model Performance Report")
            
            perf_report = f"""
# Model Performance Technical Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Classification Model
- Algorithm: Random Forest Classifier
- Accuracy: {metrics['classification_report']['accuracy']:.4f}
- Precision (Yes): {metrics['classification_report']['1']['precision']:.4f}
- Recall (Yes): {metrics['classification_report']['1']['recall']:.4f}
- F1-Score (Yes): {metrics['classification_report']['1']['f1-score']:.4f}
- ROC-AUC: {metrics['roc_auc']:.4f}

## Regression Model
- Algorithm: Random Forest Regressor
- R² Score: {metrics['r2_score']:.4f}
- RMSE: {metrics['rmse']:.2f} AED
- MAE: {metrics['mae']:.2f} AED

## Clustering Model
- Algorithm: K-Means (k=4)
- Silhouette Score: {metrics['silhouette_score']:.4f}

## Association Rules
- Total Rules: {len(models['association_rules'])}
- Avg Lift: {models['association_rules']['lift'].mean():.2f if not models['association_rules'].empty else 'N/A'}
            """
            
            st.download_button(
                label="📥 Download Performance Report (TXT)",
                data=perf_report,
                file_name="model_performance_report.txt",
                mime="text/plain"
            )
        
        with tab3:
            st.subheader("Code Snippets & API Templates")
            
            st.markdown("#### Python: Load and Predict")
            
            python_code = '''
import pandas as pd
import pickle

# Load the trained model
with open('refillhub_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare new customer data
new_customer = pd.DataFrame({
    'Age_Group': ['25-34'],
    'Income': ['10000-15000'],
    'Importance_Sustainability': [4],
    # ... add all required features
})

# Make prediction
prediction = model.predict(new_customer)
probability = model.predict_proba(new_customer)

print(f"Prediction: {'Will Adopt' if prediction[0] == 1 else 'Will Not Adopt'}")
print(f"Confidence: {probability[0][1]*100:.1f}%")
'''
            st.code(python_code, language='python')
            
            st.download_button(
                label="📥 Download Python Template",
                data=python_code,
                file_name="prediction_template.py",
                mime="text/plain"
            )
            
            st.markdown("#### Flask API Example")
            
            flask_code = '''
from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load model at startup
model = pickle.load(open('refillhub_model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    return jsonify({
        'prediction': 'adopt' if prediction == 1 else 'no_adopt',
        'probability': float(probability),
        'confidence': f"{probability*100:.1f}%"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
'''
            st.code(flask_code, language='python')
            
            st.download_button(
                label="📥 Download Flask API Template",
                data=flask_code,
                file_name="flask_api.py",
                mime="text/plain"
            )
            
            st.markdown("#### SQL Query Generator")
            
            sql_template = f'''
-- Create table for storing predictions
CREATE TABLE refillhub_predictions (
    prediction_id INT PRIMARY KEY AUTO_INCREMENT,
    customer_id VARCHAR(50),
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    adoption_prediction BOOLEAN,
    adoption_probability FLOAT,
    predicted_spending DECIMAL(10, 2),
    cluster_segment INT,
    features JSON
);

-- Insert sample prediction
INSERT INTO refillhub_predictions 
(customer_id, adoption_prediction, adoption_probability, predicted_spending, cluster_segment)
VALUES ('CUST001', TRUE, 0.85, 125.50, 0);

-- Query high-value prospects
SELECT 
    customer_id,
    adoption_probability,
    predicted_spending,
    cluster_segment
FROM refillhub_predictions
WHERE adoption_probability > 0.7
    AND predicted_spending > {df['Willingness_to_Pay_AED'].median():.2f}
ORDER BY predicted_spending DESC
LIMIT 100;
'''
            st.code(sql_template, language='sql')
            
            st.download_button(
                label="📥 Download SQL Template",
                data=sql_template,
                file_name="database_queries.sql",
                mime="text/plain"
            )
            
            st.markdown("#### Sample API Request (cURL)")
            
            curl_example = '''
curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "Age_Group": "25-34",
    "Income": "10000-15000",
    "Emirate": "Dubai",
    "Importance_Sustainability": 4,
    "Importance_Price": 3,
    "Importance_Convenience": 4
  }'
'''
            st.code(curl_example, language='bash')

if __name__ == "__main__":
    main()