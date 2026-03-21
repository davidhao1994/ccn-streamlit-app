"""
CCN Prediction Application with Lognormal Mode Fitting
=======================================================

A user-friendly web application for predicting Cloud Condensation Nuclei (CCN) 
concentrations using machine learning and lognormal mode fitting.

Features:
- Data upload and preview
- Automatic lognormal mode fitting
- Multiple ML models (Random Forest, XGBoost)
- Interactive visualizations
- Model performance metrics
- Feature importance analysis
- Export predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import io
import re

# Scipy for optimization
from scipy.optimize import curve_fit
from scipy.stats import lognorm

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

# Settings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Page config
st.set_page_config(
    page_title="CCN Prediction App",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2ca02c;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)


# ========== Helper Functions ==========

def extract_diameter(col_name):
    """Extract diameter value (in nm) from column name."""
    match = re.search(r'Particle_Size_(\d+)_(\d+)nm', col_name, re.IGNORECASE)
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")
    
    match = re.search(r'(\d+\.?\d*)\s*nm', col_name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    
    match = re.search(r'(\d+\.?\d*)', col_name)
    if match:
        return float(match.group(1))
    
    return None


def lognormal_distribution(Dp, N, Dpg, sigma_g):
    """Lognormal distribution for aerosol size distribution."""
    log_sigma = np.log(sigma_g)
    return (N / (np.sqrt(2 * np.pi) * log_sigma)) * \
           np.exp(-(np.log(Dp) - np.log(Dpg))**2 / (2 * log_sigma**2))


def three_mode_lognormal(Dp, N1, Dpg1, sigma_g1, N2, Dpg2, sigma_g2, N3, Dpg3, sigma_g3):
    """Sum of three lognormal modes."""
    return (lognormal_distribution(Dp, N1, Dpg1, sigma_g1) +
            lognormal_distribution(Dp, N2, Dpg2, sigma_g2) +
            lognormal_distribution(Dp, N3, Dpg3, sigma_g3))


def fit_three_modes_optimized(diameters, concentrations, maxfev=500):
    """
    Optimized version of three-mode lognormal fitting.
    
    Parameters:
    -----------
    diameters : array
        Particle diameters (nm)
    concentrations : array
        Particle concentrations (dN/dlogDp)
    maxfev : int
        Maximum function evaluations (default: 500, reduced from 5000)
    
    Returns:
    --------
    dict : Fitted parameters or None if failed
    """
    # Skip if insufficient data
    if len(diameters) < 9 or np.sum(concentrations > 0) < 5:
        return None
    
    # Quick quality check
    if np.max(concentrations) < 10 or np.std(concentrations) < 1:
        return None
    
    # Initial guess optimization
    max_conc = np.max(concentrations)
    
    # Find peaks for better initial guesses
    peak_idx = np.argmax(concentrations)
    peak_dp = diameters[peak_idx]
    
    # Smart initial guesses based on data
    if peak_dp < 30:
        # Nucleation mode dominant
        initial_guess = [
            max_conc * 0.7, 15, 1.5,    # Nucleation
            max_conc * 0.2, 50, 1.6,    # Aitken
            max_conc * 0.1, 150, 1.8    # Accumulation
        ]
    elif peak_dp < 100:
        # Aitken mode dominant
        initial_guess = [
            max_conc * 0.2, 15, 1.5,    # Nucleation
            max_conc * 0.6, 50, 1.6,    # Aitken
            max_conc * 0.2, 150, 1.8    # Accumulation
        ]
    else:
        # Accumulation mode dominant
        initial_guess = [
            max_conc * 0.1, 15, 1.5,    # Nucleation
            max_conc * 0.2, 50, 1.6,    # Aitken
            max_conc * 0.7, 150, 1.8    # Accumulation
        ]
    
    # Tighter bounds for faster convergence
    lower_bounds = [0, 5, 1.2,    # Nucleation
                    0, 25, 1.2,   # Aitken
                    0, 80, 1.2]   # Accumulation
    upper_bounds = [np.inf, 30, 3.0,     # Nucleation
                    np.inf, 100, 3.0,    # Aitken
                    np.inf, 500, 3.0]    # Accumulation
    
    try:
        # Optimized fitting with reduced maxfev
        popt, pcov = curve_fit(
            three_mode_lognormal,
            diameters,
            concentrations,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=maxfev,  # Reduced from 5000
            ftol=1e-4,      # Slightly relaxed tolerance
            xtol=1e-4       # Slightly relaxed tolerance
        )
        
        # Calculate integrated concentrations
        N_nuc = popt[0] * np.log(30/5)
        N_ait = popt[3] * np.log(100/30)
        N_acc = popt[6] * np.log(500/100)
        
        return {
            'N_nucleation': N_nuc,
            'Dpg_nucleation': popt[1],
            'sigma_g_nucleation': popt[2],
            'N_Aitken': N_ait,
            'Dpg_Aitken': popt[4],
            'sigma_g_Aitken': popt[5],
            'N_accumulation': N_acc,
            'Dpg_accumulation': popt[7],
            'sigma_g_accumulation': popt[8],
            'fit_params': popt
        }
    except:
        return None


def calculate_metrics(y_true, y_pred):
    """Calculate model performance metrics."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE (%)': mape
    }


# ========== Main Application ==========

def main():
    # Header
    st.markdown('<h1 class="main-header">☁️ CCN Prediction Application</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <b>About:</b> This application predicts Cloud Condensation Nuclei (CCN) concentrations 
    using machine learning models trained on aerosol size distributions fitted with lognormal modes.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # File upload
        st.subheader("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload a CSV file containing aerosol and meteorological data"
        )
        
        # Use default file if nothing uploaded
        if uploaded_file is None:
            default_file = '/Users/star/Desktop/NPF-2/NPF_with_CCN_merged2.csv'
            if Path(default_file).exists():
                st.info(f"Using default file:\n`{Path(default_file).name}`")
                use_default = st.checkbox("Load default dataset", value=True)
            else:
                use_default = False
                st.warning("No file uploaded. Please upload a CSV file.")
        else:
            use_default = False
        
        st.divider()
        
        # Model settings
        st.subheader("🤖 Model Settings")
        
        model_type = st.selectbox(
            "Select Model",
            ["Random Forest", "XGBoost", "Gradient Boosting"],
            help="Choose the machine learning model for prediction"
        )
        
        test_size = st.slider(
            "Test Size (%)",
            min_value=10,
            max_value=40,
            value=20,
            step=5,
            help="Percentage of data to use for testing"
        )
        
        random_state = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=1000,
            value=42,
            help="Random seed for reproducibility"
        )
        
        st.divider()
        
        # Fitting settings
        st.subheader("📊 Fitting Settings")
        
        maxfev = st.slider(
            "Max Function Evaluations",
            min_value=100,
            max_value=2000,
            value=500,
            step=100,
            help="Maximum number of function evaluations for lognormal fitting"
        )
        
        min_diameter = st.number_input(
            "Min Diameter (nm)",
            min_value=0.0,
            max_value=100.0,
            value=66.5,
            step=0.5,
            help="Minimum particle diameter to include"
        )
    
    # Main content
    if uploaded_file is not None or use_default:
        # Load data
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv('/Users/star/Desktop/NPF-2/NPF_with_CCN_merged2.csv')
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data Overview",
            "🔬 Lognormal Fitting",
            "🤖 Model Training",
            "📈 Results & Metrics",
            "💾 Export"
        ])
        
        # ===== Tab 1: Data Overview =====
        with tab1:
            st.markdown('<h2 class="sub-header">Data Overview</h2>', 
                       unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                if 'N_CCN' in df.columns:
                    st.metric("Target Column", "N_CCN ✓")
                else:
                    st.metric("Target Column", "Not Found ⚠️")
            
            st.subheader("Dataset Preview")
            st.dataframe(df.head(20), use_container_width=True)
            
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Non-Null': df.count().values,
                'Null %': ((df.isnull().sum() / len(df)) * 100).values.round(2)
            })
            st.dataframe(col_info, use_container_width=True, height=400)
            
            # Identify particle size columns
            particle_cols = [col for col in df.columns if 'Particle_Size' in col or 'particle' in col.lower()]
            
            st.subheader("Particle Size Columns")
            st.write(f"Found **{len(particle_cols)}** particle size columns")
            
            if len(particle_cols) > 0:
                diameter_map = {}
                for col in particle_cols:
                    dp = extract_diameter(col)
                    if dp is not None and dp > min_diameter:
                        diameter_map[col] = dp
                
                if len(diameter_map) > 0:
                    size_bins = pd.DataFrame([
                        {'Column': col, 'Diameter (nm)': dp} 
                        for col, dp in diameter_map.items()
                    ]).sort_values('Diameter (nm)')
                    
                    st.write(f"**{len(size_bins)}** bins > {min_diameter} nm")
                    st.dataframe(size_bins, use_container_width=True, height=300)
                    
                    # Store in session state
                    st.session_state['size_bins'] = size_bins
                    st.session_state['diameter_map'] = diameter_map
                else:
                    st.warning(f"No particle size columns found > {min_diameter} nm")
        
        # ===== Tab 2: Lognormal Fitting =====
        with tab2:
            st.markdown('<h2 class="sub-header">Lognormal Mode Fitting</h2>', 
                       unsafe_allow_html=True)
            
            if 'size_bins' not in st.session_state:
                st.warning("Please view the Data Overview tab first to identify particle size columns.")
            else:
                size_bins = st.session_state['size_bins']
                diameter_map = st.session_state['diameter_map']
                
                st.info(f"""
                **Fitting Process:**
                - Using {len(size_bins)} particle size bins
                - Diameter range: {size_bins['Diameter (nm)'].min():.1f} - {size_bins['Diameter (nm)'].max():.1f} nm
                - Max function evaluations: {maxfev}
                """)
                
                if st.button("🚀 Start Lognormal Fitting", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Prepare data
                    size_cols = size_bins['Column'].tolist()
                    diameters = size_bins['Diameter (nm)'].values
                    
                    # Initialize results
                    mode_features = []
                    success_count = 0
                    
                    # Fit each row
                    total_rows = len(df)
                    for idx, row in df.iterrows():
                        if idx % 100 == 0:
                            progress = idx / total_rows
                            progress_bar.progress(progress)
                            status_text.text(f"Processing: {idx:,} / {total_rows:,} rows ({progress*100:.1f}%)")
                        
                        concentrations = row[size_cols].values
                        
                        # Fit
                        fit_result = fit_three_modes_optimized(diameters, concentrations, maxfev=maxfev)
                        
                        if fit_result is not None:
                            mode_features.append(fit_result)
                            success_count += 1
                        else:
                            mode_features.append({
                                'N_nucleation': np.nan,
                                'Dpg_nucleation': np.nan,
                                'sigma_g_nucleation': np.nan,
                                'N_Aitken': np.nan,
                                'Dpg_Aitken': np.nan,
                                'sigma_g_Aitken': np.nan,
                                'N_accumulation': np.nan,
                                'Dpg_accumulation': np.nan,
                                'sigma_g_accumulation': np.nan
                            })
                    
                    progress_bar.progress(1.0)
                    status_text.text(f"✅ Complete! Successfully fitted {success_count:,} / {total_rows:,} rows")
                    
                    # Create DataFrame with mode features
                    mode_df = pd.DataFrame(mode_features)
                    
                    # Combine with original data
                    df_with_modes = pd.concat([df.reset_index(drop=True), mode_df], axis=1)
                    
                    # Store in session state
                    st.session_state['df_with_modes'] = df_with_modes
                    st.session_state['success_count'] = success_count
                    
                    st.success(f"✅ Fitting complete! Success rate: {success_count/total_rows*100:.1f}%")
                    
                    # Show sample results
                    st.subheader("Sample Fitted Parameters")
                    st.dataframe(mode_df.head(10), use_container_width=True)
                    
                    # Visualize mode distributions
                    st.subheader("Mode Concentration Distributions")
                    
                    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                    
                    mode_df_clean = mode_df.dropna()
                    
                    if len(mode_df_clean) > 0:
                        axes[0].hist(mode_df_clean['N_nucleation'], bins=50, color='steelblue', alpha=0.7)
                        axes[0].set_xlabel('N_nucleation (cm⁻³)')
                        axes[0].set_ylabel('Frequency')
                        axes[0].set_title('Nucleation Mode')
                        axes[0].grid(True, alpha=0.3)
                        
                        axes[1].hist(mode_df_clean['N_Aitken'], bins=50, color='forestgreen', alpha=0.7)
                        axes[1].set_xlabel('N_Aitken (cm⁻³)')
                        axes[1].set_ylabel('Frequency')
                        axes[1].set_title('Aitken Mode')
                        axes[1].grid(True, alpha=0.3)
                        
                        axes[2].hist(mode_df_clean['N_accumulation'], bins=50, color='coral', alpha=0.7)
                        axes[2].set_xlabel('N_accumulation (cm⁻³)')
                        axes[2].set_ylabel('Frequency')
                        axes[2].set_title('Accumulation Mode')
                        axes[2].grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
        
        # ===== Tab 3: Model Training =====
        with tab3:
            st.markdown('<h2 class="sub-header">Model Training</h2>', 
                       unsafe_allow_html=True)
            
            if 'df_with_modes' not in st.session_state:
                st.warning("Please complete the Lognormal Fitting step first.")
            else:
                df_with_modes = st.session_state['df_with_modes']
                
                st.info(f"""
                **Training Configuration:**
                - Model: {model_type}
                - Test size: {test_size}%
                - Random seed: {random_state}
                """)
                
                # Feature selection
                st.subheader("Feature Selection")
                
                mode_features = ['N_nucleation', 'N_Aitken', 'N_accumulation']
                size_features = [col for col in df_with_modes.columns if 'Particle_Size' in col]
                met_features = ['Temperature', 'RH', 'Wind_Speed', 'Pressure']
                chem_features = ['SO2', 'NOx', 'O3', 'Organics', 'Sulfate']
                
                # Filter available features
                available_mode = [f for f in mode_features if f in df_with_modes.columns]
                available_met = [f for f in met_features if f in df_with_modes.columns]
                available_chem = [f for f in chem_features if f in df_with_modes.columns]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    use_modes = st.checkbox("Use Lognormal Modes", value=True)
                    use_met = st.checkbox("Use Meteorological Features", value=True)
                
                with col2:
                    use_size = st.checkbox("Use Size Distribution", value=False)
                    use_chem = st.checkbox("Use Chemical Features", value=True)
                
                if st.button("🎯 Train Model", type="primary"):
                    # Prepare features
                    feature_list = []
                    
                    if use_modes:
                        feature_list.extend(available_mode)
                    if use_size:
                        feature_list.extend(size_features)
                    if use_met:
                        feature_list.extend(available_met)
                    if use_chem:
                        feature_list.extend(available_chem)
                    
                    # Check target
                    if 'N_CCN' not in df_with_modes.columns:
                        st.error("Target column 'N_CCN' not found!")
                        st.stop()
                    
                    # Prepare data
                    available_features = [f for f in feature_list if f in df_with_modes.columns]
                    
                    st.write(f"Selected **{len(available_features)}** features")
                    
                    # Remove rows with missing values
                    df_clean = df_with_modes[available_features + ['N_CCN']].dropna()
                    
                    st.write(f"Clean dataset: **{len(df_clean):,}** rows")
                    
                    if len(df_clean) < 100:
                        st.error("Insufficient data after removing missing values!")
                        st.stop()
                    
                    X = df_clean[available_features]
                    y = df_clean['N_CCN']
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y,
                        test_size=test_size/100,
                        random_state=random_state
                    )
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Train model
                    with st.spinner(f"Training {model_type}..."):
                        if model_type == "Random Forest":
                            model = RandomForestRegressor(
                                n_estimators=100,
                                max_depth=20,
                                random_state=random_state,
                                n_jobs=-1
                            )
                        elif model_type == "XGBoost" and XGB_AVAILABLE:
                            model = xgb.XGBRegressor(
                                n_estimators=100,
                                max_depth=10,
                                learning_rate=0.1,
                                random_state=random_state,
                                n_jobs=-1
                            )
                        else:
                            model = GradientBoostingRegressor(
                                n_estimators=100,
                                max_depth=10,
                                learning_rate=0.1,
                                random_state=random_state
                            )
                        
                        model.fit(X_train_scaled, y_train)
                    
                    # Make predictions
                    y_train_pred = model.predict(X_train_scaled)
                    y_test_pred = model.predict(X_test_scaled)
                    
                    # Calculate metrics
                    train_metrics = calculate_metrics(y_train, y_train_pred)
                    test_metrics = calculate_metrics(y_test, y_test_pred)
                    
                    # Store in session state
                    st.session_state['model'] = model
                    st.session_state['scaler'] = scaler
                    st.session_state['features'] = available_features
                    st.session_state['X_test'] = X_test
                    st.session_state['y_test'] = y_test
                    st.session_state['y_test_pred'] = y_test_pred
                    st.session_state['train_metrics'] = train_metrics
                    st.session_state['test_metrics'] = test_metrics
                    
                    st.success("✅ Model training complete!")
                    
                    # Display metrics
                    st.subheader("Model Performance")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Training Set**")
                        for metric, value in train_metrics.items():
                            st.metric(metric, f"{value:.4f}")
                    
                    with col2:
                        st.markdown("**Test Set**")
                        for metric, value in test_metrics.items():
                            st.metric(metric, f"{value:.4f}")
        
        # ===== Tab 4: Results & Metrics =====
        with tab4:
            st.markdown('<h2 class="sub-header">Results & Metrics</h2>', 
                       unsafe_allow_html=True)
            
            if 'model' not in st.session_state:
                st.warning("Please train a model first.")
            else:
                model = st.session_state['model']
                y_test = st.session_state['y_test']
                y_test_pred = st.session_state['y_test_pred']
                test_metrics = st.session_state['test_metrics']
                features = st.session_state['features']
                
                # Prediction plot
                st.subheader("Predicted vs Observed CCN")
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                ax.scatter(y_test, y_test_pred, alpha=0.5, s=20)
                
                # 1:1 line
                min_val = min(y_test.min(), y_test_pred.min())
                max_val = max(y_test.max(), y_test_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line')
                
                ax.set_xlabel('Observed N_CCN (cm⁻³)', fontsize=12)
                ax.set_ylabel('Predicted N_CCN (cm⁻³)', fontsize=12)
                ax.set_title(f'Predicted vs Observed CCN\nR² = {test_metrics["R²"]:.4f}', fontsize=14)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Feature importance
                st.subheader("Feature Importance")
                
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    top_n = min(20, len(importance_df))
                    importance_df.head(top_n).plot(
                        x='Feature',
                        y='Importance',
                        kind='barh',
                        ax=ax,
                        color='steelblue'
                    )
                    
                    ax.set_xlabel('Importance', fontsize=12)
                    ax.set_ylabel('Feature', fontsize=12)
                    ax.set_title(f'Top {top_n} Most Important Features', fontsize=14)
                    ax.legend().remove()
                    
                    st.pyplot(fig)
                    
                    st.dataframe(importance_df, use_container_width=True)
        
        # ===== Tab 5: Export =====
        with tab5:
            st.markdown('<h2 class="sub-header">Export Results</h2>', 
                       unsafe_allow_html=True)
            
            if 'df_with_modes' in st.session_state:
                df_export = st.session_state['df_with_modes']
                
                st.subheader("Download Fitted Data")
                
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV with Lognormal Modes",
                    data=csv,
                    file_name="ccn_data_with_modes.csv",
                    mime="text/csv"
                )
            
            if 'model' in st.session_state:
                st.subheader("Download Predictions")
                
                y_test = st.session_state['y_test']
                y_test_pred = st.session_state['y_test_pred']
                
                pred_df = pd.DataFrame({
                    'Observed_N_CCN': y_test.values,
                    'Predicted_N_CCN': y_test_pred
                })
                
                csv_pred = pred_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv_pred,
                    file_name="ccn_predictions.csv",
                    mime="text/csv"
                )
                
                st.subheader("Model Summary")
                
                test_metrics = st.session_state['test_metrics']
                
                summary = f"""
# CCN Prediction Model Summary

## Model Configuration
- Model Type: {model_type}
- Test Size: {test_size}%
- Random Seed: {random_state}
- Features: {len(st.session_state['features'])}

## Performance Metrics
- R²: {test_metrics['R²']:.4f}
- RMSE: {test_metrics['RMSE']:.4f}
- MAE: {test_metrics['MAE']:.4f}
- MAPE: {test_metrics['MAPE (%)']:.2f}%

## Dataset
- Training Samples: {len(st.session_state['X_test']) * (100-test_size) // test_size}
- Test Samples: {len(st.session_state['X_test'])}
"""
                
                st.download_button(
                    label="📥 Download Summary",
                    data=summary,
                    file_name="model_summary.txt",
                    mime="text/plain"
                )
    
    else:
        st.info("👈 Please upload a CSV file or select the default dataset from the sidebar to begin.")


if __name__ == "__main__":
    main()
