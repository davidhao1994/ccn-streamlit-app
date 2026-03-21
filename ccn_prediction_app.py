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
        
        st.divider()
        
        # Data size settings
        st.subheader("📏 Data Size")
        
        use_sample = st.checkbox(
            "Limit number of rows",
            value=False,
            help="Enable to process only a subset of data for faster testing"
        )
        
        if use_sample:
            sample_size = st.number_input(
                "Number of rows to process",
                min_value=100,
                max_value=1000000,
                value=10000,
                step=1000,
                help="Number of rows to use from the dataset"
            )
        else:
            sample_size = None
        
        sample_method = st.radio(
            "Sampling method",
            ["First N rows", "Random sample", "Last N rows"],
            help="How to select the subset of data"
        ) if use_sample else "First N rows"
    
    # Main content
    if uploaded_file is not None or use_default:
        # Load data
        if uploaded_file is not None:
            df_full = pd.read_csv(uploaded_file)
        else:
            df_full = pd.read_csv('/Users/star/Desktop/NPF-2/NPF_with_CCN_merged2.csv')
        
        # Apply row limit if enabled
        total_rows = len(df_full)
        
        if use_sample and sample_size is not None:
            if sample_method == "First N rows":
                df = df_full.head(sample_size).copy()
            elif sample_method == "Random sample":
                if sample_size < total_rows:
                    df = df_full.sample(n=sample_size, random_state=random_state).copy()
                else:
                    df = df_full.copy()
            else:  # Last N rows
                df = df_full.tail(sample_size).copy()
            
            st.info(f"📊 Using {len(df):,} rows out of {total_rows:,} total rows ({len(df)/total_rows*100:.1f}%)")
        else:
            df = df_full.copy()
            st.info(f"📊 Using all {total_rows:,} rows")
        
        # Create tabs
        tab1, tab1_5, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data Overview",
            "📈 Size Distribution Visualization",
            "🔬 Lognormal Fitting",
            "🤖 Model Training",
            "📈 Results & Metrics",
            "💾 Export"
        ])
        
        # ===== Tab 1: Data Overview =====
        with tab1:
            st.markdown('<h2 class="sub-header">Data Overview</h2>', 
                       unsafe_allow_html=True)
            
            # Display dataset info
            if use_sample and sample_size is not None:
                st.success(f"✅ Loaded {len(df):,} rows ({sample_method}) from {total_rows:,} total rows")
            else:
                st.success(f"✅ Loaded all {len(df):,} rows")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows (Current)", f"{len(df):,}")
            with col2:
                st.metric("Rows (Total)", f"{total_rows:,}")
            with col3:
                st.metric("Total Columns", len(df.columns))
            with col4:
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
        
        # ===== Tab 1.5: Size Distribution Visualization =====
        with tab1_5:
            st.markdown('<h2 class="sub-header">📈 Particle Size Distribution Visualization</h2>', 
                       unsafe_allow_html=True)
            
            if 'size_bins' not in st.session_state:
                st.warning("⚠️ Please view the **Data Overview** tab first to identify particle size columns.")
            else:
                size_bins = st.session_state['size_bins']
                diameter_map = st.session_state['diameter_map']
                
                st.info(f"""
                **Available Data:**
                - Total timestamps: **{len(df):,}** rows
                - Particle size bins: **{len(size_bins)}** bins
                - Diameter range: **{size_bins['Diameter (nm)'].min():.1f} - {size_bins['Diameter (nm)'].max():.1f} nm**
                """)
                
                # Visualization settings
                st.markdown("### ⚙️ Visualization Settings")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎯 Sample Selection")
                    
                    # Number of samples to plot
                    max_samples = min(len(df), 100)  # Limit to 100 for performance
                    num_samples = st.slider(
                        "Number of samples to plot",
                        min_value=1,
                        max_value=max_samples,
                        value=min(10, max_samples),
                        step=1,
                        help=f"Select up to {max_samples} samples to visualize"
                    )
                    
                    # Sample selection method
                    sample_selection = st.radio(
                        "Sample selection method",
                        ["First N samples", "Random samples", "Last N samples", "Specific indices"],
                        help="How to select which size distributions to plot"
                    )
                    
                    if sample_selection == "Specific indices":
                        indices_input = st.text_input(
                            "Enter row indices (comma-separated)",
                            value="0,100,200",
                            help="e.g., 0,100,200,500"
                        )
                
                with col2:
                    st.subheader("📏 Diameter Range")
                    
                    # Get all available diameters
                    all_diameters = sorted([diameter_map[col] for col in size_bins['Column'].values])
                    min_available = all_diameters[0]
                    max_available = all_diameters[-1]
                    
                    # Diameter range selection
                    diameter_range = st.slider(
                        "Diameter range (nm)",
                        min_value=float(min_available),
                        max_value=float(max_available),
                        value=(float(min_available), float(max_available)),
                        step=1.0,
                        help="Select the diameter range to display"
                    )
                    
                    # Y-axis scale
                    y_scale = st.radio(
                        "Y-axis scale",
                        ["Linear", "Logarithmic"],
                        help="Linear or logarithmic scale for concentration"
                    )
                    
                    # X-axis scale
                    x_scale = st.radio(
                        "X-axis scale",
                        ["Linear", "Logarithmic"],
                        index=1,  # Default to log
                        help="Linear or logarithmic scale for diameter"
                    )
                
                # Plot style options
                with st.expander("🎨 Plot Style Options"):
                    col_style1, col_style2 = st.columns(2)
                    
                    with col_style1:
                        plot_style = st.selectbox(
                            "Plot style",
                            ["Lines", "Lines + Markers", "Markers only"],
                            help="How to display the data points"
                        )
                        
                        colormap = st.selectbox(
                            "Color scheme",
                            ["viridis", "rainbow", "jet", "coolwarm", "plasma", "hsv"],
                            help="Color scheme for different samples"
                        )
                    
                    with col_style2:
                        line_alpha = st.slider(
                            "Line transparency",
                            min_value=0.1,
                            max_value=1.0,
                            value=0.7,
                            step=0.1,
                            help="Transparency of plot lines (lower = more transparent)"
                        )
                        
                        show_legend = st.checkbox(
                            "Show legend",
                            value=True,
                            help="Display legend with sample information"
                        )
                
                # Generate plot button
                st.divider()
                
                if st.button("📊 Generate Size Distribution Plot", type="primary", use_container_width=True):
                    
                    # Filter columns within diameter range
                    filtered_cols = [col for col, dp in diameter_map.items() 
                                    if diameter_range[0] <= dp <= diameter_range[1]]
                    filtered_diameters = sorted([diameter_map[col] for col in filtered_cols])
                    
                    if len(filtered_cols) == 0:
                        st.error("❌ No size bins found in the selected diameter range!")
                        st.stop()
                    
                    # Select samples
                    if sample_selection == "First N samples":
                        selected_indices = list(range(min(num_samples, len(df))))
                    elif sample_selection == "Random samples":
                        np.random.seed(random_state)
                        selected_indices = np.random.choice(len(df), size=min(num_samples, len(df)), replace=False)
                        selected_indices = sorted(selected_indices)
                    elif sample_selection == "Last N samples":
                        selected_indices = list(range(max(0, len(df) - num_samples), len(df)))
                    else:  # Specific indices
                        try:
                            selected_indices = [int(idx.strip()) for idx in indices_input.split(',')]
                            selected_indices = [idx for idx in selected_indices if 0 <= idx < len(df)]
                            if len(selected_indices) == 0:
                                st.error("❌ No valid indices found!")
                                st.stop()
                        except:
                            st.error("❌ Invalid indices format! Use comma-separated numbers (e.g., 0,100,200)")
                            st.stop()
                    
                    st.success(f"✅ Plotting {len(selected_indices)} size distributions with {len(filtered_cols)} size bins")
                    
                    # Create plot
                    fig, ax = plt.subplots(figsize=(14, 8))
                    
                    # Get colormap
                    colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(selected_indices)))
                    
                    # Plot each selected sample
                    for idx, sample_idx in enumerate(selected_indices):
                        concentrations = df.iloc[sample_idx][filtered_cols].values
                        
                        # Determine plot style
                        if plot_style == "Lines":
                            ax.plot(filtered_diameters, concentrations, 
                                   color=colors[idx], alpha=line_alpha, linewidth=1.5,
                                   label=f"Sample {sample_idx}")
                        elif plot_style == "Lines + Markers":
                            ax.plot(filtered_diameters, concentrations, 
                                   marker='o', markersize=4, color=colors[idx], 
                                   alpha=line_alpha, linewidth=1.5,
                                   label=f"Sample {sample_idx}")
                        else:  # Markers only
                            ax.scatter(filtered_diameters, concentrations, 
                                      color=colors[idx], alpha=line_alpha, s=30,
                                      label=f"Sample {sample_idx}")
                    
                    # Set scales
                    if x_scale == "Logarithmic":
                        ax.set_xscale('log')
                    if y_scale == "Logarithmic":
                        ax.set_yscale('log')
                    
                    # Labels and title
                    ax.set_xlabel('Particle Diameter (nm)', fontsize=13, fontweight='bold')
                    ax.set_ylabel('dN/dlogDp (cm⁻³)', fontsize=13, fontweight='bold')
                    ax.set_title(f'Particle Size Distribution\n({len(selected_indices)} samples, Dp: {diameter_range[0]:.1f}-{diameter_range[1]:.1f} nm)', 
                                fontsize=14, fontweight='bold', pad=15)
                    
                    # Grid
                    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                    
                    # Legend
                    if show_legend and len(selected_indices) <= 20:
                        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                                 fontsize=9, framealpha=0.9)
                    elif show_legend:
                        st.info("ℹ️ Legend hidden (too many samples). Sample info shown below.")
                    
                    plt.tight_layout()
                    
                    # Display plot
                    st.pyplot(fig)
                    
                    # Display statistics
                    st.markdown("### 📊 Statistics")
                    
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    
                    # Calculate statistics across selected samples
                    all_concentrations = df.iloc[selected_indices][filtered_cols].values
                    
                    with col_stat1:
                        mean_total = np.nanmean(np.sum(all_concentrations, axis=1))
                        st.metric("Mean Total N", f"{mean_total:.1f} cm⁻³")
                    
                    with col_stat2:
                        max_conc = np.nanmax(all_concentrations)
                        st.metric("Max Concentration", f"{max_conc:.1f} cm⁻³")
                    
                    with col_stat3:
                        # Find diameter of max concentration (on average)
                        mean_dist = np.nanmean(all_concentrations, axis=0)
                        peak_idx = np.argmax(mean_dist)
                        peak_dp = filtered_diameters[peak_idx]
                        st.metric("Peak Diameter (avg)", f"{peak_dp:.1f} nm")
                    
                    with col_stat4:
                        st.metric("Size Bins Plotted", len(filtered_cols))
                    
                    # Sample information table
                    if len(selected_indices) <= 50:
                        with st.expander("📋 View Sample Details"):
                            sample_info = []
                            for sample_idx in selected_indices:
                                sample_data = df.iloc[sample_idx][filtered_cols].values
                                sample_info.append({
                                    'Row Index': sample_idx,
                                    'Total N (cm⁻³)': f"{np.nansum(sample_data):.1f}",
                                    'Max dN/dlogDp': f"{np.nanmax(sample_data):.1f}",
                                    'Peak Dp (nm)': f"{filtered_diameters[np.argmax(sample_data)]:.1f}"
                                })
                            
                            sample_df = pd.DataFrame(sample_info)
                            st.dataframe(sample_df, use_container_width=True, height=min(400, len(sample_info) * 35 + 38))
                    
                    # Download data button
                    st.markdown("### 💾 Export Data")
                    
                    # Prepare export data
                    export_data = pd.DataFrame(
                        all_concentrations,
                        columns=[f"{dp:.2f}nm" for dp in filtered_diameters],
                        index=[f"Sample_{idx}" for idx in selected_indices]
                    )
                    export_data.insert(0, 'Row_Index', selected_indices)
                    
                    csv_export = export_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Plotted Data (CSV)",
                        data=csv_export,
                        file_name="size_distribution_data.csv",
                        mime="text/csv"
                    )
        
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
                    for row_num, (idx, row) in enumerate(df.iterrows()):
                        if row_num % 100 == 0:
                            progress = min(row_num / total_rows, 1.0)  # Ensure progress is <= 1.0
                            progress_bar.progress(progress)
                            status_text.text(f"Processing: {row_num:,} / {total_rows:,} rows ({progress*100:.1f}%)")
                        
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
                st.subheader("🎯 Feature Selection")
                
                # Define all feature categories with specific patterns
                
                # 1. Lognormal Modes (fitted features)
                mode_features = ['N_nucleation', 'N_Aitken', 'N_accumulation']

                # 2. Particle Size Distribution (raw SMPS/NanoSMPS data)
                size_features = [col for col in df_with_modes.columns if 'Particle_Size' in col]
                
                # 3. Aerosol Statistics (geometric mean, total concentration, etc.)
                aerosol_stat_patterns = ['geometric_mean', 'total_n_conc', 'arithmetic_mean', 
                                        'percentage_below', 'mode_interp', 'total_sa_conc']
                aerosol_stat_features = [col for col in df_with_modes.columns 
                                        if any(pattern in col.lower() for pattern in aerosol_stat_patterns)]
                
                # 4. Meteorological Features
                met_patterns = ['temperature', 'temp', 'rh', 'humidity', 'wind_direction', 
                               'wind_speed', 'wind', 'pressure', 'precip', 'bl_height']
                met_features = [col for col in df_with_modes.columns 
                               if any(pattern in col.lower() for pattern in met_patterns)
                               and not any(p in col.lower() for p in ['organic', 'sulfate', 'nitrate', 'so2'])]
                
                # 5. Chemical Composition Features
                chem_patterns = ['organic', 'sulfate', 'sulphate', 'nitrate', 'so2', 
                                'nox', 'no2', 'o3', 'ozone', 'nh3', 'nh4', 'ammonium', 'bc', 'pm']
                chem_features = [col for col in df_with_modes.columns 
                                if any(pattern in col.lower() for pattern in chem_patterns)]
                
                # 6. Turbulence & Energy Features
                turbulence_patterns = ['turbulent', 'kinetic_energy', 'friction', 'momentum']
                turbulence_features = [col for col in df_with_modes.columns 
                                      if any(pattern in col.lower() for pattern in turbulence_patterns)]
                
                # 7. Radiation Features
                radiation_patterns = ['flux', 'radiation', 'diffuse', 'hemisp', 'short', 'down', 'up',
                                     'bestestimate', 'solar', 'irradiance']
                radiation_features = [col for col in df_with_modes.columns 
                                     if any(pattern in col.lower() for pattern in radiation_patterns)
                                     and 'co2' not in col.lower()]
                
                # 8. CO2 Flux Features
                co2_patterns = ['co2']
                co2_features = [col for col in df_with_modes.columns 
                               if any(pattern in col.lower() for pattern in co2_patterns)]
                
                # Filter available features and ensure they are numeric
                available_mode = [f for f in mode_features 
                                 if f in df_with_modes.columns and 
                                 pd.api.types.is_numeric_dtype(df_with_modes[f])]
                
                available_aerosol_stats = [f for f in aerosol_stat_features 
                                          if f in df_with_modes.columns and 
                                          pd.api.types.is_numeric_dtype(df_with_modes[f])]
                
                available_met = [f for f in met_features 
                                if f in df_with_modes.columns and 
                                pd.api.types.is_numeric_dtype(df_with_modes[f])]
                
                available_chem = [f for f in chem_features 
                                 if f in df_with_modes.columns and 
                                 pd.api.types.is_numeric_dtype(df_with_modes[f])]
                
                available_turbulence = [f for f in turbulence_features 
                                       if f in df_with_modes.columns and 
                                       pd.api.types.is_numeric_dtype(df_with_modes[f])]
                
                available_radiation = [f for f in radiation_features 
                                      if f in df_with_modes.columns and 
                                      pd.api.types.is_numeric_dtype(df_with_modes[f])]
                
                available_co2 = [f for f in co2_features 
                                if f in df_with_modes.columns and 
                                pd.api.types.is_numeric_dtype(df_with_modes[f])]
                
                # Display available features summary
                st.info(f"""
                **📊 Available Features by Category:**
                
                🔬 **Aerosol Features:**
                - Lognormal Modes: **{len(available_mode)}** features
                - Size Distribution: **{len(size_features)}** features  
                - Aerosol Statistics: **{len(available_aerosol_stats)}** features
                
                🌡️ **Environmental Features:**
                - Meteorological: **{len(available_met)}** features
                - Chemical Composition: **{len(available_chem)}** features
                - Turbulence & Energy: **{len(available_turbulence)}** features
                - Radiation: **{len(available_radiation)}** features
                - CO2 Flux: **{len(available_co2)}** features
                
                📈 **Total Available: {len(available_mode) + len(size_features) + len(available_aerosol_stats) + len(available_met) + len(available_chem) + len(available_turbulence) + len(available_radiation) + len(available_co2)}** features
                """)
                
                # Feature selection UI
                st.markdown("### 🎛️ Select Feature Categories")
                
                # Create expandable sections for each category
                with st.expander("🔬 Aerosol Features", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        use_modes = st.checkbox(
                            f"Lognormal Modes ({len(available_mode)})", 
                            value=True,
                            help="Fitted lognormal mode parameters (N, Dpg, σg for nucleation, Aitken, accumulation modes)"
                        )
                    with col2:
                        use_aerosol_stats = st.checkbox(
                            f"Aerosol Statistics ({len(available_aerosol_stats)})", 
                            value=True,
                            help="Geometric mean, total concentration, arithmetic mean, percentages, surface area"
                        )
                    with col3:
                        use_size = st.checkbox(
                            f"Size Distribution ({len(size_features)})", 
                            value=False,
                            help="Raw particle size bin concentrations (may be many features)"
                        )
                    
                    if use_aerosol_stats and available_aerosol_stats:
                        st.caption(f"📝 Includes: {', '.join(available_aerosol_stats[:5])}{'...' if len(available_aerosol_stats) > 5 else ''}")
                
                with st.expander("🌡️ Environmental Features", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        use_met = st.checkbox(
                            f"Meteorological ({len(available_met)})", 
                            value=True,
                            help="Temperature, humidity, wind, pressure, boundary layer height"
                        )
                        use_turbulence = st.checkbox(
                            f"Turbulence & Energy ({len(available_turbulence)})", 
                            value=False,
                            help="Turbulent kinetic energy, friction velocity, momentum flux"
                        )
                    with col2:
                        use_chem = st.checkbox(
                            f"Chemical Composition ({len(available_chem)})", 
                            value=True,
                            help="SO2, NOx, O3, organics, sulfate, nitrate"
                        )
                        use_radiation = st.checkbox(
                            f"Radiation ({len(available_radiation)})", 
                            value=False,
                            help="Solar radiation, diffuse radiation, shortwave hemispheric"
                        )
                    
                    col3, col4 = st.columns(2)
                    with col3:
                        use_co2 = st.checkbox(
                            f"CO2 Flux ({len(available_co2)})", 
                            value=False,
                            help="CO2 flux measurements"
                        )
                    
                    # Show feature previews
                    if use_met and available_met:
                        st.caption(f"🌡️ Meteorological: {', '.join(available_met[:4])}{'...' if len(available_met) > 4 else ''}")
                    if use_chem and available_chem:
                        st.caption(f"🧪 Chemical: {', '.join(available_chem[:4])}{'...' if len(available_chem) > 4 else ''}")
                
                # Quick selection buttons
                st.markdown("### ⚡ Quick Selection")
                col_quick1, col_quick2, col_quick3, col_quick4 = st.columns(4)
                
                with col_quick1:
                    if st.button("✅ Select All", help="Select all available feature categories"):
                        use_modes = use_aerosol_stats = use_size = True
                        use_met = use_chem = use_turbulence = use_radiation = use_co2 = True
                        st.rerun()
                
                with col_quick2:
                    if st.button("🔬 Aerosol Only", help="Only aerosol-related features"):
                        use_modes = use_aerosol_stats = use_size = True
                        use_met = use_chem = use_turbulence = use_radiation = use_co2 = False
                        st.rerun()
                
                with col_quick3:
                    if st.button("🎯 Recommended", help="Recommended feature set for CCN prediction"):
                        use_modes = use_aerosol_stats = use_met = use_chem = True
                        use_size = use_turbulence = use_radiation = use_co2 = False
                        st.rerun()
                
                with col_quick4:
                    if st.button("⬜ Clear All", help="Deselect all features"):
                        use_modes = use_aerosol_stats = use_size = False
                        use_met = use_chem = use_turbulence = use_radiation = use_co2 = False
                        st.rerun()
                
                st.divider()
                
                if st.button("🎯 Train Model", type="primary", use_container_width=True):
                    # Prepare features
                    feature_list = []
                    feature_categories = []
                    
                    if use_modes and available_mode:
                        feature_list.extend(available_mode)
                        feature_categories.append(f"Lognormal Modes ({len(available_mode)})")
                    if use_aerosol_stats and available_aerosol_stats:
                        feature_list.extend(available_aerosol_stats)
                        feature_categories.append(f"Aerosol Statistics ({len(available_aerosol_stats)})")
                    if use_size and size_features:
                        feature_list.extend(size_features)
                        feature_categories.append(f"Size Distribution ({len(size_features)})")
                    if use_met and available_met:
                        feature_list.extend(available_met)
                        feature_categories.append(f"Meteorological ({len(available_met)})")
                    if use_chem and available_chem:
                        feature_list.extend(available_chem)
                        feature_categories.append(f"Chemical ({len(available_chem)})")
                    if use_turbulence and available_turbulence:
                        feature_list.extend(available_turbulence)
                        feature_categories.append(f"Turbulence ({len(available_turbulence)})")
                    if use_radiation and available_radiation:
                        feature_list.extend(available_radiation)
                        feature_categories.append(f"Radiation ({len(available_radiation)})")
                    if use_co2 and available_co2:
                        feature_list.extend(available_co2)
                        feature_categories.append(f"CO2 Flux ({len(available_co2)})")
                    
                    # Display selected categories
                    st.success(f"✅ Selected {len(feature_categories)} categories: {', '.join(feature_categories)}")
                    
                    # Check if any features selected
                    if len(feature_list) == 0:
                        st.error("⚠️ No features selected! Please select at least one feature category.")
                        st.stop()
                    
                    # Check target
                    if 'N_CCN' not in df_with_modes.columns:
                        st.error("❌ Target column 'N_CCN' not found!")
                        st.stop()
                    
                    # Prepare data
                    available_features = [f for f in feature_list if f in df_with_modes.columns]
                    
                    if len(available_features) == 0:
                        st.error("❌ None of the selected features are available in the dataset!")
                        st.stop()
                    
                    st.write(f"✅ Selected **{len(available_features)}** features")
                    
                    # Show selected features
                    with st.expander("View selected features"):
                        st.write(available_features)
                    
                    # Create a copy and ensure all columns are numeric
                    df_for_training = df_with_modes[available_features + ['N_CCN']].copy()
                    
                    # Convert to numeric, coercing errors to NaN
                    for col in available_features:
                        df_for_training[col] = pd.to_numeric(df_for_training[col], errors='coerce')
                    df_for_training['N_CCN'] = pd.to_numeric(df_for_training['N_CCN'], errors='coerce')
                    
                    # Remove rows with missing or infinite values
                    df_clean = df_for_training.replace([np.inf, -np.inf], np.nan).dropna()
                    
                    st.write(f"📊 Clean dataset: **{len(df_clean):,}** rows (removed {len(df_for_training) - len(df_clean):,} rows with missing/invalid values)")
                    
                    if len(df_clean) < 100:
                        st.error(f"❌ Insufficient data after cleaning! Only {len(df_clean)} rows remaining. Need at least 100 rows.")
                        st.warning("💡 Try: (1) Use fewer features, (2) Use more data rows, or (3) Check data quality")
                        st.stop()
                    
                    X = df_clean[available_features]
                    y = df_clean['N_CCN']
                    
                    # Additional validation: check for constant columns
                    constant_cols = [col for col in available_features if X[col].nunique() <= 1]
                    if constant_cols:
                        st.warning(f"⚠️ Removing {len(constant_cols)} constant columns: {constant_cols}")
                        X = X.drop(columns=constant_cols)
                        available_features = [f for f in available_features if f not in constant_cols]
                    
                    if len(available_features) == 0:
                        st.error("❌ No valid features remaining after removing constant columns!")
                        st.stop()
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y,
                        test_size=test_size/100,
                        random_state=random_state
                    )
                    
                    st.write(f"📊 Training set: **{len(X_train):,}** rows")
                    st.write(f"📊 Test set: **{len(X_test):,}** rows")
                    
                    # Scale features with error handling
                    try:
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                    except Exception as e:
                        st.error(f"❌ Error during feature scaling: {str(e)}")
                        st.error("💡 This usually happens when data contains non-numeric values or infinite values")
                        st.stop()
                    
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
