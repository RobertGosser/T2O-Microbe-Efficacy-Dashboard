# =============================================================================
# Microbe Treatment Efficacy Dashboard
# T2O Energy - A Streamlit Application
# Version: 8.0
# Author: Robert W. Gosser & Enhanced by Claude
#
# DESCRIPTION:
# ------------
# This dashboard enables our engineering team to evaluate the effectiveness of
# microbe treatments by analyzing oil, water, and gas production across wells.
# It is built using Streamlit for UI and Plotly for interactive visuals.
#
# FILES USED:
# -----------
# - T2O Daily Production (as of 05.12.2025) -- .csv File (Daily Well Production)
#   > Note: This porduction file is updated manually and should always be 
#     Updated with the most recent version.
# - Bug Treatments (as of 05.20.2025) -- .csv File (Treatment Dates and Types)
#   > Note: This treatment file is updated manually and should always be
#     Updated with the most recent version.
# - Workover Reports (as of 05.23.2025) -- .csv File (Record of Well Workovers)
#   > Note: This workover file is updated manually and should always be
#     Updated with the most recent version.
# - TMDH Well Locations -- .csv (Coordinates of Wells for Mapping)
# =============================================================================

# =============================================================================
# Section 1: Required Libraries

import pandas as pd                      # Panda Data manipulation and cleaning
import streamlit as st                   # Streamlit Web-App framework for Dashboard
import plotly.express as px              # Plotly High-level plotting for simple visualizations
import plotly.graph_objects as go        
from datetime import datetime, timedelta # Date/time handling for treatment windows
import numpy as np                       # Numerical computing for future use
# =============================================================================


# =============================================================================
# Section 2: Configuration Constants  

PRE_TREATMENT_TARGET_DAYS = 90   # Target Number of Days Before Treatment for Analysis Window
POST_TREATMENT_TARGET_DAYS = 120 # Target Number of Days After Treatment for Analysis Window
MIN_DATA_DAYS = 30               # Minimum Days of Valid Data Required to Include in Analysis
# =============================================================================


# =============================================================================
# Section 3: Streamlit App Configuration

st.set_page_config(
    page_title="T2O Microbe Dashboard",  # Page Title 
    layout="wide",                       # Use Full Browser Width for Display
    page_icon="T20_White.png"            # Custom Favicon
)
# =============================================================================


# =============================================================================
# Section 4: Data Loading Functions

## -------------------------------------------------------------------------------
## Function: load_production_data()
## Purpose : Loads and standardizes daily oil, gas, and water production data.
## Returns : Cleaned DataFrame with datetime conversion and standardized well names.
## -------------------------------------------------------------------------------
@st.cache_data
def load_production_data():
    """Load and clean production data"""
    try:
        prod = pd.read_csv("T2O Daily Production as of 05.12.2025.csv")
        prod['Date'] = pd.to_datetime(prod['Date'], errors='coerce')
        prod = prod.dropna(subset=['Date'])
        prod['WellName'] = prod['WellName'].str.upper().str.strip()
        prod.rename(columns={'WellName': 'Well'}, inplace=True)
        return prod
    except Exception as e:
        st.error(f"Error loading production data: {str(e)}")
        return pd.DataFrame()

## -------------------------------------------------------------------------------
## Function: load_treatment_data()
## Purpose : Loads bug/chemical treatment records with cleaning of dates and names.
## Returns : Cleaned DataFrame of treatments
## -------------------------------------------------------------------------------
@st.cache_data
def load_treatment_data():
    """Load and clean treatment data"""
    try:
        treat = pd.read_csv("BugTreatments_5-20_Update.csv")
        treat['TreatmentDate'] = pd.to_datetime(treat['TreatmentDate'], errors='coerce')
        treat = treat.dropna(subset=['TreatmentDate'])
        treat['WellName'] = treat['WellName'].str.upper().str.strip()
        treat.rename(columns={'WellName': 'Well'}, inplace=True)
        return treat
    except Exception as e:
        st.error(f"Error loading treatment data: {str(e)}")
        return pd.DataFrame()

## -------------------------------------------------------------------------------
## Function: load_workover_data()
## Purpose : Loads workover events and aligns them with production timeline.
## Returns : Cleaned DataFrame with renamed date column
## -------------------------------------------------------------------------------
@st.cache_data
def load_workover_data():
    """Load and clean workover data"""
    try:
        work = pd.read_csv("workovers.csv")
        work['StartDate'] = pd.to_datetime(work['StartDate'], errors='coerce')
        work = work.dropna(subset=['StartDate'])
        work.rename(columns={"StartDate": "Date"}, inplace=True)
        work['WellName'] = work['WellName'].str.upper().str.strip()
        work.rename(columns={'WellName': 'Well'}, inplace=True)
        return work
    except Exception as e:
        st.error(f"Error loading workover data: {str(e)}")
        return pd.DataFrame()

## -------------------------------------------------------------------------------
## Function: load_coordinate_data()
## Purpose : Loads well latitude/longitude data and standardizes column names.
## Returns : DataFrame with columns: Well, Latitude, Longitude (if valid)
## -------------------------------------------------------------------------------
@st.cache_data
def load_coordinate_data():
    """Load and clean well coordinate data"""
    try:
        df = pd.read_csv("TMDH_Wells_Lat_Long.csv")

        # Attempt to find and rename the well name column from multiple candidates
        well_name_candidates = ['WELLNAME', 'WellName', 'Well_Name', 'well_name', 'Well']
        for col in well_name_candidates:
            if col in df.columns:
                df.rename(columns={col: 'Well'}, inplace=True)
                break

        if 'Well' not in df.columns:
            st.warning(f"Well coordinate file missing well name column. Available columns: {list(df.columns)}")
            return pd.DataFrame()
        
        df['Well'] = df['Well'].str.upper().str.strip()
        
        # Handle different possible column names for coordinates
        lat_candidates = ['Latitude', 'LATITUDE', 'latitude', 'Lat', 'LAT', 'lat']
        lon_candidates = ['Longitude', 'LONGITUDE', 'longitude', 'Lon', 'LON', 'lon', 'Long', 'LONG', 'long']
        
        # Standardize latitude column
        lat_col_found = False
        for col in lat_candidates:
            if col in df.columns:
                df.rename(columns={col: 'Latitude'}, inplace=True)
                lat_col_found = True
                break
        
        # Standardize longitude column
        lon_col_found = False
        for col in lon_candidates:
            if col in df.columns:
                df.rename(columns={col: 'Longitude'}, inplace=True)
                lon_col_found = True
                break
        
        if not lat_col_found or not lon_col_found:
            st.warning(f"Well coordinate file missing latitude or longitude columns. Available columns: {list(df.columns)}")
            return pd.DataFrame()
            
        return df
    except FileNotFoundError:
        st.info("Well coordinate file 'TMDH_Wells_Lat_Long.csv' not found. Map functionality will be disabled.")
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not load well coordinates: {e}")
        return pd.DataFrame()
# =============================================================================


# =============================================================================
# CORE ANALYSIS & IMPACT FUNCTIONS:

## ------------------------------------------------------------------------------- 
## Process well analysis to determine treatment impact:
## ------------------------------------------------------------------------------- 
def process_well_analysis(well_name, production_data, treatments_data, workovers_data, 
                         pre_days=PRE_TREATMENT_TARGET_DAYS, post_days=POST_TREATMENT_TARGET_DAYS,
                         use_full_post_window=False):
    """
    Analyze treatment impact for a single well
    
    Parameters:
    - use_full_post_window (bool): If True, use all available production data after treatment.
                                  If False, respect the post_days parameter for windowed analysis.
    """
    
    # Get Well-Specific Data:
    well_prod = production_data[production_data['Well'] == well_name].copy()       # Well Production Data
    well_treatments = treatments_data[treatments_data['Well'] == well_name].copy() # Well Treatment Data
    
    # Ensure Data is Available for Analysis
    if well_prod.empty or well_treatments.empty:
        return None
    
    # Sort Data by Date
    well_prod = well_prod.sort_values('Date')
    well_treatments = well_treatments.sort_values('TreatmentDate')
    
    # Get First Treatment Date
    first_treatment_date = well_treatments['TreatmentDate'].iloc[0]
    if not pd.api.types.is_datetime64_any_dtype(type(first_treatment_date)):
        first_treatment_date = pd.to_datetime(first_treatment_date)
    
    # Get Production Date Range
    prod_start = well_prod['Date'].min()
    prod_end = well_prod['Date'].max()
    
    # MODIFIED: Replace 90-day pre-treatment cap with first non-zero production date logic
    # Find the first date with any non-zero production (oil, water, or gas)
    first_non_zero_date = well_prod[
        (well_prod['OilProd'] > 0) | 
        (well_prod['WaterProd'] > 0) | 
        (well_prod['GasProd'] > 0)
    ]['Date'].min()
    
    # Use the later of production start or first non-zero date as analysis start
    actual_pre_start = max(first_non_zero_date, prod_start)
    
    # CONDITIONAL POST-TREATMENT WINDOW LOGIC
    if use_full_post_window:
        # For Map, Program Summary, Treatment Impact tabs - use all available data
        actual_post_end = prod_end
    else:
        # For Production Analysis tab - respect the user-selected post_days parameter
        post_end = first_treatment_date + timedelta(days=post_days)
        actual_post_end = min(prod_end, post_end)
    
    # Extract Periods
    pre_treatment_all = well_prod[(well_prod['Date'] >= actual_pre_start) &
                                  (well_prod['Date'] < first_treatment_date)]
    post_treatment_all = well_prod[(well_prod['Date'] > first_treatment_date) &
                                   (well_prod['Date'] <= actual_post_end)]
    
    # Exclude zero production days
    pre_treatment = pre_treatment_all[~((pre_treatment_all['OilProd'] == 0) &
                                        (pre_treatment_all['WaterProd'] == 0) &
                                        (pre_treatment_all['GasProd'] == 0))]
    
    post_treatment = post_treatment_all[~((post_treatment_all['OilProd'] == 0) &
                                          (post_treatment_all['WaterProd'] == 0) &
                                          (post_treatment_all['GasProd'] == 0))]
    
    pre_days_actual = pre_treatment['Date'].nunique()
    post_days_actual = post_treatment['Date'].nunique()
    
    # MODIFIED: Calculate workovers in range for charting functions using correct column names
    workovers_in_range = None
    if workovers_data is not None and not workovers_data.empty:
        well_workovers = workovers_data[workovers_data['Well'] == well_name]
        workovers_in_range = well_workovers[
            (well_workovers['Date'] >= prod_start) &
            (well_workovers['Date'] <= prod_end)
        ].sort_values('Date')
    
    # MODIFIED: Add warning flag for insufficient pre-treatment data
    pre_treatment_warning = pre_days_actual <= 30
    
    if pre_days_actual < 30 or post_days_actual < 30:
        return None

    pre_oil_avg = pre_treatment['OilProd'].mean() if not pre_treatment.empty else 0
    post_oil_avg = post_treatment['OilProd'].mean() if not post_treatment.empty else 0
    oil_change_pct = ((post_oil_avg - pre_oil_avg) / pre_oil_avg) * 100 if pre_oil_avg != 0 else 0

    pre_water_avg = pre_treatment['WaterProd'].mean() if not pre_treatment.empty else 0
    post_water_avg = post_treatment['WaterProd'].mean() if not post_treatment.empty else 0
    water_change_pct = ((post_water_avg - pre_water_avg) / pre_water_avg) * 100 if pre_water_avg != 0 else 0

    pre_gas_avg = pre_treatment['GasProd'].mean() if not pre_treatment.empty else 0
    post_gas_avg = post_treatment['GasProd'].mean() if not post_treatment.empty else 0
    gas_change_pct = ((post_gas_avg - pre_gas_avg) / pre_gas_avg) * 100 if pre_gas_avg != 0 else 0
    
    # Include all treatment events for this well for plotting & metadata
    return {
        'well_name': well_name,
        'treatment_date': first_treatment_date,
        'pre_days_actual': pre_days_actual,
        'post_days_actual': post_days_actual,
        'pre_oil_avg': pre_oil_avg,
        'post_oil_avg': post_oil_avg,
        'oil_change_pct': oil_change_pct,
        'pre_water_avg': pre_water_avg,
        'post_water_avg': post_water_avg,
        'water_change_pct': water_change_pct,
        'pre_gas_avg': pre_gas_avg,
        'post_gas_avg': post_gas_avg,
        'gas_change_pct': gas_change_pct,
        'pre_treatment_warning': pre_treatment_warning,  # MODIFIED: Added warning flag to return dictionary
        'all_treatments': well_treatments,  # MODIFIED: Added all treatments for plotting & metadata
        'production_data': well_prod,  # MODIFIED: Added production data for graphing and analysis
        'workovers': workovers_in_range,  # MODIFIED: Added workovers data for charting functions
        'post_treatment_period': post_treatment_all,  # MODIFIED: Added post-treatment period for chart highlighting
        'analysis_periods': {
            'pre_start': actual_pre_start,
            'pre_end': first_treatment_date,
            'post_start': first_treatment_date,
            'post_end': actual_post_end
        }
    }

## ------------------------------------------------------------------------------- 
## Process Single Treatment Analysis to Determine Impact:
## ------------------------------------------------------------------------------- 
def process_single_treatment_analysis(well, production_data, treatment_date, pre_days, post_days, use_full_post_window=False):
    """
    Analyzes the impact of a single treatment on well production.
    
    Parameters:
    - well (str): Well identifier
    - production_data (DataFrame): Production history
    - treatment_date (datetime): Date of the specific treatment
    - pre_days (int): Days to analyze before treatment
    - post_days (int): Days to analyze after treatment (ignored if use_full_post_window=True)
    - use_full_post_window (bool): If True, analyze all available post-treatment data
    
    Returns:
    - dict: Analysis results with pre/post averages and metrics
    """
    well_data = production_data[production_data['Well'] == well].copy()
    
    if well_data.empty:
        return None
    
    # Ensure treatment_date is datetime
    if isinstance(treatment_date, str):
        treatment_date = pd.to_datetime(treatment_date)
    
    # Define analysis periods
    pre_start = treatment_date - pd.Timedelta(days=pre_days)
    
    if use_full_post_window:
        # Use all available data after treatment
        post_end = well_data['Date'].max()
        post_days_actual = (post_end - treatment_date).days
    else:
        # Use specified post_days
        post_end = treatment_date + pd.Timedelta(days=post_days)
        post_days_actual = post_days
    
    # Filter data for analysis periods
    pre_data = well_data[
        (well_data['Date'] >= pre_start) & 
        (well_data['Date'] < treatment_date) &
        (well_data['OilProd'] > 0)  # Exclude zero production days - FIXED column name
    ]
    
    post_data = well_data[
        (well_data['Date'] > treatment_date) & 
        (well_data['Date'] <= post_end) &
        (well_data['OilProd'] > 0)  # Exclude zero production days - FIXED column name
    ]
    
    # Check minimum data requirements
    if len(pre_data) < MIN_DATA_DAYS or len(post_data) < MIN_DATA_DAYS:
        return None
    
    # Calculate averages - FIXED all column names
    pre_oil_avg = pre_data['OilProd'].mean()
    post_oil_avg = post_data['OilProd'].mean()
    pre_water_avg = pre_data['WaterProd'].mean() if 'WaterProd' in pre_data.columns else 0
    post_water_avg = post_data['WaterProd'].mean() if 'WaterProd' in post_data.columns else 0
    pre_gas_avg = pre_data['GasProd'].mean() if 'GasProd' in pre_data.columns else 0
    post_gas_avg = post_data['GasProd'].mean() if 'GasProd' in post_data.columns else 0
    
    # Calculate percentage changes
    oil_change_pct = ((post_oil_avg - pre_oil_avg) / pre_oil_avg * 100) if pre_oil_avg > 0 else 0
    water_change_pct = ((post_water_avg - pre_water_avg) / pre_water_avg * 100) if pre_water_avg > 0 else 0
    gas_change_pct = ((post_gas_avg - pre_gas_avg) / pre_gas_avg * 100) if pre_gas_avg > 0 else 0
    
    return {
        'treatment_date': treatment_date,
        'pre_oil_avg': pre_oil_avg,
        'post_oil_avg': post_oil_avg,
        'oil_change_pct': oil_change_pct,
        'pre_water_avg': pre_water_avg,
        'post_water_avg': post_water_avg,
        'water_change_pct': water_change_pct,
        'pre_gas_avg': pre_gas_avg,
        'post_gas_avg': post_gas_avg,
        'gas_change_pct': gas_change_pct,
        'pre_days_actual': len(pre_data),
        'post_days_actual': len(post_data),
        'pre_data': pre_data,
        'post_data': post_data
    }

# -------------------------------------------------------------------------------
# Compute Cumulative Impact of All Treatments on a Well:
# -------------------------------------------------------------------------------
def compute_cumulative_impact(well, production_data, treatments_data, pre_days, post_days, use_full_post_window=False):
    """
    Computes cumulative impact of multiple treatments for a single well.
    Shows how each treatment adds to the overall production improvement.
    
    Parameters:
    - well (str): Well identifier
    - production_data (DataFrame): Production history
    - treatments_data (DataFrame): Treatment records
    - pre_days (int): Days to analyze before treatment
    - post_days (int): Days to analyze after treatment (ignored if use_full_post_window=True)
    - use_full_post_window (bool): If True, analyze all available post-treatment data
    
    Returns:
    - DataFrame: Cumulative impact analysis with individual treatment effects
    """
    well_treatments = treatments_data[treatments_data['Well'] == well].copy()
    well_treatments = well_treatments.drop_duplicates(subset=['TreatmentDate'])
    
    if well_treatments.empty:
        return pd.DataFrame()
    
    # Sort treatments by date
    well_treatments = well_treatments.sort_values('TreatmentDate')
    
    results = []
    cumulative_gain = 0
    
    for idx, (_, treatment) in enumerate(well_treatments.iterrows()):
        # Analyze each individual treatment
        # ✅ FIXED: Pass use_full_post_window parameter to process_well_analysis
        analysis = process_single_treatment_analysis(
            well, 
            production_data, 
            treatment['TreatmentDate'], 
            pre_days, 
            post_days,
            use_full_post_window=use_full_post_window
        )
        
        if analysis:
            oil_gain = analysis['post_oil_avg'] - analysis['pre_oil_avg']
            cumulative_gain += oil_gain
            
            results.append({
                'TreatmentDate': treatment['TreatmentDate'],
                'TreatmentType': treatment.get('TreatmentType', 'Unknown'),  # ✅ FIXED: Added missing TreatmentType field
                'PreOilAvg': analysis['pre_oil_avg'],
                'PostOilAvg': analysis['post_oil_avg'],
                'OilGain': oil_gain,
                'CumulativeGain': cumulative_gain,
                'PreDaysActual': analysis.get('pre_days_actual', 0),
                'PostDaysActual': analysis.get('post_days_actual', 0)
            })
    
    return pd.DataFrame(results)
# =============================================================================


# =============================================================================
# SUMMARY & GROUPING FUNCTIONS

## ------------------------------------------------------------------------------- 
# OUTLIER WELL FILTERING SYSTEM
OUTLIER_WELLS = ['PAUL MOSS 4 (G)']
## ------------------------------------------------------------------------------- 

## ------------------------------------------------------------------------------- 
## Compute Impact Summaries for All Treated Wells:
## ------------------------------------------------------------------------------- 
def compute_all_well_summaries(production_data, treatments_data, workovers_data, post_days=None):
    """
    Computes production impact summaries for all treated wells in the dataset.
    For each well, calculates average production before and after its primary treatment,
    and aggregates the key metrics for program-wide review.
    
    ⛔ EXCLUDES KNOWN OUTLIER WELLS from program summary statistics.

    Parameters:
    - production_data (DataFrame): Historical oil, gas, and water production data
    - treatments_data (DataFrame): Microbe treatment events by well
    - workovers_data (DataFrame): Workover records to be passed into analysis
    - post_days (int, optional): Post-treatment analysis window in days. If None, uses all available data.

    Returns:
    - DataFrame: One row per well summarizing pre/post averages and percent changes
    """
    # Initialize Results List
    results = []
    
    # Get Unique Wells That Received Treatments
    treated_wells = treatments_data['Well'].unique()
    
    # ⛔ FILTER OUT KNOWN OUTLIERS from program summary
    filtered_wells = [well for well in treated_wells if well not in OUTLIER_WELLS]

    # Process each treated well individually (excluding outliers)
    for well in filtered_wells:
        # MODIFIED: Use post_days parameter to control analysis window
        if post_days is not None:
            # Use specified post-treatment window
            analysis = process_well_analysis(well, production_data, treatments_data, workovers_data, 
                                           PRE_TREATMENT_TARGET_DAYS, post_days, use_full_post_window=False)
        else:
            # Use all available data (original behavior)
            analysis = process_well_analysis(well, production_data, treatments_data, workovers_data, 
                                           use_full_post_window=True)

        # Only Include Wells With Valid Analysis
        if analysis:
            results.append({
                'Well': well,
                'Treatment Date': analysis['treatment_date'].strftime('%Y-%m-%d'),
                'Pre Days': analysis['pre_days_actual'],
                'Post Days': analysis['post_days_actual'],
                'Pre Oil Avg': analysis['pre_oil_avg'],
                'Post Oil Avg': analysis['post_oil_avg'],
                'Oil Change %': analysis['oil_change_pct'],
                'Pre Water Avg': analysis['pre_water_avg'],
                'Post Water Avg': analysis['post_water_avg'],
                'Water Change %': analysis['water_change_pct'],
                'Pre Gas Avg': analysis['pre_gas_avg'],
                'Post Gas Avg': analysis['post_gas_avg'],
                'Gas Change %': analysis['gas_change_pct']
            })
    return pd.DataFrame(results)

## ------------------------------------------------------------------------------- 
## Compute Oil Change Percentages for All Mappable Wells:
## ------------------------------------------------------------------------------- 
def compute_oil_change_for_map(production_data, treatments_data, workovers_data, well_coords):
    """
    Filters for wells with coordinate data and computes oil change metrics
    for visualization on a map. Uses the primary treatment and pre/post averages.
    
    ⛔ EXCLUDES KNOWN OUTLIER WELLS from map-based oil change summaries.

    Parameters:
    - production_data (DataFrame): Oil, gas, and water volumes
    - treatments_data (DataFrame): Treatment history
    - workovers_data (DataFrame): Workover intervention history
    - well_coords (DataFrame): Latitude and longitude per well

    Returns:
    - DataFrame: Coordinates and production change metrics for mapping
    """
    # Ensure all required data is available
    results = []
    
    # Get unique wells that received treatments
    treated_wells = treatments_data['Well'].unique()
    wells_with_coords = well_coords['Well'].unique()

    # Identify wells that are both treated and have geographic data
    wells_to_analyze = [well for well in treated_wells if well in wells_with_coords]
    
    # ⛔ FILTER OUT KNOWN OUTLIERS from map visualization
    filtered_wells = [well for well in wells_to_analyze if well not in OUTLIER_WELLS]

    # Process each well that has both treatment and coordinate data (excluding outliers)
    for well in filtered_wells:
        # ✅ FIXED: Added use_full_post_window=True for Map tab
        analysis = process_well_analysis(well, production_data, treatments_data, workovers_data,
                                       use_full_post_window=True)

        # Only include wells with valid analysis
        if analysis:
            # Get this well's coordinates
            well_coord = well_coords[well_coords['Well'] == well].iloc[0]

            # Append the results with all necessary information
            results.append({
                'Well': well,
                'Latitude': well_coord['Latitude'],
                'Longitude': well_coord['Longitude'],
                'Oil_Change_Pct': analysis['oil_change_pct'],
                'Pre_Oil_Avg': analysis['pre_oil_avg'],
                'Post_Oil_Avg': analysis['post_oil_avg'],
                'Treatment_Date': analysis['treatment_date'],
                'TreatmentType': analysis['all_treatments']['TreatmentType'].iloc[0] \
                    if 'TreatmentType' in analysis['all_treatments'].columns else 'Unknown'
            })

    return pd.DataFrame(results)

# =============================================================================
# OPTIONAL: SIDEBAR TOGGLE FOR INTERACTIVE OUTLIER FILTERING
# =============================================================================
def get_filtered_well_list(well_list, exclude_outliers=True):
    """
    Helper function to filter wells based on outlier exclusion setting.
    
    Parameters:
    - well_list (list): List of all wells
    - exclude_outliers (bool): Whether to exclude known outliers
    
    Returns:
    - list: Filtered well list
    """
    if exclude_outliers:
        return [well for well in well_list if well not in OUTLIER_WELLS]
    else:
        return well_list
# =============================================================================



# =============================================================================
# UTILITY FUNCTIONS

## ------------------------------------------------------------------------------- 
## Safe Date Formatting Function to Handle Various Inputs:
## ------------------------------------------------------------------------------- 
def safe_format_date(date_series):
    """
    Converts a pandas Series of datetime or mixed-type values into a string format 'YYYY-MM-DD'.
    Handles both datetime64 and object types.

    Parameters:
    - date_series (Series): Input pandas Series that may contain datetime or mixed types

    Returns:
    - Series: Formatted string version of dates in 'YYYY-MM-DD' format
    """
    try:
        # If already datetime, apply strftime directly
        if pd.api.types.is_datetime64_any_dtype(date_series):
            return date_series.dt.strftime('%Y-%m-%d')
        else:
            # Otherwise, try converting and then formatting
            return pd.to_datetime(date_series, errors='coerce').dt.strftime('%Y-%m-%d')
    except:
        # Fallback: convert all to string if formatting fails
        return date_series.astype(str)
# -------------------------------------------------------------------------------
# Calculate Average Monthly Cost for Treatments on a Well (Revised):
# -------------------------------------------------------------------------------
def calculate_avg_monthly_cost_for_well(well_name, treatments_data, production_data, treatment_cost=450):
    """
    Calculate the average monthly cost for a well based on treatments since primary treatment
    and months of production data available after the primary treatment.
    
    Parameters:
    - well_name (str): Name of the well
    - treatments_data (DataFrame): Treatment records
    - production_data (DataFrame): Production history
    - treatment_cost (float): Cost per treatment (default $450)
    
    Returns:
    - float: Average monthly cost, or 0 if no data available
    """
    # Get treatments for this well
    well_treatments = treatments_data[treatments_data['Well'] == well_name]
    if well_treatments.empty:
        return 0
    
    # Get primary (first) treatment date
    primary_treatment_date = well_treatments['TreatmentDate'].min()
    
    # Get production data for this well after primary treatment
    well_prod = production_data[production_data['Well'] == well_name]
    post_treatment_prod = well_prod[
        (well_prod['Date'] > primary_treatment_date) &
        ((well_prod['OilProd'] > 0) | (well_prod['WaterProd'] > 0) | (well_prod['GasProd'] > 0))
    ]
    
    if post_treatment_prod.empty:
        return 0
    
    # Calculate months from primary treatment to last production date
    last_production_date = post_treatment_prod['Date'].max()
    months_span = ((last_production_date - primary_treatment_date).days / 30.44)
    
    if months_span <= 0:
        return 0
    
    # Calculate total cost for all treatments on this well
    total_treatments = len(well_treatments)
    total_cost = total_treatments * treatment_cost
    
    # Return average monthly cost
    return total_cost / months_span
# =============================================================================


# =============================================================================
# VISUALIZATION FUNCTIONS

## ------------------------------------------------------------------------------- 
## Create the Main Production History Chart:
## ------------------------------------------------------------------------------- 
def create_production_chart(analysis, selected_well):
    """
    Create an interactive time-series chart displaying oil, water, and gas production
    for a specific well. Annotates treatments, workovers, and highlights analysis windows.

    Parameters:
    - Analysis (dict): Result of process_well_analysis with production data and annotations
    - selected_well (str): Name of the well being visualized

    Returns:
    - fig (plotly.graph_objs.Figure): Interactive Production chart
    """
    df = analysis['production_data']  # Full production history
    treatment_date = analysis['treatment_date']  # Primary treatment to highlight
    periods = analysis['analysis_periods']  # Dictionary with pre/post window dates

    fig = go.Figure()

    # Add Production Traces for Oil, Water, & Gas
    fig.add_trace(go.Scatter(
        x=df['Date'], 
        y=df['OilProd'], 
        name='Oil Production',
        line=dict(color='green'),
        mode='lines'
    ))

    fig.add_trace(go.Scatter(
        x=df['Date'], 
        y=df['WaterProd'], 
        name='Water Production',
        line=dict(color='blue'),
        mode='lines'
    ))

    fig.add_trace(go.Scatter(
        x=df['Date'], 
        y=df['GasProd'], 
        name='Gas Production',
        line=dict(color='red'),
        mode='lines'
    ))

    # Add vertical dashed lines and annotations for each treatment
    all_treatments = analysis['all_treatments']
    for _, row in all_treatments.iterrows():
        treatment_dt = row['TreatmentDate']

        fig.add_shape(
            type="line",
            x0=treatment_dt, x1=treatment_dt,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="orange", width=2, dash="dash")
        )

        fig.add_annotation(
            x=treatment_dt,
            y=1.05,
            yref="paper",
            text=f"Treatment: {row.get('TreatmentType', 'N/A')}",
            showarrow=False,
            font=dict(size=10)
        )

    # Highlight primary treatment with a bold red line
    fig.add_shape(
        type="line",
        x0=treatment_date, x1=treatment_date,
        y0=0, y1=1,
        yref="paper",
        line=dict(color="red", width=3)
    )

    fig.add_annotation(
        x=treatment_date,
        y=1.1,
        yref="paper",
        text="Primary Treatment",
        showarrow=False,
        font=dict(size=12, color="red")
    )

    # Add dotted purple lines and labels for workovers
    if analysis['workovers'] is not None and not analysis['workovers'].empty:
        for _, row in analysis['workovers'].iterrows():
            workover_dt = row['Date']

            fig.add_shape(
                type="line",
                x0=workover_dt, x1=workover_dt,
                y0=0, y1=1,
                yref="paper",
                line=dict(color="purple", width=2, dash="dot")
            )

            fig.add_annotation(
                x=workover_dt,
                y=-0.05,
                yref="paper",
                text="Workover",
                showarrow=False,
                font=dict(size=10)
            )

    # MODIFIED: Handle pre-treatment period shading with gaps (similar to post-treatment)
    # Filter production data for pre-treatment period
    pre_data = df[(df['Date'] >= periods['pre_start']) & (df['Date'] < treatment_date)].copy()
    
    if not pre_data.empty:
        pre_shading = pre_data.copy()
        
        # Mark valid production days
        pre_shading['is_valid_production'] = ~(
            (pre_shading['OilProd'] == 0) & 
            (pre_shading['WaterProd'] == 0) & 
            (pre_shading['GasProd'] == 0)
        )
        
        # Create production groups
        pre_shading['production_group'] = (
            pre_shading['is_valid_production']
            .ne(pre_shading['is_valid_production'].shift())
            .cumsum()
        )
        
        # Get valid production ranges for yellow shading
        valid_pre_production = pre_shading[pre_shading['is_valid_production']]
        
        if not valid_pre_production.empty:
            pre_production_ranges = (
                valid_pre_production
                .groupby('production_group')['Date']
                .agg(['min', 'max', 'count'])
                .reset_index()
            )
            
            # Add yellow shaded rectangles for valid pre-treatment periods
            for idx, row in pre_production_ranges.iterrows():
                start_date = row['min']
                end_date = row['max']
                segment_days = row['count']
                
                # Adjust opacity based on segment length
                base_opacity = 0.1
                adjusted_opacity = min(base_opacity + (segment_days / 100 * 0.05), 0.2)
                
                fig.add_vrect(
                    x0=start_date,
                    x1=end_date + pd.Timedelta(days=1),
                    fillcolor="yellow",
                    opacity=adjusted_opacity,
                    layer="below",
                    line_width=0,
                )
        
        # Add red shading for zero production gaps in pre-treatment period
        zero_pre_production = pre_shading[~pre_shading['is_valid_production']]
        if not zero_pre_production.empty:
            pre_zero_ranges = (
                zero_pre_production
                .groupby('production_group')['Date']
                .agg(['min', 'max'])
                .reset_index()
            )
            
            for _, row in pre_zero_ranges.iterrows():
                fig.add_vrect(
                    x0=row['min'],
                    x1=row['max'] + pd.Timedelta(days=1),
                    fillcolor="red",
                    opacity=0.15,
                    layer="below",
                    line_width=0
                )

    # Handle post-treatment period shading (existing logic)
    post_data = analysis.get('post_treatment_period', pd.DataFrame())
    
    if not post_data.empty:
        post_shading = post_data.copy()
        
        # Mark valid production days
        post_shading['is_valid_production'] = ~(
            (post_shading['OilProd'] == 0) & 
            (post_shading['WaterProd'] == 0) & 
            (post_shading['GasProd'] == 0)
        )
        
        # Create production groups
        post_shading['production_group'] = (
            post_shading['is_valid_production']
            .ne(post_shading['is_valid_production'].shift())
            .cumsum()
        )
        
        # Calculate production statistics
        total_days = len(post_shading)
        production_days = post_shading['is_valid_production'].sum()
        zero_production_days = total_days - production_days
        
        # Get production ranges
        valid_production = post_shading[post_shading['is_valid_production']]
        
        if not valid_production.empty:
            production_ranges = (
                valid_production
                .groupby('production_group')['Date']
                .agg(['min', 'max', 'count'])
                .reset_index()
            )
            
            # Add shaded rectangles with different opacity based on segment length
            for idx, row in production_ranges.iterrows():
                start_date = row['min']
                end_date = row['max']
                segment_days = row['count']
                
                # Adjust opacity based on segment length (longer segments = more opaque)
                base_opacity = 0.1
                adjusted_opacity = min(base_opacity + (segment_days / 100 * 0.05), 0.2)
                
                fig.add_vrect(
                    x0=start_date,
                    x1=end_date + pd.Timedelta(days=1),
                    fillcolor="blue",
                    opacity=adjusted_opacity,
                    layer="below",
                    line_width=0
                )
            
            # Optional: Add gap indicators (uncomment to show zero production periods)
            zero_production = post_shading[~post_shading['is_valid_production']]
            if not zero_production.empty:
                zero_ranges = (
                    zero_production
                    .groupby('production_group')['Date']
                    .agg(['min', 'max'])
                    .reset_index()
                )
                
                for _, row in zero_ranges.iterrows():
                    fig.add_vrect(
                        x0=row['min'],
                        x1=row['max'] + pd.Timedelta(days=1),
                        fillcolor="red",
                        opacity=0.15,
                        layer="below",
                        line_width=0
                    )

    # Final layout configuration
    fig.update_layout(
        title=f"Production History - {selected_well}",
        xaxis_title="Date",
        yaxis_title="Production",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    return fig

## ------------------------------------------------------------------------------- 
## Create Cumulative Impact Chart for a Well:
## ------------------------------------------------------------------------------- 
def create_cumulative_impact_chart(cumulative_impact, selected_well):
    """
    Generates a dual-layered chart showing individual treatment gains and cumulative
    oil production improvement over time for a specific well.

    Parameters:
    - cumulative_impact (DataFrame): Output from compute_cumulative_impact() containing treatment metrics
    - selected_well (str): The well currently being reviewed

    Returns:
    - fig (plotly.graph_objs.Figure): Plotly figure showing cumulative and individual gains
    """
    fig = go.Figure()

    # Add cumulative gain as a line with markers
    fig.add_trace(go.Scatter(
        x=cumulative_impact['TreatmentDate'],
        y=cumulative_impact['CumulativeGain'],
        mode='lines+markers',
        name='Cumulative Oil Gain',
        line=dict(color='darkgreen', width=3),
        marker=dict(size=8, color='darkgreen'),
        hovertemplate="<b>%{x}</b><br>" +
                      "Cumulative Gain: %{y:.2f} BBL/day<br>" +
                      "<extra></extra>"
    ))

    # Add bar plot for each treatment's individual gain
    fig.add_trace(go.Bar(
        x=cumulative_impact['TreatmentDate'],
        y=cumulative_impact['OilGain'],
        name='Individual Treatment Gain',
        marker_color='lightgreen',
        opacity=0.6,
        hovertemplate="<b>%{x}</b><br>" +
                      "Treatment Gain: %{y:.2f} BBL/day<br>" +
                      "<extra></extra>"
    ))

    # Draw a horizontal line at y=0 for reference
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="Baseline (No Change)"
    )

    # Annotate treatment type above each cumulative point
    for _, row in cumulative_impact.iterrows():
        fig.add_annotation(
            x=row['TreatmentDate'],
            y=row['CumulativeGain'] + (max(cumulative_impact['CumulativeGain']) * 0.05),
            text=row['TreatmentType'],
            showarrow=False,
            font=dict(size=9),
            textangle=-45
        )

    # Final chart layout adjustments
    fig.update_layout(
        title=f"Cumulative Impact of Microbe Treatments - {selected_well}",
        xaxis_title="Treatment Date",
        yaxis_title="Oil Production Change (BBL/day)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=500
    )

    return fig

## ------------------------------------------------------------------------------- 
## Create Map with Selected Well Focus:
## ------------------------------------------------------------------------------- 
def create_map_with_selected_well_focus(map_data, selected_well, well_coords):
    """
    Creates a Mapbox-based map visualizing all wells with oil change coloring and
    highlights the selected well with a distinct ring and zoom focus.

    Parameters:
    - map_data (DataFrame): Contains well coordinates and oil change percentages
    - selected_well (str): Name of the well selected by the user
    - well_coords (DataFrame): Backup coordinate data in case map_data is incomplete

    Returns:
    - fig (plotly.graph_objs.Figure): Configured map visualization with annotations
    """

    # Get coordinates for selected well if available in map_data
    selected_well_data = map_data[map_data['Well'] == selected_well]

    if selected_well_data.empty:
        # Fallback: Use coordinates from the backup well_coords data
        selected_coords = well_coords[well_coords['Well'] == selected_well]
        if not selected_coords.empty:
            center_lat = selected_coords['Latitude'].iloc[0]
            center_lon = selected_coords['Longitude'].iloc[0]
            zoom_level = 14
        else:
            # Default zoom if selected well not found
            center_lat = map_data['Latitude'].mean() if not map_data.empty else 32.0
            center_lon = map_data['Longitude'].mean() if not map_data.empty else -102.0
            zoom_level = 10
    else:
        center_lat = selected_well_data['Latitude'].iloc[0]
        center_lon = selected_well_data['Longitude'].iloc[0]
        zoom_level = 14

    # Create base scatter mapbox chart of all wells
    fig = px.scatter_mapbox(
        map_data,
        lat="Latitude",
        lon="Longitude",
        color="Oil_Change_Pct",
        size=[15] * len(map_data),
        hover_name="Well",
        color_continuous_scale=["red", "orange", "yellow"],
        color_continuous_midpoint=0,
        range_color=[map_data["Oil_Change_Pct"].min(), map_data["Oil_Change_Pct"].max()],
        mapbox_style="open-street-map",
        labels={"Oil_Change_Pct": "Oil Change (%)"}
    )

    # Customize hover and size appearance
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>" +
                      "Oil Change: %{marker.color:.1f}%<br>" +
                      "<extra></extra>",
        marker_size=15
    )

    # Add selected well highlight if found
    if not selected_well_data.empty:
        selected_oil_change = selected_well_data['Oil_Change_Pct'].iloc[0]

        # Outer background ring
        fig.add_trace(go.Scattermapbox(
            lat=[center_lat],
            lon=[center_lon],
            mode='markers',
            marker=dict(
                size=40,
                color='black',
                opacity=0.5
            ),
            name='Selection Background',
            hoverinfo='skip',
            showlegend=False
        ))

        # Inner highlighted marker
        fig.add_trace(go.Scattermapbox(
            lat=[center_lat],
            lon=[center_lon],
            mode='markers',
            marker=dict(
                size=30,
                color='Red',
                opacity=0.5
            ),
            name=f'🎯 Selected: {selected_well}',
            hovertemplate=f"<b>🎯 SELECTED WELL</b><br>" +
                          f"<b>{selected_well}</b><br>" +
                          f"Oil Change: {selected_oil_change:.1f}%<br>" +
                          "<extra></extra>",
            showlegend=True
        ))

    # Final map layout and styling
    fig.update_layout(
        mapbox=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom_level
        ),
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(color="black")
        )
    )

    return fig
# =============================================================================

# =============================================================================
# MAIN APPLICATION
def main():
    # Load all data
    prod_history = load_production_data()
    treatments = load_treatment_data()
    workovers = load_workover_data()
    well_coords = load_coordinate_data()
    
    # Header
    st.title("T2O Microbe Treatment Efficacy Dashboard")
    
    # Try to load logo
    try:
        st.image("T20_White.png", width=200)
    except:
        pass
    
    # Data validation
    if prod_history.empty or treatments.empty:
        st.error("Unable to load required data files. Please check file paths and formats.")
        st.stop()
    
    # =============================================================================
    # SIDEBAR CONFIGURATION
    # =============================================================================

    OUTLIER_WELLS = ['PAUL MOSS 4 (G)']  # List of known biased outlier wells

    with st.sidebar:
        st.header("Analysis Settings")
        
        # Toggle to include or exclude known outliers
        exclude_outliers = st.checkbox("Exclude known outliers", value=True)
        
        # Well selection logic
        treated_wells = treatments['Well'].unique()
        raw_available_wells = [w for w in treated_wells if w in prod_history['Well'].unique()]
        
        if exclude_outliers:
            available_wells = [w for w in raw_available_wells if w not in OUTLIER_WELLS]
        else:
            available_wells = raw_available_wells
        
        if not available_wells:
            st.error("No wells found with both treatment and production data.")
            st.stop()
        
        selected_well = st.selectbox("Select Well", sorted(available_wells))
        
        # Data quality info
        st.subheader("Data Quality")
        st.info(f"Minimum required days: {MIN_DATA_DAYS}")
        st.info("Zero-production days excluded from averages")
    
    # =============================================================================
    # TAB STRUCTURE - UPDATED WITH ECONOMICS TAB
    # =============================================================================
    if well_coords.empty:
        tabs = st.tabs([
            "📈 Production Analysis",
            #"📊 Treatment Impact",
            "🛠️ Workover Records",
            "📋 Program Summary",
            "💰 Economics"
        ])
        map_available = False
    else:
        tabs = st.tabs([
            "🗺️ Well Map",
            "📈 Production Analysis",
            #"📊 Treatment Impact", 
            "🛠️ Workover Records",
            "📋 Program Summary",
            "💰 Economics"
        ])
        map_available = True
    
    # =============================================================================
    # TAB 1: WELL MAP 🗺️(if coordinates available)
    # =============================================================================
    tab_index = 0
    
    if map_available:
        with tabs[tab_index]:
            st.subheader(f"Well Locations - Focused on {selected_well}")
            st.caption("Hover Over Points for Individual Well Oil Change Metrics.")
            
            with st.spinner("Loading map data..."):
                map_data = compute_oil_change_for_map(prod_history, treatments, workovers, well_coords)
                
                if not map_data.empty:
                    map_fig = create_map_with_selected_well_focus(map_data, selected_well, well_coords)
                    st.plotly_chart(map_fig, use_container_width=True)
                    
                    # Map summary stats
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Wells Mapped", len(map_data))
                    with col2:
                        positive_wells = len(map_data[map_data['Oil_Change_Pct'] > 0])
                        st.metric("Wells with Positive Response", positive_wells)
                    with col3:
                        avg_change = map_data['Oil_Change_Pct'].mean()
                        st.metric("Average Oil Change", f"{avg_change:.1f}%")
                    with col4:
                        max_change = map_data['Oil_Change_Pct'].max()
                        st.metric("Best Response", f"{max_change:.1f}%")

                    # Analysis summary caption:
                    # May Need to adjust this caption based on the Post- analysis period
                    st.caption("Analysis Complete Via the Following Period: All non-zero days before treatment · Post-window: All available data")

                else:
                    st.warning("No wells with both treatment data and coordinates found.")
        
        tab_index += 1
    
    # =============================================================================
    # TAB 2: PRODUCTION ANALYSIS📈
    # =============================================================================
    with tabs[tab_index]:
        st.subheader(f"Production Analysis - {selected_well}")
        
        # Calculate dynamic post-treatment window options
        with st.spinner("Calculating available data..."):
            # Get treatment date for selected well
            well_treatments = treatments[treatments['Well'] == selected_well]
            if not well_treatments.empty:
                treatment_date = well_treatments['TreatmentDate'].iloc[0]
                well_prod_data = prod_history[prod_history['Well'] == selected_well]
                
                if not well_prod_data.empty:
                    max_date = well_prod_data['Date'].max()
                    max_post_days = (max_date - treatment_date).days
                    
                    # Create intervals of 30 days up to max available
                    if max_post_days >= 30:
                        intervals = list(range(30, max_post_days, 30))
                        if max_post_days not in intervals:
                            intervals.append(max_post_days)
                        window_options = [f"{d} days" for d in intervals]
                        
                        # Default to 90 days if available, otherwise first option
                        default_index = 2 if len(window_options) > 2 else 0
                    else:
                        window_options = [f"{max_post_days} days"]
                        default_index = 0
                else:
                    window_options = ["30 days"]
                    default_index = 0
            else:
                window_options = ["30 days"]
                default_index = 0
        
        # Post-treatment window selection
        post_window_option = st.selectbox(
            "Post-Treatment Analysis Window",
            window_options,
            index=default_index
        )
        
        post_days = int(post_window_option.split()[0])
        
        with st.spinner("Analyzing production data..."):
            # FIXED: Explicitly set use_full_post_window=False for Production Analysis tab
            analysis = process_well_analysis(
                selected_well, 
                prod_history, 
                treatments, 
                workovers, 
                PRE_TREATMENT_TARGET_DAYS, 
                post_days,
                use_full_post_window=False  # <-- This ensures user-selected post_days is respected
            )
            
            if analysis:
                # Key metrics - UPDATED to include Average Monthly Cost
                col1, col2, col3, col4, col5 = st.columns(5)  # Changed from 4 to 5 columns
                with col1:
                    st.metric(
                        "Oil Change", 
                        f"{analysis['oil_change_pct']:.1f}%",
                        delta=f"{analysis['post_oil_avg'] - analysis['pre_oil_avg']:.2f} BBL/day"
                    )
                with col2:
                    st.metric(
                        "Water Change",
                        f"{analysis['water_change_pct']:.1f}%",
                        delta=f"{analysis['post_water_avg'] - analysis['pre_water_avg']:.2f} BBL/day"
                    )
                with col3:
                    st.metric(
                        "Gas Change",
                        f"{analysis['gas_change_pct']:.1f}%",
                        delta=f"{analysis['post_gas_avg'] - analysis['pre_gas_avg']:.2f} MCF/day"
                    )
                with col4:
                    treatment_date_str = analysis['treatment_date'].strftime('%Y-%m-%d')
                    st.metric("Primary Treatment Date", treatment_date_str)
                with col5:
                    # NEW: Calculate and display Average Monthly Cost
                    avg_monthly_cost = calculate_avg_monthly_cost_for_well(
                        selected_well, treatments, prod_history
                    )
                    st.metric(
                        "Avg Monthly Cost", 
                        f"${avg_monthly_cost:,.0f}",
                        help="Average monthly cost based on all treatments and production months since primary treatment"
                    )
                
                # Production chart
                prod_fig = create_production_chart(analysis, selected_well)
                st.plotly_chart(prod_fig, use_container_width=True)
                
                # Analysis Period Details Table
                window_text = f"{post_days} days" if post_days != 9999 else "Max Available"
                st.subheader(f"Analysis Period Details - Post-Treatment Window: {window_text}")
                
                # Enhanced analysis table with cost information
                enhanced_analysis_data = {
                    'Metric': [
                        'Oil Production (BBL/day)', 
                        'Water Production (BBL/day)', 
                        'Gas Production (MCF/day)', 
                    ],
                    'Pre-Treatment': [
                        f"{analysis['pre_oil_avg']:.2f}",
                        f"{analysis['pre_water_avg']:.2f}",
                        f"{analysis['pre_gas_avg']:.2f}",
                    ],
                    'Post-Treatment': [
                        f"{analysis['post_oil_avg']:.2f}",
                        f"{analysis['post_water_avg']:.2f}",
                        f"{analysis['post_gas_avg']:.2f}",
                    ],
                    'Change (%)': [
                        f"{analysis['oil_change_pct']:.1f}%",
                        f"{analysis['water_change_pct']:.1f}%",
                        f"{analysis['gas_change_pct']:.1f}%",
                    ]
                }

                # For Production Analysis tab (use dynamic post_days):
                st.caption(f"Analysis Complete Via the Following Period: All non-zero days before treatment · Post-window: {post_days} days")

                # Create analysis summary table
                analysis_df = pd.DataFrame(enhanced_analysis_data)
                st.dataframe(analysis_df, use_container_width=True)

                # Download option for analysis table
                analysis_csv = analysis_df.to_csv(index=False)
                st.download_button(
                    label="Download Analysis Summary as CSV",
                    data=analysis_csv,
                    file_name=f"analysis_summary_{selected_well.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
                
                # Show raw data if requested
                if st.checkbox("Show raw production data"):
                    st.subheader("Raw Production Data")
                    display_data = analysis['production_data'].copy()
                    display_data['Date'] = display_data['Date'].dt.strftime('%Y-%m-%d')
                    st.dataframe(display_data, use_container_width=True)
                    
            else:
                st.error(f"Insufficient data for analysis of well {selected_well}. Need at least {MIN_DATA_DAYS} days of data in both pre and post periods.")

    tab_index += 1
    
    # =============================================================================
    # TAB 3: TREATMENT IMPACT📊 - UPDATED VERSION
    # =============================================================================
    #with tabs[tab_index]:
        #st.subheader(f"Treatment Impact Analysis - {selected_well}")
        
        #with st.spinner("Computing cumulative impact..."):
            # ✅ FIXED: Ensure compute_cumulative_impact uses full post-window
            # This function should be updated to pass use_full_post_window=True to any 
            # internal process_well_analysis calls
            #cumulative_impact = compute_cumulative_impact(selected_well, prod_history, treatments, 
                                                        #PRE_TREATMENT_TARGET_DAYS, POST_TREATMENT_TARGET_DAYS,
                                                        #use_full_post_window=True)
            
            #if not cumulative_impact.empty:
                # Summary metrics
                #total_treatments = len(cumulative_impact)
                #total_gain = cumulative_impact['CumulativeGain'].iloc[-1]
                #avg_gain_per_treatment = cumulative_impact['OilGain'].mean()
                
                #col1, col2, col3 = st.columns(3)
                #with col1:
                    #st.metric("Total Treatments", total_treatments)
                #with col2:
                    #st.metric("Total Oil Gain", f"{total_gain:.2f} BBL/day")
                #with col3:
                    #st.metric("Avg Gain per Treatment", f"{avg_gain_per_treatment:.2f} BBL/day")
                
                # Cumulative impact chart
                #cum_fig = create_cumulative_impact_chart(cumulative_impact, selected_well)
                #st.plotly_chart(cum_fig, use_container_width=True)

                # ✅ UPDATED CAPTION: Now references full post-window analysis
                #st.caption("Pre-window: All non-zero days before each treatment · Post-window: All available data")

                # Treatment details table with NEW COLUMNS
                #st.subheader("Individual Treatment Summary")
                #display_cumulative = cumulative_impact.copy()
                #display_cumulative['TreatmentDate'] = safe_format_date(display_cumulative['TreatmentDate'])
                
                # ADD Pre Days and Post Days columns if they exist in the cumulative_impact DataFrame
                # (These should be available if compute_cumulative_impact calls the updated process_single_treatment_analysis)
                #if 'PreDaysActual' in display_cumulative.columns and 'PostDaysActual' in display_cumulative.columns:
                    # Reorder columns to include the new ones
                    #column_order = ['TreatmentDate', 'PreOilAvg', 'PostOilAvg', 'OilGain', 'CumulativeGain', 
                                'PreDaysActual', 'PostDaysActual']
                    # Only include columns that actually exist
                    #available_columns = [col for col in column_order if col in display_cumulative.columns]
                    #display_cumulative = display_cumulative[available_columns]
                    
                    # Rename columns for better display
                    #display_cumulative = display_cumulative.rename(columns={
                        #'PreDaysActual': 'Pre Days',
                        #'PostDaysActual': 'Post Days'
                    #})
                
                #display_cumulative = display_cumulative.round(2)
                #st.dataframe(display_cumulative, use_container_width=True)

            #else:
                #st.warning(f"No treatment impact data available for well {selected_well}")

    #tab_index += 1
   
    # =============================================================================
    # TAB 4: WORKOVER RECORDS🛠️
    # =============================================================================
    with tabs[tab_index]:
        st.subheader(f"Workover Records - {selected_well}")
        
        if not workovers.empty:
            well_workovers = workovers[workovers['Well'] == selected_well].copy()
            
            if not well_workovers.empty:
                # Format dates for display
                display_workovers = well_workovers.copy()
                display_workovers['Date'] = safe_format_date(display_workovers['Date'])
                
                st.dataframe(display_workovers, use_container_width=True)
                
                # Workover timeline relative to treatments
                if selected_well in treatments['Well'].values:
                    st.subheader("Workover Timeline vs Treatment Dates")
                    
                    well_treatments = treatments[treatments['Well'] == selected_well]
                    
                    timeline_data = []
                    
                    # Add treatments
                    for _, treatment in well_treatments.iterrows():
                        timeline_data.append({
                            'Date': treatment['TreatmentDate'],
                            'Event': 'Treatment',
                            'Type': treatment.get('TreatmentType', 'Unknown'),
                            'Color': 'orange'
                        })
                    
                    # Add workovers
                    for _, workover in well_workovers.iterrows():
                        timeline_data.append({
                            'Date': workover['Date'],
                            'Event': 'Workover',
                            'Type': workover.get('WorkoverType', 'Workover'),
                            'Color': 'purple'
                        })
                    
                    timeline_df = pd.DataFrame(timeline_data)
                    timeline_df = timeline_df.sort_values('Date')
                    
                    # Create timeline chart
                    fig_timeline = px.scatter(
                        timeline_df,
                        x='Date',
                        y='Event',
                        color='Event',
                        hover_data=['Type'],
                        title=f"Treatment and Workover Timeline - {selected_well}",
                        color_discrete_map={'Treatment': 'orange', 'Workover': 'purple'}
                    )
                    
                    fig_timeline.update_traces(marker_size=12)
                    fig_timeline.update_layout(height=300)
                    
                    st.plotly_chart(fig_timeline, use_container_width=True)
                    
            else:
                st.info(f"No workover records found for well {selected_well}")
        else:
            st.info("No workover data loaded")
    
    tab_index += 1
    
    # =============================================================================
    # TAB 5: PROGRAM SUMMARY📋
    # =============================================================================

    with tabs[tab_index]:
        st.subheader("Microbe Treatment Program Summary")
        
        # ADDED: Post-Treatment Window Selector
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("**Analysis Configuration:**")
        with col2:
            # Post-treatment window options (30-day increments from 30 to 180 days)
            window_options = [30, 60, 90, 120, 150, 180]
            post_window_labels = [f"{days} days" for days in window_options] + ["All available data"]
            
            selected_window = st.selectbox(
                "Post-Treatment Analysis Window",
                post_window_labels,
                index=2,  # Default to 90 days
                help="Select the number of days after treatment to analyze. 'All available data' uses the maximum available period for each well."
            )
            
            # Convert selection to post_days parameter
            if selected_window == "All available data":
                post_days_param = None
                window_text = "All available data"
            else:
                post_days_param = int(selected_window.split()[0])
                window_text = selected_window
        
        with st.spinner("Computing program summary..."):
            # MODIFIED: Pass post_days_param to compute_all_well_summaries
            summary_df = compute_all_well_summaries(prod_history, treatments, workovers, post_days=post_days_param)
            
            if not summary_df.empty:
                # Overall program metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_wells = len(summary_df)
                    st.metric("Total Wells Treated", total_wells)
                
                with col2:
                    positive_response = len(summary_df[summary_df['Oil Change %'] > 0])
                    success_rate = (positive_response / total_wells) * 100 if total_wells > 0 else 0
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                with col3:
                    avg_oil_change = summary_df['Oil Change %'].mean()
                    st.metric("Avg Oil Change", f"{avg_oil_change:.1f}%")
                
                with col4:
                    best_performer = summary_df.loc[summary_df['Oil Change %'].idxmax(), 'Well']
                    best_change = summary_df['Oil Change %'].max()
                    st.metric("Best Performer", f"{best_performer} ({best_change:.1f}%)")
                
                # MODIFIED: Updated caption to reflect selected analysis window
                st.caption(f"Analysis Complete Via the Following Period: All non-zero days before treatment · Post-window: {window_text}")
                
                # Distribution charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Oil change distribution
                    fig_dist = px.histogram(
                        summary_df,
                        x='Oil Change %',
                        nbins=20,
                        title="Distribution of Oil Production Changes",
                        labels={'Oil Change %': 'Oil Change (%)', 'count': 'Number of Wells'}
                    )
                    fig_dist.add_vline(x=0, line_dash="dash", line_color="red", 
                                    annotation_text="No Change")
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                with col2:
                    # Success/failure pie chart
                    success_data = pd.DataFrame({
                        'Response': ['Positive', 'Negative'],
                        'Count': [
                            len(summary_df[summary_df['Oil Change %'] > 0]),
                            len(summary_df[summary_df['Oil Change %'] <= 0])
                        ]
                    })
                    
                    fig_success = px.pie(
                        success_data,
                        values='Count',
                        names='Response',
                        title="Treatment Response Distribution",
                        color_discrete_map={'Positive': 'green', 'Negative': 'red'}
                    )
                    st.plotly_chart(fig_success, use_container_width=True)
                
                # ADDED: Analysis window summary info
                st.info(f"📊 **Analysis Window**: Using {window_text} post-treatment for all wells. Wells with insufficient data (< {MIN_DATA_DAYS} days) are excluded from summary.")
                
                # Top performers table
                st.subheader("Top 10 Performing Wells")
                top_performers = summary_df.nlargest(10, 'Oil Change %')
                st.dataframe(top_performers, use_container_width=True)
                
                # Full summary table
                if st.checkbox("Show all well results"):
                    st.subheader("Complete Program Results")
                    st.dataframe(summary_df.sort_values('Oil Change %', ascending=False), use_container_width=True)
                    
                    # Download option - MODIFIED: Include window info in filename
                    csv = summary_df.to_csv(index=False)
                    filename_suffix = f"_{post_days_param}days" if post_days_param else "_all_data"
                    st.download_button(
                        label="Download Complete Results as CSV",
                        data=csv,
                        file_name=f"microbe_treatment_summary{filename_suffix}.csv",
                        mime="text/csv"
                    )
            else:
                st.error("Unable to generate program summary. Check data availability.")

    tab_index += 1
    
    # =============================================================================
    # TAB 6: ECONOMICS 💰 - NEW TAB
    # =============================================================================
    with tabs[tab_index]:
        st.subheader("Microbe Treatment Economics")
        
        with st.spinner("Computing economic analysis..."):
            # Treatment cost per application
            treatment_cost = 450
            
            # Apply outlier filtering if enabled
            if exclude_outliers:
                economics_treatments = treatments[~treatments['Well'].isin(OUTLIER_WELLS)]
            else:
                economics_treatments = treatments
            
            # Total program costs
            total_treatments = len(economics_treatments)
            total_cost = total_treatments * treatment_cost
            
            # Overall program metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Treatments", total_treatments)
            
            with col2:
                st.metric("Cost per Treatment", f"${treatment_cost:,.0f}")
            
            with col3:
                st.metric("Total Program Cost", f"${total_cost:,.0f}")
            
            with col4:
                if total_treatments > 0:
                    # Calculate treatment timespan
                    min_date = pd.to_datetime(economics_treatments['TreatmentDate']).min()
                    max_date = pd.to_datetime(economics_treatments['TreatmentDate']).max()
                    months_span = ((max_date - min_date).days / 30.44) + 1  # Add 1 to include both end months
                    avg_monthly_cost = total_cost / months_span if months_span > 0 else 0
                    st.metric("Avg Monthly Cost", f"${avg_monthly_cost:,.0f}")
                else:
                    st.metric("Avg Monthly Cost", "$0")
            
            # Monthly cost breakdown
            if not economics_treatments.empty:
                st.subheader("Monthly Treatment Costs")
                
                # Create monthly breakdown
                economics_treatments_copy = economics_treatments.copy()
                economics_treatments_copy['TreatmentDate'] = pd.to_datetime(economics_treatments_copy['TreatmentDate'])
                economics_treatments_copy['Month'] = economics_treatments_copy['TreatmentDate'].dt.to_period('M')
                
                # Group by month and calculate costs
                monthly_summary = economics_treatments_copy.groupby('Month').agg({
                    'Well': 'count',  # Count of treatments
                    'TreatmentDate': 'count'  # Alternative count for verification
                }).rename(columns={'Well': 'Treatments'})
                
                monthly_summary['Monthly Cost'] = monthly_summary['Treatments'] * treatment_cost
                monthly_summary = monthly_summary.drop('TreatmentDate', axis=1)
                monthly_summary = monthly_summary.reset_index()
                monthly_summary['Month'] = monthly_summary['Month'].astype(str)
                
                # Display monthly cost table
                st.dataframe(monthly_summary, use_container_width=True)
                
                # Monthly cost chart
                if len(monthly_summary) > 1:
                    fig_monthly = px.line(
                        monthly_summary,
                        x='Month',
                        y='Monthly Cost',
                        title="Monthly Treatment Costs Over Time",
                        labels={'Monthly Cost': 'Cost ($)', 'Month': 'Month'},
                        markers=True
                    )
                    fig_monthly.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig_monthly, use_container_width=True)
                    
                    # Additional monthly bar chart
                    fig_monthly_bar = px.bar(
                        monthly_summary,
                        x='Month',
                        y='Treatments',
                        title="Number of Treatments per Month",
                        labels={'Treatments': 'Number of Treatments', 'Month': 'Month'}
                    )
                    fig_monthly_bar.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig_monthly_bar, use_container_width=True)
                
                # Cost per well analysis
                st.subheader("Cost Analysis by Well")
                
                # Group by well and get treatment info
                well_costs = economics_treatments_copy.groupby('Well').agg({
                    'TreatmentDate': ['count', 'min', 'max']
                }).reset_index()

                # Flatten column names
                well_costs.columns = ['Well', 'Treatments', 'First_Treatment', 'Last_Treatment']

                # Calculate total cost
                well_costs['Total Cost'] = well_costs['Treatments'] * treatment_cost

                # Calculate average monthly cost per well
                def calculate_avg_monthly_cost(row):
                    """Calculate average monthly cost from first treatment to last production day"""
                    well_name = row['Well']
                    first_treatment = row['First_Treatment']
                    
                    # Get last production date for this well
                    well_prod = prod_history[prod_history['Well'] == well_name]
                    if well_prod.empty:
                        return 0
                    
                    last_production = well_prod['Date'].max()
                    
                    # Calculate months from first treatment to last production
                    months_active = ((last_production - first_treatment).days / 30.44)
                    
                    # Avoid division by zero
                    if months_active <= 0:
                        return row['Total Cost']  # If less than a month, return total cost
                    
                    return row['Total Cost'] / months_active

                well_costs['Avg Monthly Cost'] = well_costs.apply(calculate_avg_monthly_cost, axis=1)
                well_costs = well_costs.sort_values('Total Cost', ascending=False)
                
                # Summary stats for well costs
                col1, col2, col3, col4 = st.columns(4)  # Change from 3 to 4 columns
                with col1:
                    st.metric("Wells Treated", len(well_costs))
                with col2:
                    avg_cost_per_well = well_costs['Total Cost'].mean()
                    st.metric("Avg Cost per Well", f"${avg_cost_per_well:,.0f}")
                with col3:
                    max_cost_well = well_costs.loc[well_costs['Total Cost'].idxmax(), 'Well']
                    max_cost = well_costs['Total Cost'].max()
                    st.metric("Highest Cost Well", f"{max_cost_well} (${max_cost:,.0f})")
                with col4:
                    avg_monthly_cost = well_costs['Avg Monthly Cost'].mean()
                    st.metric("Avg Monthly Cost/Well", f"${avg_monthly_cost:,.0f}")
                
                # Well cost distribution
                fig_well_dist = px.histogram(
                    well_costs,
                    x='Total Cost',
                    nbins=15,
                    title="Distribution of Treatment Costs by Well",
                    labels={'Total Cost': 'Total Cost per Well ($)', 'count': 'Number of Wells'}
                )
                st.plotly_chart(fig_well_dist, use_container_width=True)
                
                # Show detailed well costs if requested
                # Format the dataframe for display
                if st.checkbox("Show detailed well costs"):
                    display_well_costs = well_costs.copy()
                    display_well_costs['Total Cost'] = display_well_costs['Total Cost'].apply(lambda x: f"${x:,.0f}")
                    display_well_costs['Avg Monthly Cost'] = display_well_costs['Avg Monthly Cost'].apply(lambda x: f"${x:,.0f}")
                    # Remove intermediate columns for cleaner display
                    display_well_costs = display_well_costs[['Well', 'Treatments', 'Total Cost', 'Avg Monthly Cost']]
                    
                    st.dataframe(display_well_costs, use_container_width=True)
                
                # Economic summary note
                st.info(
                    f"💡 **Economic Summary**: The microbe treatment program has invested "
                    f"${total_cost:,.0f} across {total_treatments} treatments on {len(well_costs)} wells. "
                    f"This represents an average investment of ${avg_cost_per_well:,.0f} per well."
                )
                
            else:
                st.warning("No treatment data available for economic analysis.")

if __name__ == "__main__":
    main()
# =============================================================================
