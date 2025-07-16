import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Tailflow SKU Forecasting",
    page_icon="📦",
    layout="wide"
)

# Title
st.title("📦 Tailflow SKU Forecasting Dashboard")
st.markdown("SKU demand forecasting and trend analysis")

def generate_mock_sku_data():
    """Generate mock SKU data with dates as rows and SKUs as columns"""
    np.random.seed(42)
    
    # Create date range - 18 months of daily data
    start_date = datetime(2022, 6, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Mock SKU names
    skus = [
        'SKU-A001-Electronics',
        'SKU-B002-Clothing', 
        'SKU-C003-Home',
        'SKU-D004-Sports',
        'SKU-E005-Books',
        'SKU-F006-Beauty',
        'SKU-G007-Automotive',
        'SKU-H008-Toys'
    ]
    
    # Create base data for each SKU
    sku_data = {}
    
    for i, sku in enumerate(skus):
        # Different base demand levels for each SKU
        base_demand = [50, 120, 80, 200, 30, 90, 60, 150][i]
        
        # Create trend and seasonality for each SKU
        n_days = len(dates)
        t = np.arange(n_days)
        
        # Trend (some SKUs growing, some declining)
        trend_rates = [0.02, -0.01, 0.03, 0.01, -0.02, 0.04, 0.00, 0.02]
        trend = base_demand * (1 + trend_rates[i] * t / 365)
        
        # Seasonal patterns (different for each SKU)
        yearly_seasonal = [10, 25, 15, 30, 8, 20, 12, 35][i] * np.sin(2 * np.pi * t / 365.25 + i)
        weekly_seasonal = [5, 8, 4, 12, 3, 6, 5, 10][i] * np.sin(2 * np.pi * t / 7 + i)
        
        # Monthly patterns (some SKUs have monthly cycles)
        monthly_seasonal = [3, 6, 2, 8, 2, 4, 3, 7][i] * np.sin(2 * np.pi * t / 30.44 + i)
        
        # Random noise
        noise = np.random.normal(0, base_demand * 0.1, n_days)
        
        # Combine all components and ensure non-negative
        values = trend + yearly_seasonal + weekly_seasonal + monthly_seasonal + noise
        values = np.maximum(values, 0)  # Ensure non-negative demand
        
        sku_data[sku] = np.round(values, 0).astype(int)
    
    # Create DataFrame with dates as index and SKUs as columns
    df = pd.DataFrame(sku_data, index=dates)
    df.index.name = 'date'
    
    return df

def load_sku_data():
    """Load SKU data from CSV file, fallback to mock data"""
    try:
        # Try to load from CSV first
        df = pd.read_csv('sku_data.csv', index_col=0, parse_dates=True)
        return df
    except FileNotFoundError:
        # Generate mock data if CSV doesn't exist
        return generate_mock_sku_data()

def generate_sku_forecast(df, sku, forecast_days=30):
    """Generate forecast for a single SKU using Prophet"""
    # Prepare data for Prophet
    sku_data = df[[sku]].reset_index()
    sku_data.columns = ['ds', 'y']
    
    # Initialize and fit Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive'
    )
    
    # Suppress Prophet logging
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(sku_data)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    
    return forecast

def create_sku_trend_chart(df, selected_skus, forecast_data=None, chart_height=600):
    """Create interactive Plotly chart for selected SKUs with optional forecast"""
    fig = go.Figure()
    
    # Add historical data (blue) for each selected SKU
    for sku in selected_skus:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[sku],
            mode='lines',
            name=f'{sku} (Historical)',
            line=dict(width=2, color='blue'),
            hovertemplate='<b>' + sku + '</b><br>' +
                         'Date: %{x}<br>' +
                         'Demand: %{y}<br>' +
                         '<extra></extra>'
        ))
    
    # Add forecast data (orange) if provided
    if forecast_data is not None:
        for sku in selected_skus:
            if sku in forecast_data:
                forecast = forecast_data[sku]
                # Get forecast portion only (future dates)
                forecast_portion = forecast[forecast['ds'] > pd.Timestamp(df.index.max())]
                
                fig.add_trace(go.Scatter(
                    x=forecast_portion['ds'],
                    y=forecast_portion['yhat'],
                    mode='lines',
                    name=f'{sku} (Forecast)',
                    line=dict(width=2, color='orange', dash='dash'),
                    hovertemplate='<b>' + sku + ' (Forecast)</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Forecast: %{y:.0f}<br>' +
                                 '<extra></extra>'
                ))
                
                # Add confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast_portion['ds'],
                    y=forecast_portion['yhat_upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_portion['ds'],
                    y=forecast_portion['yhat_lower'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255,165,0,0.2)',
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # Update layout
    title = 'SKU Demand Trends'
    if forecast_data is not None:
        title += ' with Forecast'
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Demand',
        height=chart_height,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def create_sku_table(df, forecast_data=None, forecast_start=None, forecast_end=None):
    """Create SKU table with checkboxes and forecast values"""
    skus = list(df.columns)
    
    # Initialize table data
    table_data = []
    
    for sku in skus:
        row = {'SKU': sku}
        
        # Add forecast values if forecast data is provided
        if forecast_data is not None and sku in forecast_data and forecast_start and forecast_end:
            forecast = forecast_data[sku]
            forecast_period = forecast[
                (forecast['ds'].dt.date >= forecast_start) & 
                (forecast['ds'].dt.date <= forecast_end)
            ]
            total_forecast = forecast_period['yhat'].sum()
            row['Forecasted Sales'] = f"{total_forecast:.0f}"
        else:
            row['Forecasted Sales'] = None
            
        table_data.append(row)
    
    return pd.DataFrame(table_data)

# Load data
df_sku = load_sku_data()

# Sidebar configuration
st.sidebar.header("📊 Configuration")

# Forecast date range selection
st.sidebar.subheader("🔮 Forecast Settings")
enable_forecast = st.sidebar.checkbox(
    "Enable Forecasting",
    value=False,
    help="Enable forecasting to see future projections"
)

forecast_data = None
forecast_start = None
forecast_end = None

if enable_forecast:
    # Date range for forecast
    min_forecast_date = df_sku.index.max() + timedelta(days=1)
    max_forecast_date = df_sku.index.max() + timedelta(days=365)
    
    st.sidebar.write("**Forecast Date Range:**")
    forecast_start = st.sidebar.date_input(
        "Forecast Start Date",
        value=min_forecast_date.date(),
        min_value=min_forecast_date.date(),
        max_value=max_forecast_date.date(),
        help="Start date for forecast period"
    )
    
    forecast_end = st.sidebar.date_input(
        "Forecast End Date",
        value=(min_forecast_date + timedelta(days=30)).date(),
        min_value=min_forecast_date.date(),
        max_value=max_forecast_date.date(),
        help="End date for forecast period"
    )
    
    # Validate forecast date range
    if forecast_start >= forecast_end:
        st.sidebar.error("Forecast start date must be before end date!")
        enable_forecast = False
    else:
        forecast_days = (forecast_end - forecast_start).days + 1
        st.sidebar.info(f"📊 Forecast Period: {forecast_days} days")

# Chart configuration (collapsed)
with st.sidebar.expander("📈 Chart Settings"):
    chart_height = st.slider(
        "Chart height (pixels)",
        min_value=400,
        max_value=900,
        value=600,
        help="Height of the trend chart",
        key="chart_height"
    )

# Data info (collapsed)
with st.sidebar.expander("📅 Data Info"):
    st.info(f"📅 Historical Data: {df_sku.index.min().date()} to {df_sku.index.max().date()}")
    st.info(f"📦 Total SKUs: {len(df_sku.columns)}")
    st.info(f"📊 Total Records: {len(df_sku)} days")

# Generate forecasts if enabled
if enable_forecast:
    st.sidebar.write("**Generating forecasts...**")
    forecast_data = {}
    
    progress_bar = st.sidebar.progress(0)
    available_skus = list(df_sku.columns)
    
    for i, sku in enumerate(available_skus):
        try:
            forecast_days_total = (forecast_end - df_sku.index.max().date()).days
            forecast_data[sku] = generate_sku_forecast(df_sku, sku, forecast_days_total)
            progress_bar.progress((i + 1) / len(available_skus))
        except Exception as e:
            st.sidebar.error(f"Error forecasting {sku}: {str(e)}")
    
    st.sidebar.success("✅ Forecasts generated!")

# Main content
st.subheader("📦 SKU Management")

# Create SKU table with forecast data
sku_table = create_sku_table(df_sku, forecast_data, forecast_start, forecast_end)

# Display the state info
if enable_forecast:
    st.info(f"🔮 **Forecast Mode**: Showing forecasted sales for {forecast_start} to {forecast_end}")
else:
    st.info("📊 **Historical Mode**: Select 'Enable Forecasting' to see future projections")

# Create interactive table with checkboxes
st.write("**Select SKUs to display in chart:**")

# Create columns for checkboxes
col1, col2 = st.columns([3, 1])

with col1:
    # Display SKU table
    st.dataframe(sku_table, use_container_width=True)

with col2:
    # Create checkboxes for SKU selection
    st.write("**Chart Selection:**")
    selected_skus = []
    
    for sku in df_sku.columns:
        if st.checkbox(sku, key=f"cb_{sku}", value=True if sku in df_sku.columns[:4] else False):
            selected_skus.append(sku)

# Check if any SKUs are selected
if not selected_skus:
    st.warning("🔍 Please select at least one SKU to display in the chart.")
    st.stop()

# Display chart info
st.subheader("📈 SKU Demand Trends")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Selected SKUs", len(selected_skus))
with col2:
    if enable_forecast:
        st.metric("Chart Mode", "Historical + Forecast")
    else:
        st.metric("Chart Mode", "Historical Only")
with col3:
    total_demand = df_sku[selected_skus].sum().sum()
    st.metric("Total Historical Demand", f"{total_demand:,}")

# Create and display the main trend chart
trend_chart = create_sku_trend_chart(df_sku, selected_skus, forecast_data, chart_height)
st.plotly_chart(trend_chart, use_container_width=True)

# Show forecast summary if enabled
if enable_forecast and forecast_data:
    st.subheader("📊 Forecast Summary")
    
    forecast_summary = []
    for sku in selected_skus:
        if sku in forecast_data:
            forecast = forecast_data[sku]
            forecast_period = forecast[
                (forecast['ds'].dt.date >= forecast_start) & 
                (forecast['ds'].dt.date <= forecast_end)
            ]
            total_forecast = forecast_period['yhat'].sum()
            avg_daily = forecast_period['yhat'].mean()
            
            forecast_summary.append({
                'SKU': sku,
                'Total Forecast': f"{total_forecast:.0f}",
                'Avg Daily': f"{avg_daily:.0f}",
                'Forecast Days': len(forecast_period)
            })
    
    if forecast_summary:
        forecast_df = pd.DataFrame(forecast_summary)
        st.dataframe(forecast_df, use_container_width=True)
        
        # Total forecast across all selected SKUs
        total_forecast_all = sum([float(row['Total Forecast']) for row in forecast_summary])
        st.metric("**Total Forecast (All Selected SKUs)**", f"{total_forecast_all:,.0f}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, Plotly, and Prophet for SKU demand forecasting")

if __name__ == "__main__":
    # This won't run when using streamlit run
    pass
