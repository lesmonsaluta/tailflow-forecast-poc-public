# 📦 Tailflow SKU Forecasting

A streamlined SKU demand forecasting dashboard with two-state flow: Historical analysis and Forecast mode.

## 🚀 Features

### 📊 Two-State Flow
- **Historical Mode**: View SKU trends and select which ones to display
- **Forecast Mode**: Enable forecasting to see future projections with Prophet

### 📈 Interactive Dashboard
- **SKU Table**: Shows all SKUs with optional forecasted sales column
- **Checkbox Selection**: Choose which SKUs to display in the chart
- **Interactive Chart**: Historical data (blue) + Forecast data (orange) with confidence intervals
- **Real-time Updates**: Chart updates based on SKU selections

### 🔮 Forecasting Capabilities
- **Prophet Integration**: Advanced time series forecasting
- **Configurable Date Range**: Select forecast start and end dates
- **Progress Tracking**: Visual progress bar during forecast generation
- **Confidence Intervals**: Forecast uncertainty visualization

## 📊 Data Structure

The project uses `sku_data.csv` with:
- **Dates as rows**: Each row represents a day
- **SKUs as columns**: Each column represents a different SKU
- **Demand values**: Integer values representing daily demand

### Sample Data Format
```csv
date,SKU-A001-Electronics,SKU-B002-Clothing,SKU-C003-Home,...
2022-06-01,52,118,82,...
2022-06-02,48,125,78,...
...
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd tailflow-forecasting
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

## 🏃 Usage

**Run the Streamlit app**:
```bash
uv run streamlit run main.py
```

Then open your browser to `http://localhost:8501`

## 📈 Two-State Flow

### 🔵 Historical Mode (Default)
- **SKU Table**: Shows all SKUs with null forecasted sales
- **Chart**: Blue lines showing historical demand trends
- **Selection**: Use checkboxes to select which SKUs to display

### 🟠 Forecast Mode (Enabled)
1. **Enable Forecasting**: Check the "Enable Forecasting" checkbox
2. **Set Date Range**: Choose forecast start and end dates
3. **Generate Forecasts**: Prophet models are automatically generated
4. **View Results**: 
   - SKU table shows forecasted sales for the selected period
   - Chart shows historical (blue) + forecast (orange) with confidence intervals
   - Forecast summary table with totals and averages

## 🎨 Visual Design

- **Historical Data**: Blue solid lines
- **Forecast Data**: Orange dashed lines with confidence intervals
- **Interactive Hover**: Detailed information on hover
- **Responsive Layout**: Adapts to different screen sizes

## 📦 Mock Data

The included mock data contains 8 SKUs with realistic patterns:
- `SKU-A001-Electronics` - Growing trend with seasonal patterns
- `SKU-B002-Clothing` - Declining trend with strong seasonality
- `SKU-C003-Home` - Steady growth with monthly cycles
- `SKU-D004-Sports` - High volume with weekly patterns
- `SKU-E005-Books` - Low volume, declining trend
- `SKU-F006-Beauty` - Moderate growth with seasonal spikes
- `SKU-G007-Automotive` - Stable demand with noise
- `SKU-H008-Toys` - Growing trend with holiday seasonality

## 🔧 Dependencies

- `streamlit` - Web dashboard framework
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `plotly` - Interactive visualizations
- `prophet` - Time series forecasting

## 📁 Project Structure

```
tailflow-forecasting/
├── main.py          # Main Streamlit application
├── sku_data.csv     # SKU demand data (dates × SKUs)
├── pyproject.toml   # Project dependencies
└── README.md        # This file
```

## 🎯 Use Cases

- **Demand Planning**: Forecast future demand for inventory optimization
- **Sales Forecasting**: Predict sales performance for different SKUs
- **Trend Analysis**: Compare historical vs. forecasted trends
- **Capacity Planning**: Understand future resource requirements
- **Budget Planning**: Forecast revenue based on demand projections

## 📝 Using Your Own Data

To use your own SKU data:

1. **Replace `sku_data.csv`** with your data file
2. **Ensure the format** matches: dates as rows, SKUs as columns
3. **Date column** should be the index with format YYYY-MM-DD
4. **SKU columns** should contain numeric demand values

The app will automatically detect and load your SKU columns.

## 🎨 Code Structure

The application is organized with separate functions for easy modification:

- `generate_mock_sku_data()` - Creates synthetic SKU data
- `load_sku_data()` - Loads data from CSV with fallback to mock data
- `generate_sku_forecast()` - Creates Prophet forecast for a single SKU
- `create_sku_trend_chart()` - Generates interactive Plotly chart with optional forecast
- `create_sku_table()` - Builds SKU table with optional forecast values

This modular structure makes it easy to:
- Swap in real data
- Modify forecasting parameters
- Customize visualization styles
- Add new features

## 🔮 Forecasting Details

### Prophet Model Configuration
- **Yearly Seasonality**: Enabled for annual patterns
- **Weekly Seasonality**: Enabled for weekly cycles
- **Daily Seasonality**: Disabled for better performance
- **Seasonality Mode**: Additive (can be modified in code)

### Forecast Validation
- **Date Range Validation**: Ensures forecast dates are in the future
- **Error Handling**: Graceful handling of forecasting failures
- **Progress Tracking**: Real-time progress during forecast generation

## 🤝 Contributing

Feel free to contribute by:
- Adding new forecasting models
- Improving the UI/UX
- Adding data export functionality
- Enhancing error handling
- Adding more statistical metrics

## 📄 License

This project is open source and available under the MIT License.
