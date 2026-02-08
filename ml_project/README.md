# 📞 Call Center Performance Dashboard

An interactive Streamlit dashboard for analyzing call center performance metrics.

## Features

- **Overview Dashboard**: Key performance indicators and metrics at a glance
- **Answered vs Abandoned Analysis**: Multiple chart styles (stacked, side-by-side, gradient)
- **Incoming vs Answered Relationship**: Scatter plots with trend lines
- **Performance Pie Charts**: Total summary, categories, and donut charts
- **Heatmap Analysis**: Correlation matrix and time series heatmaps
- **Trend Analysis**: Moving averages and service level tracking

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Data Requirements

Make sure `Call Center Data.csv` is in the same directory as `app.py`

## Dashboard Sections

1. **Overview** - Complete overview with time series and distributions
2. **Answered vs Abandoned** - Compare answered and abandoned calls
3. **Incoming vs Answered** - Analyze relationship between incoming and answered calls
4. **Performance Pie Chart** - Visual breakdown of performance metrics
5. **Heatmap Analysis** - Correlation and pattern analysis
6. **Trend Analysis** - Moving averages and service level trends

## Controls

Use the sidebar to:
- Select different visualization types
- Choose chart styles
- Adjust parameters (like moving average window)

Enjoy analyzing your call center data! 📊
