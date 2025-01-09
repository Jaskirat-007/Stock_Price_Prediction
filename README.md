# Stock_Price_Prediction

A machine learning project that predicts future stock prices using historical data and an LSTM model, enhanced with technical indicators like SMA for better analysis.

## Features

- Fetches historical stock data using Yahoo Finance.
- Calculates technical indicators like 50-day and 200-day Simple Moving Averages (SMA).
- Predicts future stock prices using an LSTM model.
- Displays visualizations of historical data, technical indicators, and forecasted prices.
- User-friendly interface built with Streamlit.

## Files

- **`stock_prediction.py`**: Contains the logic for fetching data, adding indicators, preprocessing data, and building the LSTM model.
- **`frontend.py`**: Implements the Streamlit interface for user interaction and data visualization.
- **`housing.csv`**: Dataset placeholder (not used in this project; replace with stock-related data if necessary).

## Requirements

- Python 3.7 or higher
- Required libraries:
  - `pandas`
  - `numpy`
  - `yfinance`
  - `streamlit`
  - `scikit-learn`
  - `tensorflow`
  - `plotly`
  - `matplotlib`

Install dependencies using:
```bash
pip install -r requirements.txt

