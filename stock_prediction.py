# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
import plotly.graph_objects as go
import streamlit as st

# Fetch historical stock data
def fetch_data(ticker, start_date):
    data = yf.download(ticker, start=start_date)
    if data.empty:
        raise ValueError("Invalid ticker or no data found.")
    return data

# Add technical indicators
def add_technical_indicators(data):
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    return data

# Preprocess data for LSTM
def preprocess_data(data, sequence_length=50):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']])
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i])
    return np.array(X), np.array(y), scaler

# Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Streamlit app interface
st.title("Stock Price Prediction App")

# User inputs
ticker = st.text_input("Enter Stock Ticker (e.g., TSLA):", "AAPL")
start_date = st.date_input("Start Date")

if st.button("Fetch Data"):
    try:
        # Fetch data
        data = fetch_data(ticker, start_date)
        data = add_technical_indicators(data)

        # Display historical data
        st.write("### Historical Stock Data")
        st.dataframe(data.tail())

        # Plot historical prices with SMA
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='50-Day SMA', line=dict(dash='dash', color='orange')))
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_200'], mode='lines', name='200-Day SMA', line=dict(dash='dash', color='green')))
        fig.update_layout(title="Stock Price Chart with SMA", xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig)

        # Preprocess data and build the model
        sequence_length = 50
        if len(data) < sequence_length:
            st.error("Not enough data to generate predictions. Try a later start date.")
        else:
            X, y, scaler = preprocess_data(data, sequence_length)
            model = build_lstm_model((X.shape[1], X.shape[2]))
            model.fit(X, y, epochs=10, batch_size=32, verbose=0)

            # Predict next 30 days
            forecast = []
            input_seq = X[-1]
            for _ in range(30):
                pred = model.predict(input_seq.reshape(1, X.shape[1], X.shape[2]))
                forecast.append(pred[0, 0])
                input_seq = np.append(input_seq[1:], pred, axis=0)

            # Scale back predictions
            forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
            forecast_dates = pd.date_range(start=data.index[-1], periods=30, freq='D')

            # Display forecasted prices
            forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Prices': forecast.flatten()})
            st.write("### Forecasted Stock Prices for the Next 30 Days")
            st.dataframe(forecast_df)

            # Plot forecasted prices
            forecast_fig = go.Figure()
            forecast_fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecasted Prices'], mode='lines+markers', name='Forecast'))
            forecast_fig.update_layout(title="Forecasted Stock Prices", xaxis_title="Date", yaxis_title="Price (USD)")
            st.plotly_chart(forecast_fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")
