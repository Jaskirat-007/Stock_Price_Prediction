import streamlit as st
from stock_prediction import fetch_data, add_technical_indicators, preprocess_data, build_lstm_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

st.title("Stock Price Prediction App")

# User Inputs with unique keys
ticker = st.text_input("Enter Stock Ticker (e.g., TSLA):", "AAPL", key="ticker_input")
start_date = st.date_input("Start Date:", key="start_date_input")

if st.button("Fetch Data"):
    try:
        # Fetch and Display Data
        data = fetch_data(ticker, start_date)
        data = add_technical_indicators(data)

        st.write("### Historical Stock Data")
        st.dataframe(data.tail())

        st.write("### Stock Price Chart")
        fig, ax = plt.subplots()
        ax.plot(data['Close'], label='Close Price')
        ax.plot(data['SMA_50'], label='50-Day SMA', linestyle='--')
        ax.plot(data['SMA_200'], label='200-Day SMA', linestyle='--')
        ax.legend()
        st.pyplot(fig)

        # Predict Future Prices
        st.write("### Predicting Future Prices")
        sequence_length = 50  # Ensure the input length for LSTM
        if len(data) < sequence_length:
            st.error("Not enough data to generate predictions. Try a later start date.")
        else:
            # Preprocess data
            X, y, scaler = preprocess_data(data, sequence_length)
            model = build_lstm_model((X.shape[1], X.shape[2]))
            model.fit(X, y, epochs=10, batch_size=32, verbose=0)

            # Forecast Next 30 Days
            forecast = []
            input_seq = X[-1]
            for _ in range(30):
                pred = model.predict(input_seq.reshape(1, X.shape[1], X.shape[2]))
                forecast.append(pred[0, 0])
                input_seq = np.append(input_seq[1:], pred, axis=0)

            # Scale back forecasted prices
            forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
            forecast_dates = pd.date_range(start=data.index[-1], periods=30, freq='D')
            forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast.flatten()})

            # Display Forecast Data
            st.write("### Forecast for the Next 30 Days")
            st.dataframe(forecast_df)

            # Plot Forecast
            st.line_chart(forecast_df.set_index('Date'))

    except Exception as e:
        st.error(f"Error: {e}")
