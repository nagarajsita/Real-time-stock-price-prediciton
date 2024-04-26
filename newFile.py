import asyncio
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import requests
from bs4 import BeautifulSoup
import time

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Streamlit app
symbol = st.sidebar.text_input('Stock Symbol')
st.title(symbol)

markets = {
    'NSE': f'https://www.google.com/finance/quote/{symbol}:NSE',  # Replace with NSE URL
    'BSE': f'https://www.google.com/finance/quote/{symbol}:BOM',  # Replace with BSE URL
    'NASDAQ': f'https://www.google.com/finance/quote/{symbol}:NASDAQ',  # Replace with NASDAQ URL
}

market = st.sidebar.selectbox('Select Market', list(markets.keys()))
url = markets[market]

def get_live_price():
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    class1 = "YMlKec fxKbKc"
    return soup.find(class_=class1).text

if(market=='NSE'):
    symbol2=symbol+'.NS'
elif(market=='BSE'):
    symbol2=symbol+'.BO'
elif(market=='NASDAQ'):
    symbol2=symbol


data=yf.download(symbol2,period='1y',interval="1d")
fundamentals, pricing_data, prediction = st.tabs(["Fundamentals", "Historical Data", "Predictions"])

with pricing_data:
    st.header('Stock Price Data')
    st.write(data)

with fundamentals:
    st.header('Stock Fundamentals')
    try:
        company_info = yf.Ticker(symbol2).info
        df_fundamentals = pd.DataFrame(company_info.items(), columns=['Metric', 'Value'])

        selected_fundamentals = [
            'regularMarketVolume', 'shortName', 'fiftyTwoWeekHigh', 'fiftyTwoWeekLow', 'sector', 'industry', 'website',
            'marketCap', 'trailingPE', 'forwardPE', 'priceToBook', 'dividendYield',
        ]
        df_fundamentals = df_fundamentals[df_fundamentals['Metric'].isin(selected_fundamentals)]
        df_fundamentals = df_fundamentals.reset_index(drop=True)  # Reset index to remove row numbers
        df_fundamentals = df_fundamentals.rename(columns={'Metric': 'Description', 'Value': 'Value'})
        st.table(df_fundamentals)
    except (KeyError, ValueError) as e:
        st.error(f"Error retrieving fundamentals: {e}")
        st.info("Some fundamental data may not be available, but historical data is still accessible.")

# Live price update (moved outside prediction tab)
live_price = st.empty()
live_price.subheader(f"Live Price: {get_live_price()}")

with prediction:
    closing_prices = data['Close']
    closing_prices.to_numpy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(closing_prices.values.reshape(-1, 1))

    # Function to create dataset
    def create_dataset(data, time_step):
        X, y = [], []
        if isinstance(data, pd.Series):
            data = data.to_numpy()  # Convert to NumPy array if it's a Series
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)


    # Define time steps and split data into train and test sets
    time_step = 100
    X_train, y_train = create_dataset(scaled_data, time_step)

    # Reshape input to be [samples, time steps, features] for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=5, batch_size=64)

    # Make predictions
    # Assuming you have real-time data stored in 'real_time_data'
    # Replace 'real_time_data' with your actual real-time data
    # scaled_real_time_data = scaler.transform(closing_prices.reshape(-1, 1))
    X_test, y_test = create_dataset(scaled_data, time_step)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    predicted_stock_price = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_stock_price)
    st.subheader(f"Predicted Stock Prices: {predicted_prices[0].flatten()}")

  