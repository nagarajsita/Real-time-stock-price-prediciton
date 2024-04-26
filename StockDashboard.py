import streamlit as st,pandas as pd,numpy as np,yfinance as yf
import requests
from bs4 import BeautifulSoup
import time
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, window_size):
    sequences = []
    for i in range(len(data) - window_size + 1):
        window = data[i:(i + window_size)]
        sequences.append(window)
    return np.array(sequences)

symbol = st.sidebar.text_input('Stock Symbol')
st.title(symbol)

markets = {
    'NSE': f'https://www.google.com/finance/quote/{symbol}:NSE',  # Replace with NSE URL
    'BSE': f'https://www.google.com/finance/quote/{symbol}:BOM',  # Replace with BSE URL
    'NASDAQ': f'https://www.google.com/finance/quote/{symbol}:NASDAQ',  # Replace with NASDAQ URL
}

market = st.sidebar.selectbox('Select Market', list(markets.keys()))
url = markets[market]

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
class1 = "YMlKec fxKbKc"
x = soup.find(class_=class1).text
price = x.replace("₹", "").replace(",", "")

st_subheader = st.subheader(x)

# Main loop

if(market=='NSE'):
    symbol2=symbol+'.NS'
elif(market=='BSE'):
    symbol2=symbol+'.BO'
elif(market=='NASDAQ'):
    symbol2=symbol


data=yf.download(symbol2,period='1y',interval="1d")

fundamentals,pricing_data,prediction = st.tabs(["Fundamentals","Historical Data","Predictions"])

with pricing_data:
    st.header('Stock Price Data');
    st.write(data)
    st.line_chart(data['Close'],color="#ff00ff")
    

with fundamentals:
    st.header('Stock Fundamentals')
    try:
        company_info = yf.Ticker(symbol2).info
        df_fundamentals = pd.DataFrame(company_info.items(), columns=['Metric', 'Value'])

        selected_fundamentals = [
            'regularMarketVolume','shortName', 'fiftyTwoWeekHigh','fiftyTwoWeekLow', 'sector', 'industry', 'website',
            'marketCap', 'trailingPE', 'forwardPE', 'priceToBook', 'dividendYield',
        ]
        df_fundamentals = df_fundamentals[df_fundamentals['Metric'].isin(selected_fundamentals)]
        df_fundamentals = df_fundamentals.reset_index(drop=True)  # Reset index to remove row numbers
        df_fundamentals = df_fundamentals.rename(columns={'Metric': 'Description', 'Value': 'Value'})
      
        st.table(df_fundamentals)


    except (KeyError, ValueError) as e:
        st.error(f"Error retrieving fundamentals: {e}")
        st.info("Some fundamental data may not be available, but historical data is still accessible.")



with prediction:
        data1=yf.download(symbol2,period='1mo',interval="1h")
        closing_prices = data1['Close']
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
        time_step = (int)(scaled_data.size/3)
        X_train, y_train = create_dataset(scaled_data, time_step)

        # Reshape input to be [samples, time steps, features] for LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=64, return_sequences=True, input_shape=(time_step, 1)))
        model.add(LSTM(units=64))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=20, batch_size=32)

        
        X_test, y_test = create_dataset(scaled_data, time_step)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        predicted_stock_price = model.predict(X_test)
        predicted_prices = scaler.inverse_transform(predicted_stock_price)
        st.subheader(f"Predicted Stock Prices: {predicted_prices[predicted_prices.size-1].flatten()[0]}")
    
        if(predicted_prices[predicted_prices.size-1].flatten()[0]>float(price)):{
            st.subheader("BUY!!")}
        elif(predicted_prices[predicted_prices.size-1].flatten()[0]==float(price)):{st.subheader("HOLD")}
        else :st.subheader("SELL!!")

  
for i in range(1000):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    x = soup.find(class_=class1).text    
    st_subheader.subheader(x)
    time.sleep(0.01)  