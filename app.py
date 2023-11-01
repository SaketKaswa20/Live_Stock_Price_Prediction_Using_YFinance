import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st
from datetime import datetime, timedelta  # Import the datetime module

st.title('Stock Market Prediction')

start = '2010-01-01'
end = datetime.today().strftime('%Y-%m-%d')  # Get the current date and format it

df = yf.download('AAPL', start, end)
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start, end)

#Describing Data
st.subheader('Data from 2010 to 2022')
st.write(df.describe())

#Visualizations
st.subheader('Closing Price v/s Time Chart')
fig= plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price v/s Time Chart with 100 Moving Average')
ma100= df.Close.rolling(100).mean()
fig= plt.figure(figsize=(12,6))
plt.plot(ma100, label= 'MA 100')
plt.plot(df.Close, label= 'Closing Price')
st.pyplot(fig)

st.subheader('Closing Price v/s Time Chart with 100 & 200 Moving Average')
ma100= df.Close.rolling(100).mean()
ma200= df.Close.rolling(200).mean()
fig= plt.figure(figsize=(12,6))
plt.plot(ma100, label='MA 100')
plt.plot(ma200, label='MA 200')
plt.plot(df.Close, label='Closing Price')
st.pyplot(fig)

#Splitting Data into Training and Testing

data_training= pd.DataFrame(df['Close'][0:int(len(df)*0.70)]) #Keeping 70% Data for training that's why 0.7
data_testing= pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler #To Scale data between 0 and 1
scaler= MinMaxScaler(feature_range=(0,1))

data_training_array= scaler.fit_transform(data_training)

#Load my model
model= load_model('keras_model.h5')

#Testing Part
past_100_days=data_training.tail(100)
#final_df= past_100_days.append(data_testing, ignore_index=True)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100: i])
  y_test.append(input_data[i, 0])

x_test, y_test= np.array(x_test), np.array(y_test)
y_predicted= model.predict(x_test)

scaler= scaler.scale_

scale_factor=1/scaler[0]
y_predicted= y_predicted * scale_factor
y_test= y_test * scale_factor

#Final Visualization
st.subheader('Predictions v/s Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label= 'Original Price')
plt.plot(y_predicted, 'r', label= 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# Predict tomorrow's price based on today's data

if not data_testing.empty:
    last_data = data_testing.iloc[-1]  # Access the last row
    last_scaled_price = np.array([last_data])
    x_test_tomorrow = []
    x_test_tomorrow.append(last_scaled_price)
    x_test_tomorrow = np.array(x_test_tomorrow)

    # Make predictions for tomorrow
    y_predicted_tomorrow = model.predict(x_test_tomorrow)

    # Scale the prediction back to the original scale
    y_predicted_tomorrow = y_predicted_tomorrow * scale_factor

    # Calculate tomorrow's date
    tomorrow = datetime.today() + pd.DateOffset(days=1)

    # Display today's closing price
    st.subheader("Today's Closing Price")
    st.write("Closing Price for Today:", last_data[0])

    # Display tomorrow's predicted price
    st.subheader("Tomorrow's Predicted Price")
    st.write("Predicted Closing Price for Tomorrow (Based on Today's Data):", y_predicted_tomorrow[0][0])
    st.write("Date:", tomorrow.strftime('%Y-%m-%d'))
else:
    st.subheader("Tomorrow's Predicted Price")
    st.write("No data available for prediction. Please make sure you have historical data.")
