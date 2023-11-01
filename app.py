import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st
from datetime import datetime  # Import the datetime module

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

# Subheader and input for predicting tomorrow's closing price
st.subheader("Predict Tomorrow's Closing Price")
most_recent_price = st.number_input("Enter the most recent closing price:", min_value=0.0)

# Button to trigger the prediction
if st.button("Predict Tomorrow's Closing Price"):
    most_recent_data = pd.Series([most_recent_price])
    most_recent_scaled_data = scaler.fit_transform(most_recent_data)
    next_day_prediction = model.predict(np.array([most_recent_scaled_data]))
    next_day_predicted_price = next_day_prediction[0][0] * scaler.scale_[0]
    st.write(f"Predicted Closing Price for Tomorrow: {next_day_predicted_price:.2f}")

