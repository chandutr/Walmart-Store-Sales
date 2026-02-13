import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

st.title("Walmart Store Sales Prediction")

@st.cache_data
def load_data():
    df = pd.read_csv("Walmart DataSet.csv")
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    df.fillna(method='ffill', inplace=True)
    return df

df = load_data()

features = ['Store','Holiday_Flag','Temperature','Fuel_Price',
            'CPI','Unemployment','Year','Month','Week']

X = df[features]
y = df['Weekly_Sales']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_scaled, y)

st.subheader("Enter Store Details")

store = st.number_input("Store ID")
holiday = st.number_input("Holiday Flag")
temperature = st.number_input("Temperature")
fuel_price = st.number_input("Fuel Price")
cpi = st.number_input("CPI")
unemployment = st.number_input("Unemployment")
year = st.number_input("Year")
month = st.number_input("Month")
week = st.number_input("Week")

if st.button("Predict Weekly Sales"):
    input_data = np.array([[store, holiday, temperature,
                            fuel_price, cpi, unemployment,
                            year, month, week]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.success(f"Predicted Weekly Sales: ${prediction[0]:,.2f}")

if st.button("Predict Monthly Sales"):

    monthly_sales = 0

    for w in range(int(week), int(week) + 4):
        input_data = np.array([[store, holiday, temperature,
                                fuel_price, cpi, unemployment,
                                year, month, w]])

        input_scaled = scaler.transform(input_data)
        weekly_prediction = model.predict(input_scaled)

        monthly_sales += weekly_prediction[0]

    st.success(f"Predicted Monthly Sales: ${monthly_sales:,.2f}")
