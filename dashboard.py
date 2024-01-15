"""
    Brent Crude Oil Dashboard using Streamlit
    -----------------------------------------
Nama: Diva Putra Almeyda
Email: diva.putra.almeyda@students.untidar.ac.id
"""

# Import Library
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA


# Load Dataset

@st.cache_resource
def load_data():
    df_daily = pd.read_csv('dailybrentoil.csv', parse_dates=['Date'], index_col='Date')
    return df_daily

df_daily = load_data()

# Title
st.title('Brent Crude Oil Dashboard')

# Sidebar
with st.sidebar:
    # Image
    st.image('https://cdn.vectorstock.com/i/1000x1000/23/67/oil-barrel-icon-for-price-forecast-prese-vector-24802367.webp')
    st.sidebar.header('Filter:')
    # Filter by date
    start_date, end_date = st.date_input(
        label='Filter by Date:',
        min_value=df_daily.index.min(),
        max_value=df_daily.index.max(),
        value=(df_daily.index.min(), df_daily.index.max())
        )
    # Filter by price
    price_options = ['Close', 'Low', 'High']
    selected_prices = st.multiselect('Filter by Price:', price_options, default=price_options)


# My Profile
st.sidebar.header('My Profile:')
st.sidebar.markdown('Diva Putra Almeyda')
col1, col2 = st.sidebar.columns(2)
with col1:
    st.markdown("[![LinkedIn](https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg)](https://www.linkedin.com/in/divaalmeyda/)")
with col2:
    st.markdown("[![Github](https://img.icons8.com/glyph-neue/64/FFFFFF/github.png)](https://github.com/Dvaalmeyda)")

# Filtered Data
filtered_data = df_daily.loc[start_date:end_date, selected_prices]

# Helper Function

# Raw & Predicted Price Chart
def plot_price_chart():
    if not selected_prices:
        st.warning('Please select at least one price option.')
        return
    
    # Set palet warna yang berbeda untuk setiap opsi harga dan perubahannya
    color_palette = {
        'Close': 'lightblue',
        'Low': 'lightcoral',
        'High': 'lightgreen',
        'chg(close)': 'dodgerblue',
        'chg(low)': 'tomato',
        'chg(high)': 'limegreen'
    }

    # Create subplot with 2 rows and 1 column
    fig = sp.make_subplots(
        rows=2, cols=1, 
        subplot_titles=('Brent Crude Oil Price',
                        'Change Price'),
        )

    # Plot Price
    for price in selected_prices:
        fig.add_trace(go.Scatter(
            x=filtered_data.index, y=filtered_data[price],
            mode='lines+markers', name=price, 
            line=dict(color=color_palette[price]), marker=dict(size=4)), 
            row=1, col=1,
        )

    # Plot ARIMA Forecast
    for price in selected_prices:
        fig.add_trace(go.Scatter(
            x=arima_forecast_df.index, y=arima_forecast_df['Predicted Price'],
            mode='lines', name=f'ARIMA Forecast ({price})',
            line=dict(color='lightgray', dash='dash'), marker=dict(size=4)),
            row=1, col=1,
        )

    # Plot Confidence Intervals
        fig.add_trace(go.Scatter(
            x=arima_forecast_df.index,
            y=arima_forecast_df['Upper CI'],
            fill='tonexty',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False),
            row=1, col=1,
        )

        fig.add_trace(go.Scatter(
            x=arima_forecast_df.index,
            y=arima_forecast_df['Lower CI'],
            fill='tonexty',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False),
            row=1, col=1,
        )

    # Plot Change in Price
    for price in selected_prices:
        chg_col = f'chg({price.lower()})'
        filtered_data[chg_col] = filtered_data[price].pct_change() * 100
        fig.add_trace(go.Scatter(
            x=filtered_data.index, y=filtered_data[chg_col],
            mode='lines+markers', name=chg_col, 
            line=dict(color=color_palette[chg_col]),  marker=dict(size=4)),
            row=2, col=1)

    # Update layout
    fig.update_layout(height=600, width=800, showlegend=True)

    st.plotly_chart(fig, use_container_width=True)

# Evaluation Chart
def plot_evaluation_chart():
    if not selected_prices:
        st.warning('Please select at least one price option.')
        return
    
    color_palette = {
        'Close': 'lightblue',
        'Low': 'lightcoral',
        'High': 'lightgreen',
    }

    # Create subplot with 2 rows and 1 column
    fig = sp.make_subplots(
        rows=1, cols=1, 
        subplot_titles=('Test Data vs Predictions'),
        )

    # Plot Test Data Price
    for price in selected_prices:
        fig.add_trace(go.Scatter(
            x=test_data.index, y=test_data[selected_price],
            mode='lines+markers', name=price, 
            line=dict(color=color_palette[price]), marker=dict(size=4)), 
            row=1, col=1,
        )

    # Plot ARIMA Test Predictions
    for price in selected_prices:
        fig.add_trace(go.Scatter(
            x=test_data.index, y=predictions,
            mode='lines', name=f'ARIMA Test Data Pred ({price})',
            line=dict(color='orange', dash='dash'), marker=dict(size=4)),
            row=1, col=1,
        )


    # Update layout
    fig.update_layout(height=600, width=800, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)


# ARIMA Prediction
def arima_prediction(model, data, target_col, n_forecast):
    future_dates = pd.date_range(start=data.index[-1],
                                 periods=n_forecast, 
                                 freq='D')[0:]

    forecast = model.get_forecast(steps=n_forecast) 
    forecast_index = pd.date_range(start=future_dates[0],
                                   periods=n_forecast,
                                   freq='D')
    forecast_values = forecast.predicted_mean
    confidence_interval = forecast.conf_int(alpha=0.2) # Conf_Int = 80%

    # Create dataframe
    forecast_df = pd.DataFrame(
        index=forecast_index,
        columns=['Predicted Price','Lower CI','Upper CI']
        )
    
    forecast_df['Predicted Price'] = forecast_values
    forecast_df['Lower CI'] = confidence_interval[:, 0]
    forecast_df['Upper CI'] = confidence_interval[:, 1]
    return forecast_df

# Model Evaluation
def model_evaluation(actual, predictions):
    mae = mean_absolute_error(actual, predictions)
    mape = mean_absolute_percentage_error(actual, predictions)
    return mae, mape

# Load Model
with open('arima_model.pkl', 'rb') as file:
    arima_model = pickle.load(file)

# Number of Forecast Days
n_forecast = st.number_input('Number of Forecast Days', min_value=1, max_value=365, value=30, step=1)

# ARIMA Forecast using the loaded model
arima_forecast_df = arima_prediction(arima_model,
                                     df_daily,
                                     selected_prices,
                                     n_forecast)

# Evaluate the model for each selected price
evaluation_results = {}

# Evaluation loop through each selected price
for selected_price in selected_prices:
    # Set ratio & split index
    train_ratio = 0.8

    # Split data
    train_data, test_data = train_test_split(
        df_daily[[selected_price]], train_size=train_ratio, shuffle=False)
    
    # Test ARIMA Model using the loaded model
    history = [x for x in train_data[selected_price]]
    predictions = list()
    n_pred = len(test_data)

    for i in range(n_pred):
        
        output = arima_model.forecast(30)
        yhat = output[0]
        predictions.append(yhat)
        obs = test_data.values[i]
        history.append(obs)

    # Evaluate the model
    mae, mape = model_evaluation(test_data.values.flatten(), predictions)

    # Store results in a dictionary
    evaluation_results[selected_price] = {
        'Testing MAE': mae,
        'Testing MAPE': mape
    }

# Graph
st.markdown('<h2 style="text-align: center;">Brent Crude Oil Price Forecast</h2>', unsafe_allow_html=True)
plot_price_chart()

# Display evaluation result
st.markdown('<h2 style="text-align: center;">Model Evaluation Results</h2>', unsafe_allow_html=True)
st.markdown('<h4 style="text-align: center;">Model Performance</h4>', unsafe_allow_html=True)

# Create a list to store results for each selected price
results_table = []

for selected_price, results in evaluation_results.items():
    # Append results to the table
    results_table.append({
        'Price': selected_price,
        'MAE': results["Testing MAE"],
        'MAPE': results["Testing MAPE"]
    })

# Convert the results table to a DataFrame
results_df = pd.DataFrame(results_table)
st.markdown('<style>table {margin-left: auto;margin-right: auto;}</style>', unsafe_allow_html=True)

# Display the results table in Streamlit
st.write(results_df)

# Graph
st.markdown('<h4 style="text-align: center;">Evaluation Chart</h4>', unsafe_allow_html=True)
st.markdown('<p>Note: Choose only 1 Price for better experience</p>', unsafe_allow_html=True)
plot_evaluation_chart()

# ----- HIDE STREAMLIT STYLE -----
hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)