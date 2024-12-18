import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import os
from functions import (
    calculate_income_tax,
    calculate_cpi_growth_rate,
    calculate_inflation_growth_rate,
    calculate_annual_geometric_return,
    calculate_retirement_savings,
    convert_yoy_to_mom,
    predict_income_growth,
    scrape_table
)
if not os.path.exists("incomebyage.csv"):
    scrape_table()

# Define base directory
base_dir = Path(__file__).parent.parent
data_dir = base_dir / 'data'

# Read Data Files
cpi = pd.read_csv(data_dir / "CPI-421.csv")
inflation = pd.read_excel(data_dir / "Inflation-421.xlsx")
sp500 = pd.read_csv(data_dir / "S&P 500 Historical Data.csv")
income = pd.read_csv(data_dir / "PIncomePerCap.csv")
incbyage = pd.read_csv(data_dir / "incomebyage.csv")


# Calculate cumulative CPI growth
cpi['CPI_MoM'] = convert_yoy_to_mom(cpi['MEDCPIM158SFRBCLE'])
cpi['CPI_Cumulative'] = (cpi['CPI_MoM'] / 100 + 1).cumprod() - 1

# Calculate S&P 500 Geometric Return
sp500['SP500_MoM'] = sp500['Change %'].str.replace('%', '').astype(float) / 100
sp500['Cumulative_SP500'] = (sp500['SP500_MoM'] + 1).cumprod()
sp500_geometric_return = calculate_annual_geometric_return(sp500['Cumulative_SP500'])

# Update default slider values dynamically based on calculations
calculated_cpi_growth = calculate_cpi_growth_rate(cpi['CPI_Cumulative'])
calculated_inflation_growth = calculate_inflation_growth_rate(cpi['CPI_Cumulative'])
calculated_avg_return = sp500_geometric_return / 100

# Streamlit Sidebar for User Inputs
st.sidebar.header("User Inputs")
retirement_age = st.sidebar.slider("Retirement Age", min_value=50, max_value=80, value=66)
starting_age = st.sidebar.slider("Starting Age", min_value=18, max_value=retirement_age-1, value=22)
current_income = st.sidebar.number_input("Current Income ($)", min_value=1000, value=68516, step=500)
yearly_cost_of_living = st.sidebar.number_input("Yearly Cost of Living ($)", min_value=1000.0, value=32603.54,max_value=80000.0, step=500.0)
current_savings = st.sidebar.number_input("Current Savings ($)", min_value=0, value=0, step=500)
manual_goal = st.sidebar.number_input("Manual Savings Goal ($)", min_value=0, value=0, step=1000)

# CPI Growth Rate Slider
st.sidebar.subheader("Default Calculated Values")
inflation_growth_rate = st.sidebar.slider("Inflation Growth Rate (%)", min_value=0.0, max_value=10.0, value=calculated_inflation_growth*100, step=0.1) / 100
average_geometric_return = st.sidebar.slider("Average Market Return (%)", min_value=0.0, max_value=15.0, value=sp500_geometric_return, step=0.1)

# Data Cleaning and Processing

# Convert YoY inflation to MoM
for col in inflation.columns[1:]:
    # Ensure all values are numeric and handle non-numeric data by coercing them to NaN
    inflation[col] = pd.to_numeric(inflation[col], errors='coerce')
    inflation[col] = convert_yoy_to_mom(inflation[col])

income_growth_rates = predict_income_growth(starting_age, current_income, inflation_growth_rate, incbyage)

# Display calculated defaults
st.sidebar.write(f"Default Inflation Growth Rate: {calculated_inflation_growth:.5%}")
st.sidebar.write(f"Default Average Market Return: {sp500_geometric_return:.2f}%")

# Retirement Savings Projection
st.title("Retirement Savings Projection")

# Calculate Retirement Savings
result = calculate_retirement_savings(
    current_income=current_income,
    age=starting_age,  # Starting age
    yearly_cost_of_living=yearly_cost_of_living,
    average_geometric_return=average_geometric_return,
    income_growth_rates=income_growth_rates,
    cpi_growth_rate=inflation_growth_rate,
    current_savings=current_savings,
    manual_goal=manual_goal,
    retirement_age=retirement_age
)

projection_data = result['projection_data']
savings_rate = result['savings_rate']

# Display Savings Rate
st.subheader(f"Required Savings Rate: {savings_rate}%")

# Plot Projection Data
if projection_data is not None:
    import matplotlib.pyplot as plt

    projection_long = projection_data.melt(
        id_vars=['Year'],
        value_vars=['Income', 'After_Tax_Income', 'Spare_Cash', 'Amount_Invested', 'Cost_of_Living'],
        var_name='Variable',
        value_name='Value'
    )

    # Define Colors
    color_palette = {
        'Income': '#6a3d9a',           # purple
        'After_Tax_Income': '#1f78b4', # blue
        'Cost_of_Living': '#33a02c',   # green
        'Spare_Cash': '#e31a1c',       # red
        'Amount_Invested': '#ffeb3b'   # yellow
    }

    # Plot Data
    plt.figure(figsize=(12, 6))
    for variable, color in color_palette.items():
        subset = projection_long[projection_long['Variable'] == variable]
        plt.plot(subset['Year'], subset['Value'], label=variable, color=color)
        plt.fill_between(subset['Year'], 0, subset['Value'], alpha=0.2, color=color)

    plt.title('Income, After-Tax Income, Spare Cash, and Amount Invested Over Time')
    plt.xlabel('Year')
    plt.ylabel('Value ($)')
    plt.legend(title='Variable')
    plt.grid(True)
    st.pyplot(plt)

# Show Table Data
if projection_data is not None:
    st.write("### Detailed Projection Data")
    st.dataframe(projection_data)
