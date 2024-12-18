# main.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# Import custom functions
from functions import (
    calculate_income_tax,
    convert_yoy_to_mom,
    calculate_cpi_growth_rate,
    calculate_inflation_growth_rate,
    calculate_annual_geometric_return,
    predict_income_growth,
    calculate_retirement_savings
)

# Set base directories
base_dir = Path(__file__).parent.parent
data_dir = base_dir / 'data'

# Title and Introduction
st.title("Retirement Savings and Income Growth Calculator")
st.markdown("### Analyze your income growth, tax implications, and savings over time.")

# Sidebar Inputs
st.sidebar.header("Input Parameters")
current_income = st.sidebar.number_input("Current Income ($)", value=50000, step=1000)
current_savings = st.sidebar.number_input("Current Savings ($)", value=0, step=1000)
starting_age = st.sidebar.slider("Your Age", 18, 65, 22)
yearly_cost_of_living = st.sidebar.number_input("Yearly Cost of Living ($)", value=30000, step=1000)
market_return = st.sidebar.slider("Average Annual Market Return (%)", 0.0, 10.0, 5.0)
inflation_rate = st.sidebar.slider("Inflation Rate (%)", 0.0, 10.0, 2.0)

# Load Income by Age Data
st.sidebar.header("Income Growth Calculation")
incbyage = pd.read_excel(data_dir / "incomebyage.xlsx")

# Predict Income Growth Rates
income_growth_rates = predict_income_growth(
    start_age=starting_age,
    current_income=current_income,
    inflation_rate=inflation_rate / 100,
    incbyage_df=incbyage
)

# Calculate Retirement Savings
result = calculate_retirement_savings(
    current_income=current_income,
    age=starting_age,
    yearly_cost_of_living=yearly_cost_of_living,
    average_geometric_return=market_return,
    income_growth_rates=income_growth_rates,
    cpi_growth_rate=inflation_rate / 100,
    current_savings=current_savings
)

projection_data = result['projection_data']
savings_rate = result['savings_rate']

# Display Results
st.header("Results")
if projection_data is not None:
    st.write(f"### Required Savings Rate: **{savings_rate:.2f}%**")
    st.dataframe(projection_data)

    # Visualize Projection Data
    st.subheader("Projected Income, Costs, and Savings Over Time")
    projection_long = projection_data.melt(
        id_vars=['Year'], 
        value_vars=['Income', 'After_Tax_Income', 'Spare_Cash', 'Amount_Invested', 'Cost_of_Living'], 
        var_name='Variable', 
        value_name='Value'
    )

    # Plot with Plotly
    fig = px.area(
        projection_long, 
        x="Year", 
        y="Value", 
        color="Variable",
        title="Income, After-Tax Income, Cost of Living, Spare Cash, and Amount Invested",
        labels={"Value": "Amount ($)"}
    )
    st.plotly_chart(fig)
else:
    st.warning("Income level is insufficient to cover cost of living. Adjust your inputs and try again.")

# Footer
st.sidebar.markdown("Developed by **Your Name**")
