# functions.py

import pandas as pd
import numpy as np
from scipy.optimize import bisect
from statsmodels.formula.api import ols
import warnings

# Suppress warnings within functions
warnings.filterwarnings('ignore')

# Define tax brackets and rates for single filers
tax_brackets = pd.DataFrame({
    'bracket': [10275, 41775, 89075, 170050, 215950, 539900, np.inf],
    'rate': [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]
})

def calculate_income_tax(income):
    """
    Calculate the federal income tax for a single filer based on IRS tax brackets.
    """
    tax = 0
    previous_bracket = 0
    
    for _, row in tax_brackets.iterrows():
        if income > row['bracket']:
            tax += (row['bracket'] - previous_bracket) * row['rate']
            previous_bracket = row['bracket']
        else:
            tax += (income - previous_bracket) * row['rate']
            break
    return tax

def convert_yoy_to_mom(yoy):
    """
    Convert Year-over-Year (YoY) inflation rates to Month-over-Month (MoM) rates.
    """
    return ((1 + yoy / 100) ** (1/12) - 1) * 100

def calculate_cpi_growth_rate(cumulative_cpi):
    """
    Calculate CPI growth rate using linear regression on log-transformed data.
    """
    # Aggregate CPI data to yearly growth
    num_years = len(cumulative_cpi) // 12
    yearly_cpi = []
    for i in range(num_years):
        start_index = i * 12
        end_index = (i + 1) * 12
        annual_growth = (np.prod(cumulative_cpi[start_index:end_index] + 1) ** (1/12)) - 1
        yearly_cpi.append(annual_growth)
    
    # Create a time variable for regression
    time = np.arange(1, len(yearly_cpi) + 1)
    
    # Fit a linear model using log
    df = pd.DataFrame({'log_cpi': np.log(np.array(yearly_cpi) + 1), 'time': time})
    model = ols('log_cpi ~ time', data=df).fit()
    
    # Extract the slope (growth rate)
    slope = model.params['time']
    cpi_growth_rate = np.exp(slope) - 1
    return cpi_growth_rate

def calculate_inflation_growth_rate(cumulative_inflation):
    """
    Calculate Inflation growth rate using linear regression on log-transformed data.
    """
    # Aggregate inflation data to yearly growth
    num_years = len(cumulative_inflation) // 12
    yearly_inflation = []
    for i in range(num_years):
        start_index = i * 12
        end_index = (i + 1) * 12
        annual_growth = (np.prod(cumulative_inflation[start_index:end_index] + 1) ** (1/12)) - 1
        yearly_inflation.append(annual_growth)
    
    # Create a time variable for regression
    time = np.arange(1, len(yearly_inflation) + 1)
    
    # Fit a linear model using log
    df = pd.DataFrame({'log_inflation': np.log(np.array(yearly_inflation) + 1), 'time': time})
    model = ols('log_inflation ~ time', data=df).fit()
    
    # Extract the slope (growth rate)
    slope = model.params['time']
    inflation_growth_rate = np.exp(slope) - 1
    return inflation_growth_rate

def calculate_annual_geometric_return(cumulative_data):
    """
    Calculate annual geometric return based on cumulative data.
    """
    num_months = len(cumulative_data)
    num_years = num_months / 12
    annual_geometric_return = (cumulative_data[-1] ** (1 / num_years) - 1) * 100
    return annual_geometric_return

def predict_income_growth(start_age, current_income, inflation_rate, incbyage_df):
    """
    Predict income growth rates based on polynomial regression and adjust for inflation.
    """
    # Filter data from the given age to the end
    filtered_data = incbyage_df[incbyage_df['Age'] >= start_age]
    
    # Fit a polynomial regression model (degree 3)
    coeffs = np.polyfit(filtered_data['Age'], filtered_data['Median'], 3)
    model = np.poly1d(coeffs)
    
    # Predict incomes from start_age to max age
    max_age = incbyage_df['Age'].max()
    predicted_ages = np.arange(start_age, max_age + 1)
    predicted_incomes = model(predicted_ages)
    
    # Adjust predicted income based on current income
    adjustment_factor = current_income / predicted_incomes[0]
    adjusted_incomes = predicted_incomes * adjustment_factor
    
    # Calculate growth rates year over year, adjusting for inflation
    growth_rates = ((adjusted_incomes[1:] / adjusted_incomes[:-1]) - 1) + inflation_rate
    return growth_rates

def calculate_retirement_savings(current_income, age, yearly_cost_of_living, average_geometric_return, income_growth_rates, cpi_growth_rate):
    """
    Calculate the required savings rate for retirement using binary search.
    """
    # Parameters
    retirement_age = 65 + 1  # Assuming retirement at 66
    years_to_retire = retirement_age - age
    retirement_goal_years = 25  # 25 years after retirement
    avg_market_return = average_geometric_return / 100  # Convert to decimal
    
    # Initialize projection data
    projection_data = pd.DataFrame({
        'Year': np.arange(age, retirement_age),
        'Income': np.nan,
        'After_Tax_Income': np.nan,
        'Cost_of_Living': np.nan,
        'Spare_Cash': np.nan,
        'Amount_Invested': np.nan,
        'Savings_Rate': np.nan,
        'Total_Savings': np.nan
    })
    
    # Initialize variables
    income = current_income
    cost_of_living = yearly_cost_of_living
    after_tax_income = income - calculate_income_tax(income)
    
    # Early exit if after-tax income is less than or equal to cost of living
    if after_tax_income <= cost_of_living:
        print(f"Warning: Income level of {current_income} is not sufficient to cover cost of living. Retirement savings are infeasible.")
        return {'projection_data': None, 'savings_rate': None}
    
    # Calculate retirement goal based on ending income
    for i in range(years_to_retire - 1):
        if i < len(income_growth_rates):
            income_growth_rate = income_growth_rates[i] / 100
        else:
            income_growth_rate = income_growth_rates[-1] / 100
        income = income * ((1 + income_growth_rate) * (1 - cpi_growth_rate))
    
    final_income = income
    retirement_goal = final_income * 0.8 * retirement_goal_years
    
    # Binary search to find the correct savings rate
    lower_bound = 0
    upper_bound = 150  # Allow for >100% if needed
    tolerance = 1e-2
    
    def calculate_total_savings(rate):
        total_savings = 0
        income = current_income
        cost_of_living = yearly_cost_of_living
        after_tax_income = income - calculate_income_tax(income)
        for i in range(years_to_retire):
            amount_invested = max(0, (after_tax_income - cost_of_living) * (rate / 100))
            total_savings = (total_savings * (1 + avg_market_return)) + amount_invested
            # Grow income and cost of living
            if i < len(income_growth_rates):
                income_growth_rate = income_growth_rates[i] / 100
            else:
                income_growth_rate = income_growth_rates[-1] / 100
            income = income * ((1 + income_growth_rate) * (1 - cpi_growth_rate))
            after_tax_income = income - calculate_income_tax(income)
            cost_of_living = cost_of_living * (1 + cpi_growth_rate)
        return total_savings
    
    def func(rate):
        return calculate_total_savings(rate) - retirement_goal
    
    try:
        savings_rate = bisect(func, lower_bound, upper_bound, xtol=tolerance)
    except ValueError:
        print("Unable to find a suitable savings rate within bounds.")
        return {'projection_data': None, 'savings_rate': None}
    
    # Now, generate the projection data with the found savings rate
    total_savings = 0
    income = current_income
    cost_of_living = yearly_cost_of_living
    after_tax_income = income - calculate_income_tax(income)
    
    for i in range(years_to_retire):
        projection_data.at[i, 'Income'] = income
        projection_data.at[i, 'After_Tax_Income'] = after_tax_income
        projection_data.at[i, 'Cost_of_Living'] = cost_of_living
        projection_data.at[i, 'Savings_Rate'] = savings_rate
        spare_cash = after_tax_income - cost_of_living
        projection_data.at[i, 'Spare_Cash'] = spare_cash
        amount_invested = max(0, spare_cash * (savings_rate / 100))
        projection_data.at[i, 'Amount_Invested'] = amount_invested
        total_savings = (total_savings * (1 + avg_market_return)) + amount_invested
        projection_data.at[i, 'Total_Savings'] = total_savings
        # Grow income and cost of living for the next year
        if i < len(income_growth_rates):
            income_growth_rate = income_growth_rates[i] / 100
        else:
            income_growth_rate = income_growth_rates[-1] / 100
        income = income * ((1 + income_growth_rate) * (1 - cpi_growth_rate))
        after_tax_income = income - calculate_income_tax(income)
        cost_of_living = cost_of_living * (1 + cpi_growth_rate)
    
    return {'projection_data': projection_data, 'savings_rate': round(savings_rate, 2)}
