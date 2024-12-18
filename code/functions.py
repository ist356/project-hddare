# functions.py

import pandas as pd
import numpy as np
from scipy.optimize import bisect
from statsmodels.formula.api import ols
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

# Define tax brackets
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
    # convert CPI data to yearly growth from monthly data
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
    # convert inflation data to yearly growth
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


def calculate_retirement_savings(current_income, age, yearly_cost_of_living, average_geometric_return, income_growth_rates, cpi_growth_rate, current_savings=0):
    """
    Calculates the savings rate needed to achieve retirement goals.
    """
    # Parameters
    retirement_age = 65 + 1
    years_to_retire = retirement_age - age
    retirement_goal_years = 25  # Assume 25 years after retirement
    avg_market_return = average_geometric_return / 100  # Convert to decimal

    # Initialize data frame to store yearly projections
    projection_data = pd.DataFrame({
        'Year': np.arange(age, retirement_age),
        'Income': np.zeros(years_to_retire),
        'After_Tax_Income': np.zeros(years_to_retire),
        'Cost_of_Living': np.zeros(years_to_retire),
        'Spare_Cash': np.zeros(years_to_retire),
        'Amount_Invested': np.zeros(years_to_retire),
        'Savings_Rate': np.zeros(years_to_retire),
        'Total_Savings': np.zeros(years_to_retire)
    })

    # Initialize variables to track income, cost of living, and savings rate
    income = current_income
    cost_of_living = yearly_cost_of_living
    total_savings = current_savings  # Start with the current savings
    after_tax_income = income - calculate_income_tax(income)

    ...


    # Early exit if after-tax income is less than or equal to cost of living
    if after_tax_income <= cost_of_living:
        print(f"Warning: Income level of {current_income} is not sufficient to cover cost of living. Retirement savings are infeasible.")
        return {'projection_data': None, 'savings_rate': None}

    years_to_retire2 = years_to_retire - 1

    # Calculate retirement goal based on ending income and retirement goal years
    for i in range(years_to_retire2):
        income_growth_rate = income_growth_rates[i] if i < len(income_growth_rates) else income_growth_rates[-1]
        income *= (1 + income_growth_rate) * (1 - cpi_growth_rate)

    final_income = income
    retirement_goal = final_income * 0.8 * retirement_goal_years

    # Use a binary search approach to find the correct savings rate
    lower_bound = 0
    upper_bound = 150  # Increase upper bound to allow for cases where >100% is required
    tolerance = 1e-2
    savings_rate = (lower_bound + upper_bound) / 2

    iteration_count = 0
    max_iterations = 10000

    while True:
        iteration_count += 1

        # Early break if iteration count exceeds a reasonable limit
        if iteration_count > max_iterations:
            print(f"Warning: Exceeded maximum iterations for income level: {current_income}")
            return {'projection_data': None, 'savings_rate': None}

        # Initialize total savings to zero
        total_savings = 0
        income = current_income
        cost_of_living = yearly_cost_of_living
        after_tax_income = income - calculate_income_tax(income)

        # Loop through each year to project income, after-tax income, cost of living, and calculate total savings
        for i in range(years_to_retire):
            # Calculate amount invested and update total savings
            amount_invested = max(0, (after_tax_income - cost_of_living) * (savings_rate / 100))
            total_savings = (total_savings * (1 + avg_market_return)) + amount_invested

            # Grow income and cost of living for the next year
            income_growth_rate = income_growth_rates[i] if i < len(income_growth_rates) else income_growth_rates[-1]
            income *= (1 + income_growth_rate) * (1 - cpi_growth_rate)
            after_tax_income = income - calculate_income_tax(income)
            cost_of_living *= (1 + cpi_growth_rate)

        # Check if the total savings is close enough to the retirement goal
        if abs(total_savings - retirement_goal) <= tolerance:
            break
        elif total_savings > retirement_goal:
            upper_bound = savings_rate
        else:
            lower_bound = savings_rate

        savings_rate = (lower_bound + upper_bound) / 2

    # Loop through each year to record projection data
    income = current_income
    cost_of_living = yearly_cost_of_living
    after_tax_income = income - calculate_income_tax(income)
    total_savings = 0

    for i in range(years_to_retire):
        # Record income, after-tax income, cost of living, and savings rate
        projection_data.loc[i, 'Income'] = income
        projection_data.loc[i, 'After_Tax_Income'] = after_tax_income
        projection_data.loc[i, 'Cost_of_Living'] = cost_of_living
        projection_data.loc[i, 'Savings_Rate'] = savings_rate
        projection_data.loc[i, 'Spare_Cash'] = after_tax_income - cost_of_living

        # Calculate total savings for the year
        amount_invested = max(0, (after_tax_income - cost_of_living) * (savings_rate / 100))
        total_savings = (total_savings * (1 + avg_market_return)) + amount_invested
        projection_data.loc[i, 'Total_Savings'] = total_savings
        projection_data.loc[i, 'Amount_Invested'] = amount_invested

        # Grow income and cost of living for the next year
        income_growth_rate = income_growth_rates[i] if i < len(income_growth_rates) else income_growth_rates[-1]
        income *= (1 + income_growth_rate) * (1 - cpi_growth_rate)
        after_tax_income = income - calculate_income_tax(income)
        cost_of_living *= (1 + cpi_growth_rate)

    # Return the projection data and savings rate
    return {'projection_data': projection_data, 'savings_rate': round(savings_rate, 2)}

