# functions.py
from playwright.sync_api import sync_playwright
import pandas as pd
import numpy as np
from scipy.optimize import bisect
from statsmodels.formula.api import ols
import warnings
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import os

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

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols

def convert_yoy_to_mom(yoy):
    """
    Convert Year-over-Year (YoY) inflation rates to Month-over-Month (MoM) rates.
    """
    return ((1 + yoy / 100) ** (1/12) - 1) * 100


def calculate_cpi_growth_rate(cumulative_cpi):
    """
    Calculate CPI growth rate using linear regression on yearly aggregated data.
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
    
    # Fit a linear model on log-transformed yearly CPI growth
    df = pd.DataFrame({'log_cpi': np.log(np.array(yearly_cpi) + 1), 'time': time})
    model = ols('log_cpi ~ time', data=df).fit()
    
    # Extract the slope and calculate growth rate
    slope = model.params['time']
    cpi_growth_rate = np.exp(slope) - 1
    return cpi_growth_rate


def calculate_inflation_growth_rate(cumulative_inflation):
    """
    Calculate Inflation growth rate using linear regression on yearly aggregated data.
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
    
    # Fit a linear model on log-transformed yearly inflation growth
    df = pd.DataFrame({'log_inflation': np.log(np.array(yearly_inflation) + 1), 'time': time})
    model = ols('log_inflation ~ time', data=df).fit()
    
    # Extract the slope and calculate growth rate
    slope = model.params['time']
    inflation_growth_rate = np.exp(slope) - 1
    return inflation_growth_rate


def calculate_annual_geometric_return(cumulative_data):
    """
    Calculate annual geometric return based on cumulative data.
    """
    num_months = len(cumulative_data)
    num_years = num_months / 12
    annual_geometric_return = (cumulative_data.iloc[-1] ** (1 / num_years) - 1) * 100
    return annual_geometric_return


def predict_income_growth(start_age, current_income, inflation_rate, incbyage_df):
    """
    Predict income growth rates based on polynomial regression and adjust for inflation.
    """
    # Ensure 'Median' column is numeric and remove invalid rows
    incbyage_df['Median'] = pd.to_numeric(incbyage_df['Median'], errors='coerce')
    incbyage_df = incbyage_df.dropna(subset=['Median'])  # Drop rows with NaN in Median

    # Filter data from the given age to the end
    filtered_data = incbyage_df[incbyage_df['Age'] >= start_age]
    if filtered_data.empty:
        raise ValueError("No valid data for the specified starting age. Check the input data.")

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






def calculate_retirement_savings(current_income, age, yearly_cost_of_living, average_geometric_return, income_growth_rates, cpi_growth_rate=0.029, current_savings=0, retirement_age=66, manual_goal=None):
    """
    Calculate retirement savings rate and yearly projections.
    """
    years_to_retire = retirement_age - age
    retirement_goal_years = 25  # Post-retirement years
    avg_market_return = average_geometric_return / 100  # Convert to decimal
    
    # Initialize projection DataFrame
    projection_data = pd.DataFrame({
        'Year': np.arange(age, retirement_age),
        'Income': np.zeros(years_to_retire),
        'After_Tax_Income': np.zeros(years_to_retire),
        'Cost_of_Living': np.zeros(years_to_retire),
        'Spare_Cash': np.zeros(years_to_retire),
        'Amount_Invested': np.zeros(years_to_retire),
        'Total_Savings': np.zeros(years_to_retire),
    })
    
    # Initialize income, cost of living, and total savings
    income = current_income
    cost_of_living = yearly_cost_of_living
    total_savings = current_savings

    # Binary search setup for savings rate
    lower_bound = 0
    upper_bound = 150
    tolerance = 1e-2
    savings_rate = (lower_bound + upper_bound) / 2
    
    # Calculate Retirement Goal
    for i in range(years_to_retire - 1):
        income_growth_rate = income_growth_rates[i] if i < len(income_growth_rates) else income_growth_rates[-1]
        income *= (1 + income_growth_rate) * (1 - cpi_growth_rate)  # Adjust for inflation
    
    final_income = income
    retirement_goal = manual_goal if manual_goal else final_income * 0.8 * retirement_goal_years

    # Binary Search for Savings Rate
    while True:
        temp_savings = current_savings
        income = current_income
        cost_of_living = yearly_cost_of_living

        for i in range(years_to_retire):
            after_tax_income = income - calculate_income_tax(income)
            amount_invested = max(0, (after_tax_income - cost_of_living) * (savings_rate / 100))
            temp_savings = (temp_savings * (1 + avg_market_return)) + amount_invested

            # Adjust income and cost of living for next year
            income_growth_rate = income_growth_rates[i] if i < len(income_growth_rates) else income_growth_rates[-1]
            income *= (1 + income_growth_rate) * (1 - cpi_growth_rate)
            cost_of_living *= (1 + cpi_growth_rate)

        if abs(temp_savings - retirement_goal) <= tolerance:
            break
        elif temp_savings > retirement_goal:
            upper_bound = savings_rate
        else:
            lower_bound = savings_rate
        savings_rate = (lower_bound + upper_bound) / 2
    
    # Final Projection Data
    income = current_income
    cost_of_living = yearly_cost_of_living
    total_savings = current_savings

    for i in range(years_to_retire):
        after_tax_income = income - calculate_income_tax(income)
        amount_invested = max(0, (after_tax_income - cost_of_living) * (savings_rate / 100))
        total_savings = (total_savings * (1 + avg_market_return)) + amount_invested

        projection_data.loc[i] = [
            age + i, income, after_tax_income, cost_of_living, after_tax_income - cost_of_living,
            amount_invested, total_savings
        ]

        # Adjust for next year
        income_growth_rate = income_growth_rates[i] if i < len(income_growth_rates) else income_growth_rates[-1]
        income *= (1 + income_growth_rate) * (1 - cpi_growth_rate)
        cost_of_living *= (1 + cpi_growth_rate)

    return {'projection_data': projection_data, 'savings_rate': round(savings_rate, 2)}

def scrape_table():
    '''
    scrape for income by age data
    '''
    # Create the data folder if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Start a Playwright session
    with sync_playwright() as p:
        # Launch a browser
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the target page
        url = 'https://dqydj.com/income-percentile-by-age-calculator/'
        page.goto(url)

        # Wait for the table to load
        page.wait_for_selector("table")

        # Select the table element
        table = page.query_selector("table")

        # Extract table rows
        rows = table.query_selector_all("tbody tr")
        data = []
        for row in rows:
            cells = row.query_selector_all("td")
            data.append([cell.inner_text().strip() for cell in cells])

        # Use the first row as headers and the rest as data
        headers = data[0]  # The first row becomes the headers
        table_data = data[1:]  # Remaining rows are the data

        # Create a DataFrame
        df = pd.DataFrame(table_data, columns=headers)

        # Convert all columns to numeric where possible
        for column in df.columns:
            df[column] = pd.to_numeric(df[column].str.replace(r'[^0-9.\-]', '', regex=True), errors='coerce')

        # Save the DataFrame to the data folder
        file_path = os.path.join("data", "incomebyage.csv")
        df.to_csv(file_path, index=False)

        # Print the DataFrame
        print(f"Data saved to {file_path}")
        print(df)

        # Close the browser
        browser.close()
