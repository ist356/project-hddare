# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy.optimize import bisect
from statsmodels.formula.api import ols
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# For interpolation
from scipy.interpolate import interp1d

# For reading Excel files
import openpyxl

# Reading CSV and Excel files
base_dir = Path(__file__).parent.parent  # Adjust if your structure differs
data_dir = base_dir / 'data'

# Read CSV and Excel files using relative paths
cpi = pd.read_csv(data_dir / "CPI-421.csv")
inflation = pd.read_excel(data_dir / "Inflation-421.xlsx")
dji = pd.read_csv(data_dir / "DJI2-421.csv")
income = pd.read_csv(data_dir / "PIncomePerCap.csv")
sp500 = pd.read_csv(data_dir / "S&P 500 Historical Data.csv")
incbyage = pd.read_excel(data_dir / "incomebyage.xlsx")
col_city = pd.read_excel(data_dir / "col_city.xlsx")
