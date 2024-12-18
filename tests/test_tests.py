import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "code"))

from code.functions import (
    calculate_income_tax,
    convert_yoy_to_mom,
    calculate_cpi_growth_rate,
    calculate_inflation_growth_rate,
    calculate_annual_geometric_return,
    predict_income_growth,
    calculate_retirement_savings
)

def test_calculate_income_tax():
    # Test income tax calculation
    assert calculate_income_tax(5000) == 500
    assert calculate_income_tax(10275) == pytest.approx(1027.5, 0.1)
    assert calculate_income_tax(50000) == pytest.approx(6797.5, 0.1)

def test_convert_yoy_to_mom():
    # Test YoY to MoM conversion
    yoy = 12.0
    expected_mom = ((1 + yoy / 100) ** (1 / 12) - 1) * 100
    assert convert_yoy_to_mom(yoy) == pytest.approx(expected_mom, 0.00001)

def test_calculate_cpi_growth_rate():
    # Test CPI growth rate calculation
    cumulative_cpi = np.array([0.01, 0.02, 0.03, 0.04, 0.05] * 12)
    cpi_growth = calculate_cpi_growth_rate(cumulative_cpi)
    assert 0 < cpi_growth < 0.1  # Example: Ensure it's within a reasonable range

def test_calculate_inflation_growth_rate():
    # Test inflation growth rate calculation
    cumulative_inflation = np.array([0.01, 0.02, 0.03, 0.04, 0.05] * 12)
    inflation_growth = calculate_inflation_growth_rate(cumulative_inflation)
    assert 0 < inflation_growth < 0.1

def test_calculate_annual_geometric_return():
    # Test annual geometric return
    cumulative_data = pd.Series([1.01, 1.02, 1.03, 1.05, 1.08])
    annual_return = calculate_annual_geometric_return(cumulative_data)
    assert annual_return > 0

