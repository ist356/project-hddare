# About My Project

Student Name:  name
Student Email:  email

### What it does
What it does
This project calculates and visualizes retirement savings projections based on user inputs like income, cost of living, savings, and retirement age.

Features:

Predicts income growth adjusted for inflation.
Determines the savings rate needed to meet a retirement goal.
Displays interactive graphs showing income, savings, expenses, and investments over time.
Uses default values for market returns and inflation based on real historical data.

### How you run my project
Install required libraries:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run code/main.py

Use the sidebar to input:

Current income, cost of living, savings, and retirement age.
Adjust inflation and market return rates as needed.
View the results:

A savings rate will be displayed.
The graph will show income, savings, and expenses over time.

### Other things you need to know

Files:
code/functions.py: Core calculations.
code/main.py: Streamlit application.
tests/tests.py: Unit tests.

Testing:
Run tests with pytest tests/
Data:

Data files are stored in the data folder.