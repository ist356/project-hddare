# Reflection

Student Name:  Henry Dare
Student Email:  hddare@syr.edu

## Instructions

Reflection is a key activity of learning. It helps you build a strong metacognition, or "understanding of your own learning." A good learner not only "knows what they know", but they "know what they don't know", too. Learning to reflect takes practice, but if your goal is to become a self-directed learner where you can teach yourself things, reflection is imperative.

- Now that you've completed the assignment, share your throughts. What did you learn? What confuses you? Where did you struggle? Where might you need more practice?
- A good reflection is: **specific as possible**,  **uses the terminology of the problem domain** (what was learned in class / through readings), and **is actionable** (you can pursue next steps, or be aided in the pursuit). That last part is what will make you a self-directed learner.
- Flex your recall muscles. You might have to review class notes / assigned readings to write your reflection and get the terminology correct.
- Your reflection is for **you**. Yes I make you write them and I read them, but you are merely practicing to become a better self-directed learner. If you read your reflection 1 week later, does what you wrote advance your learning?

Examples:

- **Poor Reflection:**  "I don't understand loops."   
**Better Reflection:** "I don't undersand how the while loop exits."   
**Best Reflection:** "I struggle writing the proper exit conditions on a while loop." It's actionable: You can practice this, google it, ask Chat GPT to explain it, etc. 
-  **Poor Reflection** "I learned loops."   
**Better Reflection** "I learned how to write while loops and their difference from for loops."   
**Best Reflection** "I learned when to use while vs for loops. While loops are for sentiel-controlled values (waiting for a condition to occur), vs for loops are for iterating over collections of fixed values."

`--- Reflection Below This Line ---`

In this project, I developed a deeper understanding of integrating data analysis, user inputs, and visualization into a functional Streamlit app. One of the significant challenges I encountered was cleaning and preparing data for analysis, especially when scraping or handling datasets with inconsistent formatting. For example, I had to address non-numeric values and ensure columns were properly formatted before running calculations. This reinforced the importance of thorough data validation and error handling, especially working with scraped or downloaded datasets instead of what is provided in class.

Another major challenge was debugging the calculate_retirement_savings function. Initially, the function produced incorrect results due to issues with how income, cost of living, and savings compounded over time. Correctly incorporating inflation adjustments and market returns required careful step-by-step testing to ensure accuracy. This process highlighted how small errors in logic can cascade into major discrepancies in outputs, especially in large projections.

Additionally, setting up and running tests was a learning experience. I faced issues with module imports due to the project’s folder structure, which prevented the tests from running initially. Through research and adjustments, I learned how to structure a project properly for testing, including resolving relative import issues. Writing meaningful tests allowed me to verify critical functionality and gave me confidence that the app was working as intended.

Key takeaways include:

- The importance of data cleaning for reliable analysis.
- Debugging and validating complex projections step by step.
- Structuring a project for effective unit testing and troubleshooting import errors.

////
There are a few packages that are giving me the error
ERROR: Could not find a version that satisfies the requirement os (from versions: none)
ERROR: No matching distribution found for os
packages include: os, warnings, and sklearn. I have no idea why this is happening, and the program is running fine currently.