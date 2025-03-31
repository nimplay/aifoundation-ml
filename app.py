import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Linear Regression Example
""" Problem: We have collected data on class size and test scores
for a sample of students and we want to model the relationship between
class size and test scores using linear regression.

       Test Scores = B0 + B1 * Class Size + E

1.Test Scores is the dependent variable
2.Class size is the independent variable
3.B1 is the coefficient for Class Size
4.B0 is the y-intercept
5.E is the error term
"""

student_data = pd.read_csv("student_data.csv")

# Create the X (class size) and y (test scores) arrays
X = student_data[['Class Size']].values
y = student_data['Test Scores'].values

# fit the model to the data
reg = LinearRegression().fit(X, y)

# Print the coefficients
print(f"Intercept (B0): {reg.intercept_}")
print(f"Coefficient (B1): {reg.coef_[0]}")
