import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
#import data
student_data = pd.read_csv("student_data.csv")

# Create the X (class size) and y (test scores) arrays
X = student_data[['Class Size']].values
y = student_data['Test Scores'].values

# create model
reg = LinearRegression()

# fit or training the model to the data
reg.fit(X, y)

# Print the coefficients
print(f"Intercept (B0): {reg.intercept_}")
print(f"Coefficient (B1): {reg.coef_[0]}")

# make predictions
y_pred = 10
predicted_spend = reg.predict([[y_pred]])

# Print predictions
print(f"Predicted Test Scores for Class Size {y_pred}: {predicted_spend[0]}")

# Plot the data and the regression line
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, reg.predict(X), color='red', label='Regression Line')
plt.title('Linear Regression: Class Size vs Test Scores')
plt.xlabel('Class Size')
plt.ylabel('Test Scores')
plt.grid(True)
plt.legend("best")
plt.show()


# Display Predictions
print(f"Predictions: {predicted_spend[0]} for Class Size {y_pred}")


