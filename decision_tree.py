import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree


# Linear Regression Example
""" Problem:
Alex's Question: Is there a link between study time and grades?

Data: Alex collected info on study hours and test scores from his classmates.

Tool: Decision Tree Regression can help find patterns in this data.

How it works: Decision Tree Regression split the data into groups based on study time, finding average scores within each groun to make predictions.
"""

# Sample Data
# [hours_studied]
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)  # Reshape for sklearn
test_scores = np.array([50, 55, 70, 80, 85, 90, 92, 98])  # Test scores

# Creating a Decision Tree Regression Model
model = DecisionTreeRegressor(max_depth=3)  # Limiting depth for simplicity

# trainig the model
model.fit(study_hours, test_scores)

# Predictions
new_study_hour = np.array([[5.5]])  # New study hours for prediction
predicted_score = model.predict(new_study_hour)

# Plotting the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=['Study Hours'], rounded=True)
plt.scatter(new_study_hour, predicted_score, color='green', s=100, label='Predicted Score')
plt.title('Decision Tree Regression: Study Hours vs Test Scores')
plt.xlabel('Study Hours')
plt.ylabel('Test Scores')
plt.grid(True)
plt.show()

# Displaying the Prediction
print(f"Based on the model, if Alex studies for {new_study_hour[0, 0]} hours, he can expect a score of approximately {predicted_score[0]:.2f}.")


