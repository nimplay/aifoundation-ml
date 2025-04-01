import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Sample Data
client_data = np.array([[2000, 60], [2500, 45], [1800, 75], [2200, 50], [2100, 62], [2300, 70], [1900, 55], [2000, 65]])
weight_loss = np.array([3, 2, 4, 3, 3.5, 4.5, 3.7, 4.2])  # Weight loss in kg

# Train-test Split
x_train, x_test, y_train, y_test = train_test_split(client_data, weight_loss, test_size=0.25, random_state=42)

# Creating a Bagging Model
base_estimator = DecisionTreeRegressor(max_depth=4)  # Base estimator
model = BaggingRegressor(estimator=base_estimator, n_estimators=10, random_state=42)

# Training the model
model.fit(x_train, y_train)

# Prediction & Evaluation
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)

# Displaying Predictions and Evaluation
print(f"True Weight Loss: {y_test}")
print(f"Predicted Weight Loss: {y_pred}")
print(f"Mean Squared Error: {mse:.2f}")

# Visualization of the Base Estimator (if desired)
plt.figure(figsize=(12, 8))
tree = model.estimators_[0]  # Get the first base estimator
plt.title('Decision Tree from Bagging')
plot_tree(tree, filled=True, rounded=True, feature_names=['Calorie Intake', "Workout Duration"])
plt.show()
