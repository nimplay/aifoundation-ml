import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Seed for reproducibility
np.random.seed(42)

# Generate synthetic data
num_samples = 200
num_rooms = np.random.randint(3, 10, num_samples)  # Number of rooms
house_age = np.random.randint(1, 100, num_samples) # Age of the house in years
noise = np.random.normal(0, 50, num_samples)  # Random noise

# Assume a linear relation with price = 50 * rooms + 0.5 * age + noise
price = 50 * num_rooms + 0.5 * house_age + noise

# Create DataFrame
data = pd.DataFrame({'num_rooms': num_rooms, 'house_age': house_age, 'price': price})

# Plot
plt.scatter(data['num_rooms'], data['price'], color='blue', label='Price vs. Number of Rooms')
plt.scatter(data['house_age'], data['price'], color='orange', label='Price vs. House Age')
plt.xlabel('Features')
plt.ylabel('Price')
plt.title('House Price Data')
plt.legend()
plt.show()

# Splitting data into training and testing sets
X = data[['num_rooms', 'house_age']]
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train GBR Regressor model
model_gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=42)
model_gbm.fit(X_train, y_train)

# Predictions
predictions = model_gbm.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse:.2f}')
print(f'Root Mean Square Error: {rmse}')

# Visualization: Actual vs Predicted Prices
plt.scatter(y_test, predictions, color='green', label='Predicted Prices')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3, label='Perfect Prediction')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.show()


