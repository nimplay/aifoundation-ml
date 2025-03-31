import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Logistic Regression Example
"""
Problem: We have collected data on class size and test scores
for a sample of students and we want to model the probability
of students passing (scoring ≥ 80) based on class size using
logistic regression.

       log(P/(1-P)) = B0 + B1 * Class Size + E

Where:
1. P is the probability of passing (Test Score ≥ 80)
2. Class Size is the independent variable
3. B1 is the coefficient for Class Size
4. B0 is the y-intercept
5. E is the error term
6. The left side is the log-odds (logit) of passing

Key differences from linear regression:
- Output is probability (0 to 1) rather than continuous value
- Uses sigmoid function to model probabilities
- Interpret coefficients in terms of odds ratios
"""

# Import and prepare data
student_data = pd.read_csv("student_data.csv")
# Convert test scores to binary outcome (1 if ≥ 80, else 0)
student_data['Pass'] = (student_data['Test Scores'] >= 80).astype(int)

# Create feature matrix (X) and target vector (y)
X = student_data[['Class Size']].values  # Independent variable
y = student_data['Pass'].values          # Binary dependent variable

# Create and train logistic regression model
reg = LogisticRegression()
reg.fit(X, y)  # Learn the relationship between X and y

# Print model coefficients
print(f"Intercept (B0): {reg.intercept_[0]:.4f}")  # Log-odds when Class Size=0
print(f"Coefficient (B1): {reg.coef_[0][0]:.4f}")  # Change in log-odds per unit increase in Class Size

# Predict probabilities of passing
y_prob = reg.predict_proba(X)[:, 1]  # Probability of class=1 (passing)
print("\nPredicted probabilities:")
for size, prob in zip(X, y_prob):
    print(f"Class Size {size[0]}: {prob:.2%} chance of passing")

# Visualization
plt.figure(figsize=(10, 6))
# Plot actual binary outcomes
plt.scatter(X, y, color='blue', label='Actual Outcomes (0=Fail, 1=Pass)')

# Create smooth curve of predicted probabilities
X_sorted = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_prob_sorted = reg.predict_proba(X_sorted)[:, 1]
plt.plot(X_sorted, y_prob_sorted, color='red',
         label='Predicted Probability of Passing')

plt.title('Logistic Regression: Class Size vs Passing Probability')
plt.xlabel('Class Size')
plt.ylabel('Probability of Passing (Score ≥ 80)')
plt.yticks([0, 0.25, 0.5, 0.75, 1], ['0%', '25%', '50%', '75%', '100%'])
plt.grid(True)
plt.legend()
plt.show()
