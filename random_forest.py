import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Expanded Data
plants_features = np.array([
    [3, 1], [2, 2], [4, 1], [3, 2], [5, 1], [2, 2], [4, 1], [5, 2],
    [3, 1], [4, 2], [5, 1], [3, 2], [2, 1], [4, 2], [3, 1], [4, 2],
    [5, 1], [2, 2], [3, 1], [4, 2], [2, 1], [5, 2], [3, 1], [4, 2]
])

plants_species = np.array([
    0, 1, 0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 1, 0, 1
])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(plants_features, plants_species, test_size=0.25, random_state=42)

# Creating a Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Training the model
model.fit(X_train, y_train)

# Prediction & Evaluation
y_pred = model.predict(X_test)
classification_rep = classification_report(y_test, y_pred)

# Displaying Prediction and Evaluation
print("Predictions on Test Set:")
print(classification_rep)

# Scatter Plot Visualization Classes
plt.figure(figsize=(8, 4))
for species, maker, color in zip([0, 1], ['o', 's'], ['blue', 'orange']):
    plt.scatter(plants_features[plants_species == species][:, 0],
                plants_features[plants_species == species][:, 1],
                marker=maker, color=color, label=f'Species {species}')
plt.xlabel('Leaf Size')
plt.ylabel('Flower Color (coded)')
plt.title('Plant Species Classification')
plt.legend()
plt.tight_layout()
plt.show()

# Visualization Feature Importances
plt.figure(figsize=(8, 4))
features_importances = model.feature_importances_
features = ['Leaf Size', 'Flower Color']
plt.barh(features, features_importances, color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importances in Random Forest Model')
plt.show()

