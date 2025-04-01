import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Linear Regression Example
""" Problem:
Sarah's Challenge: Loves trying new fruits, but wants to increase her odds of finding winners.

LDA to the Rescue: This technique helps predict "like" or "dislike" based on both size and sweetness.

How LDA Works: It analyzes Sarah's past choices, lookimg for the best way to separete the fruits she's enjoyed from the ones she hasn't.

Imagime drawing a line on the grahp to divide the "like" and "dislike" fruits as cleanly as possible.

The power of Combination: LDA shines when multiple features matters. It helps Sarah use size and sweetness together for potencially better predictions.
"""
# Sample Data
# [size, sweetness]
fruits_features = np.array([[3, 7], [2, 8], [3, 6], [4, 7], [1, 4], [2 ,3], [3, 2], [4, 3]])
fruits_likes = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # 1: like, 0: dislike

# Create LDA model
model = LDA()

# Fit the model to the data
model.fit(fruits_features, fruits_likes)

# Preditions
new_fruits = np.array([[2.5, 6]]) # [size, sweetness]
predicted_likes = model.predict(new_fruits)

# Plotting
plt.scatter(fruits_features[:, 0], fruits_features[:, 1], c=fruits_likes, cmap='viridis', marker='o', label='Fruits')
plt.scatter(new_fruits[:, 0], new_fruits[:, 1], color='darkred', marker='x', label='New Fruit')
plt.title('LDA: Fruits Based on Size and Sweetness')
plt.xlabel('Size')
plt.ylabel('Sweetness')
plt.show()

# display predictions
print(f"Sarah will {'like' if predicted_likes[0] == 1 else 'dislike'} the new fruit with size {new_fruits[0][0]} and sweetness {new_fruits[0][1]}")
