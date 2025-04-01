import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

# sample data
# [movie_length, genre_code] (assuming genre is coded as 0 for Action, 1 for Romance, etc.)
movies_features = np.array([[120, 0], [150, 1], [90, 0], [140, 1], [100, 0], [80, 1], [110, 0], [130, 1]])
movies_likes = np.array([1, 1, 0, 1, 0, 1, 0, 1])  # 1: like, 0: dislike

# Creating a Naive Bayes Model
model = GaussianNB()

# Training the model
model.fit(movies_features, movies_likes)

# Predictions
new_movies = np.array([[100, 1]])  # [movie_length, genre_code]
predicted_likes = model.predict(new_movies)

# Plotting
plt.scatter(movies_features[:, 0], movies_features[:, 1], c=movies_likes, cmap='viridis', marker='o', label='Movies')
plt.scatter(new_movies[:, 0], new_movies[:, 1], color='darkred', marker='x', label='New Movie')
plt.title('Naive Bayes: Movies Based on Length and Genre')
plt.xlabel('Movie Length (min)')
plt.ylabel('Genre Code')
plt.show()

# Display Predictions
print(f"Sarah will {'like' if predicted_likes[0] == 1 else 'dislike'} the new movie with length {new_movies[0][0]} and genre code {new_movies[0][1]}")
