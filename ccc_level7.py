import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

# open and read the file
#filename = 'level6_pred.txt'
# filename = 'level_7preds.txt'
# filename = 'level_8_pred.txt'
filename = 'level10_pred.txt'


array = []

with open(filename, 'r') as f:
    for line in f:
        value = float(line.strip())
        array.append(value)

#print(array)
array = np.array(array)

matrix= np.reshape(array,(50, 50))

#print(matrix)

from sklearn.decomposition import PCA

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(matrix)

# Plot the data as a scatter plot
plt.scatter(X_pca[:,0], X_pca[:,1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Scatter Plot of Data After PCA")
plt.savefig("scatter_plot.png")  # save the plot as a PNG image file
plt.show()

# Generate a random 50x50 matrix
#matrix = np.random.rand(50, 50)

# Create a heatmap plot of the matrix
plt.imshow(matrix, cmap='hot', interpolation='nearest')

# Add a colorbar to show the values corresponding to the colors
plt.colorbar()

# Show the plot
plt.show()