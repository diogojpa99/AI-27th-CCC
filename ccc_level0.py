import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.decomposition import PCA


###### Lets define levels


file_path = '/content/gdrive/MyDrive/CCC/training_data.csv'
file_path = 'input/data.csv'
df = pd.read_csv(file_path)

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df)

print(X_pca.shape)
# # Calculate the standard deviations of the two resulting dimensions
# stds = np.sqrt(pca.explained_variance_)

# # Round the standard deviations to two decimal places
# stds_rounded = [round(std, 2) for std in stds]
stds = np.std(X_pca, axis=0)

# Print the results
# print(stds_rounded[0], stds_rounded[1])
print("Standard deviation of column 1: {:.2f}".format(stds[0]))
print("Standard deviation of column 2: {:.2f}".format(stds[1]))

# Plot the data as a scatter plot
plt.scatter(X_pca[:,0], X_pca[:,1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Scatter Plot of Data After PCA")
plt.savefig("scatter_plot.png")  # save the plot as a PNG image file
plt.show()