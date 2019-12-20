import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.cluster import KMeans

# load the dataset
iris_df = datasets.load_iris()
pca = PCA(2)

# print y-values names
print(iris_df.target_names)

# split the dataset
X, y = iris_df.data, iris_df.target
X_proj = pca.fit_transform(X)

# plot the dataset
plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y)
plt.show()

# set the size of the plot
plt.figure(figsize=(10, 4))

# create color map
colormap = np.array(['red', 'lime', 'black', 'blue', 'yellow', 'green', 'red'])

k = 3  # running kmeans clustering into two
kmeans = KMeans(n_clusters=k, random_state=0).fit(X_proj)  # train the classifier

# set the classifiers y-values to labels
labels = kmeans.labels_

# plot the original classifier
plt.subplot(1, 2, 1)
plt.scatter(X_proj[:, 0], X_proj[:, 1], c=colormap[y], s=40)
plt.title('Real Classification')

# plot the model classifier
plt.subplot(1, 2, 2)
plt.scatter(X_proj[:, 0], X_proj[:, 1], c=colormap[labels], s=40)
plt.title('K-Mean Classification')

plt.show()
