import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from matplotlib import pyplot
from pandas import DataFrame
from matplotlib.colors import ListedColormap

# makes 100 moons - our dataset
X, y = make_moons(n_samples=100, noise=0.1)

# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
colors = {0: 'red', 1: 'blue'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])

pyplot.show()

# running kmeans clustering into k clusters
k = 2
kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
labels = kmeans.labels_

# the centers of the clusters
clusters = kmeans.cluster_centers_

# set the colors
cmap_bold = [ListedColormap(['#FF0000', '#00FF00'])]

# now plot the same points, but this time assigning the colors to indicate the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, edgecolor='black', cmap=cmap_bold[0], s=20)
plt.show()
