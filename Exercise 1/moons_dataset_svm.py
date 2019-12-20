import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.datasets import make_moons
from matplotlib import pyplot
from pandas import DataFrame

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

h = .02

# SVM regularization parameter
C = 1000
poly_svc = svm.SVC(kernel='poly', degree=5, C=C).fit(X, y)

# assign a mesh to each point
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
plt.subplot(1, 1, 1)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

# predict the color for each x,y point

Z = poly_svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# makes figure
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title("SVC with linear kernel")

plt.show()
