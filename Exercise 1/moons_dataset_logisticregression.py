import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
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

# draw new points
pos = np.where(y == 1)
neg = np.where(y == 0)
plt.plot(X[pos[0], 0], X[pos[0], 1], 'bo')
plt.plot(X[neg[0], 0], X[neg[0], 1], 'ro')
plt.xlim([min(X[:, 0]), max(X[:, 0])])
plt.ylim([min(X[:, 1]), max(X[:, 1])])

logreg = linear_model.LogisticRegression(C=1000)
h = .005

# train our classifier
model = logreg.fit(X, y)
score = logreg.score(X, y)

# assign a mesh to each point
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# predict the color for each x,y point
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# makes figure
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

plt.show()
