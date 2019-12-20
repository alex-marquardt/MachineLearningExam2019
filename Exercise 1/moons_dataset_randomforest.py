import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
accuracy = []
error = []

# train classifier
for i in range(1, 10):
    clf = RandomForestClassifier(n_estimators=i)
    clf.fit(xtrain, ytrain)
    acc = clf.score(xtest, ytest)
    accuracy.append(acc)

# plot out figure size and labels
plt.figure(figsize=(8, 8))
plt.plot(accuracy, label='Accuracy')
plt.legend()
plt.title("RandomForest training - different number of trees")
plt.xlabel("Number of Trees used")
plt.show()
