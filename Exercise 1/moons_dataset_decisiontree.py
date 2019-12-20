import pydotplus
import matplotlib.image as mpimg
import io
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# makes 100 moons
X, y = make_moons(n_samples=100, noise=0.1)

# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
colors = {0: 'red', 1: 'blue'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])

pyplot.show()

# splits data up in training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# makes labels for picture
target_names = ['Red', 'Blue']
feature_names = ['x', 'y']

# train classifier
tree_clf = DecisionTreeClassifier(max_depth=5)  # indicate we do not want the tree to be deeper than 2 levels
tree_clf.fit(X, y)  # training the classifier
accuracy = tree_clf.score(X, y)

# print accuracy
print(accuracy)

for name, score in zip(feature_names, tree_clf.feature_importances_):
    print("feature importance: ", name, score)

dot_data = io.StringIO()

# export picture labels
export_graphviz(tree_clf,
                out_file=dot_data,
                feature_names=feature_names,
                class_names=target_names,
                rounded=True,
                filled=True)

# makes decision tree picture
filename = "tree.png"
pydotplus.graph_from_dot_data(dot_data.getvalue()).write_png(filename)  # write the dot data to a png file
img = mpimg.imread(filename)  # read this png file

# image size in plt
plt.figure(figsize=(8, 8))  # setting the size to 10 x 10 inches of the figure.
imgplot = plt.imshow(img)  # plot the image.
plt.show()
