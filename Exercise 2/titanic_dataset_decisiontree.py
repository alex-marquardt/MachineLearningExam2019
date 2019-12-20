import pandas as pd
import pydotplus
import matplotlib.image as mpimg
import io
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix

# import dataset
data = pd.read_csv('titanic_800.csv', sep=',', header=0)

# yvalues only contain the survived column
yvalues = pd.DataFrame(dict(Survived=[]), dtype=int)
yvalues["Survived"] = data["Survived"].copy()

# gives row with missing data the average of that column
avg = data["Age"].mean()
data['Age'] = data['Age'].fillna(avg)

# delete irrelevant columns
data.drop('Survived', axis=1, inplace=True)
data.drop('PassengerId', axis=1, inplace=True)
data.drop('Name', axis=1, inplace=True)
data.drop('Ticket', axis=1, inplace=True)
data.drop('Fare', axis=1, inplace=True)
data.drop('Cabin', axis=1, inplace=True)
data.drop('Embarked', axis=1, inplace=True)

# replace female with 1.0 and male with 0.0
data['Sex'] = data['Sex'].replace(['male'], '0.0')
data['Sex'] = data['Sex'].replace(['female'], '1.0')

# split data up in train and test sets
xtrain = data.head(600)
xtest = data.tail(200)
ytrain = yvalues.head(600)
ytest = yvalues.tail(200)

# labels for picture
target_names = ['Die', 'Survived']
feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']

# train classifier
tree_clf = DecisionTreeClassifier(max_depth=10)  # levels of decision tree
tree_clf.fit(data, yvalues)

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
pydotplus.graph_from_dot_data(dot_data.getvalue()).write_png(filename)
img = mpimg.imread(filename)

# image size in plt
plt.figure(figsize=(8, 8))
imgplot = plt.imshow(img)
plt.show()

# to predict on our xtest set
predictions = tree_clf.predict(xtest)

# print out prediction accuracy
matrix = confusion_matrix(ytest, predictions)
print(matrix)
tp, fn, fp, tn = matrix.ravel()
print("(TP, FN, FP, TN):", (tp, fn, fp, tn))
print(classification_report(ytest, predictions))
print("Accuracy:", tree_clf.score(xtest, ytest))  # print the accuracy of the testdata
