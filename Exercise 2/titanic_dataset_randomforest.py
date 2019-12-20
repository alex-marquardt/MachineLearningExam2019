import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# import dataset
data = pd.read_csv('titanic_800.csv ', sep=',', header=0)

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

xtrain, xtest, ytrain, ytest = train_test_split(data, yvalues.values.ravel(), test_size=0.2)
accuracy = []
error = []

# train classifier
for i in range(1, 20):
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

# to predict on our xtest set
predictions = clf.predict(xtest)

# print out prediction accuracy
matrix = confusion_matrix(ytest, predictions)
print(matrix)
tp, fn, fp, tn = matrix.ravel()
print("(TP, FN, FP, TN):", (tp, fn, fp, tn))
print(classification_report(ytest, predictions))
print("Accuracy:", clf.score(xtest, ytest))  # print the accuracy of the testdata
