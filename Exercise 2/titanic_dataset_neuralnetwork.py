import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
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

# scale/transform the xtrain and xtest data
scaler = StandardScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

# train classifier
mlp = MLPClassifier(hidden_layer_sizes=(200, 100, 50, 25, 10), max_iter=1000, random_state=0, alpha=0.5)
mlp.fit(xtrain, ytrain.values.ravel())

# to predict on our xtest set
predictions = mlp.predict(xtest)

# print out prediction accuracy
matrix = confusion_matrix(ytest, predictions)
print(matrix)
tp, fn, fp, tn = matrix.ravel()
print("(TP, FN, FP, TN):", (tp, fn, fp, tn))
print(classification_report(ytest, predictions))
print("Accuracy:", mlp.score(xtest, ytest))  # print the accuracy of the testdata
