import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import decomposition
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from timeit import default_timer as timer

# load dataset
faces = datasets.fetch_olivetti_faces()
faces.data.shape

# As usual, then lest split the dataset in a train and a test dataset.
X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, random_state=0)

# Lets downscale the original pics with PCA

# n_components = Number of components to keep,
# Whitening = true can sometimes
# improve the predictive accuracy of the downstream estimators
# by making their data respect some hard-wired assumptions.

pca = decomposition.PCA(n_components=150, whiten=True)
pca.fit(X_train)

# With a PCA projection, the original pictures, train and test,
# can now be projected onto the PCA basis:
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

start = timer()  # timer start

# neutral network classifier
mlp = MLPClassifier(hidden_layer_sizes=(36), max_iter=1000, random_state=0, alpha=0.5)
mlp.fit(X_train_pca, y_train)

end = timer()  # timer end

# accuracy prediction
y_pred = mlp.predict(X_test_pca)
print(metrics.classification_report(y_test, y_pred))

# It is now time to evaluate how well this classification did.
# Lets look at the first 25 pics in he test set.
fig = plt.figure(figsize=(8, 6))
for i in range(25):
    ax = fig.add_subplot(5, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(X_test[i].reshape(faces.images[0].shape),
              cmap=plt.cm.bone)
    y_pred = mlp.predict(X_test_pca[i, np.newaxis])[0]
    color = ('blue' if y_pred == y_test[i] else 'red')
    ax.set_title(y_pred, fontsize='small', color=color)

plt.show()

print("Timer:", end - start)  # print time
