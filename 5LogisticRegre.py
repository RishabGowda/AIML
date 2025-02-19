import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X[:, :2], (y != 0) * 1, test_size=0.4, random_state=9)
X_train, X_test = StandardScaler().fit_transform(X_train), StandardScaler().fit(X_train).transform(X_test)
clf = LogisticRegression().fit(X_train, y_train)

xx, yy = np.meshgrid(*[np.arange(i.min()-1, i.max()+1, 0.1) for i in X_train.T])
plt.contourf(xx, yy, clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape), alpha=0.4)
plt.scatter(*X_train.T, c=y_train, alpha=0.8), plt.show()
