from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

iris = load_iris()
x, y = iris.data, iris.target
class_names = iris.target_names
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=9)
nb = GaussianNB()
nb.fit(x_train, y_train)
y_pred = nb.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print("Accuracy:%.4f"%accuracy_score(y_test,y_pred))
print("Predictions:", class_names[y_pred])
print("\n Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("\n Classification Report:\n",classification_report(y_test,y_pred))