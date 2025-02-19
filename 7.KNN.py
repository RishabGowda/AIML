import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from collections import Counter

iris=load_iris()
x=iris.data
y=iris.target
class_names=iris.target_names

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

class KNN:
    def __init__(self,k=3):
        self.k=k
    def fit(self,x,y):
        self.x_train=x
        self.y_train=y
    def predict(self,x):
        distances=[np.linalg.norm(x-inst) for inst in self.x_train]
        k_indices=np.argsort(distances)[:self.k]
        k_labels=[self.y_train[i] for i in k_indices]
        self.ctr=Counter(k_labels)
        return self.ctr.most_common(1)[0][0]
    
    def _predict(self,x):
        return [self.predict(inst) for inst in x]

knn=KNN(k=3)
knn.fit(x_train,y_train)
y_pred=knn._predict(x_test)
np.mean(y_pred==y_test)
cm=confusion_matrix(y_test,y_pred)
print("Accuracy:%.4f"%accuracy_score(y_test,y_pred))
print("Predictions:", class_names[y_pred])
print("\n Classification Report:\n",classification_report(y_test,y_pred))
sns.heatmap(cm)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
