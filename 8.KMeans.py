import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris=load_iris()
X=iris.data

k=3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) 
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_ 
colors=['r','g','b']
7
for i in range(k):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], c=colors[i], label=f'Cluster {i+1}')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='black', label='Centroids')
plt.title('KMeans Clustering for Iris')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()
