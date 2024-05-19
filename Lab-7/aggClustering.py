from sklearn.cluster import AgglomerativeClustering
import numpy as np

X = np.array([[1, 2], [1, 4], [1, 0], [4, 0]])

def hierarchical_clustering(X, n_clusters):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    model.fit(X)
    return model.labels_

n_clusters = 2
labels = hierarchical_clustering(X, n_clusters)
print("Cluster labels:", labels)
