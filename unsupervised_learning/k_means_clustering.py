from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Generate random data for demonstration
np.random.seed(42)
X = np.array(([[1,2],[5,8],[1.5,1.8],[8,8],[1, 0.6],[9, 11]]))

# specify the number of cluster (K)
kmeans = KMeans(n_clusters=2)

# fit the K-means model to the data
kmeans.fit(X)

# Get cluster assignment and centroid
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Visualize the cluster
colors = ["g.", "r."]
for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
    
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)
plt.show()