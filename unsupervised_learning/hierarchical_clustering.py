from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Generate random data for demonstration
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# Perform hierarchical clustering using AgglomerativeClustering
agg_clustering = AgglomerativeClustering(n_clusters=3, linkage="ward")
agg_labels = agg_clustering.fit_predict(X)

# create a dendrogram for visualization
linkage_matrix = linkage(X, method="ward")
dendrogram(linkage_matrix)
plt.title("Hierarchical Clustering Dendrogram")
plt.show()