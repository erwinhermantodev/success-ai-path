from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import pandas as pd

# Load iris dataset for demonstration
iris = load_iris()
X = iris.data
y = iris.target

# apply PCA to reduce to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Create a DataFrame for a Visualization
df = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
df['Target'] = y

# Visualize the reduce-dimensional data
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x='Principal Component 1', y='Principal Component 2', hue="Target", data=df)
plt.title('PCA iris dataset')
plt.show()