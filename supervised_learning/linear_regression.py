import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate some random data for demonstration
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
X_new = np.array([[0], [2]])
y_pred = model.predict(X_new)

# Visualize the data and the linear regression line
plt.scatter(X, y, label='Actual Data')
plt.plot(X_new, y_pred, color='red', label='Linear Regression Prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Print the model parameters
print("Intercept (beta0):", model.intercept_)
print("Slope (beta1):", model.coef_)
