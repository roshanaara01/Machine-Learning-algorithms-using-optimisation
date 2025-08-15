import numpy as np

def hypothesis(theta, X):
    return 1 / (1 + np.exp(-np.dot(X, theta)))

def cost_function(theta, X, y, lambda_value):
    m = len(y)
    h = hypothesis(theta, X)
    regularization_term = (lambda_value / (2 * m)) * np.sum(theta[1:]**2)
    cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) + regularization_term
    return cost

def gradient(theta, X, y, lambda_value):
    m = len(y)
    h = hypothesis(theta, X)
    regularization_term = (lambda_value / m) * np.insert(theta[1:], 0, 0)
    grad = (1 / m) * np.dot(X.T, (h - y)) + regularization_term
    return grad

def newton_raphson_optimization(X, y, lambda_value, max_iterations):
    m, n = X.shape
    theta = np.zeros(n)
    
    for i in range(max_iterations):
        cost = cost_function(theta, X, y, lambda_value)
        grad = gradient(theta, X, y, lambda_value)
        hessian = (1 / m) * np.dot(X.T, np.dot(np.diag(hypothesis(theta, X) * (1 - hypothesis(theta, X))), X))
        theta -= np.linalg.solve(hessian, grad)
    
    return theta

# Example usage
X = ...  # Your feature matrix
y = ...  # Your target vector
lambda_value = 0.1
max_iterations = 100
optimized_theta = newton_raphson_optimization(X, y, lambda_value, max_iterations)

import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(theta, X, y):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

def gradient(theta, X, y):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    grad = (1 / m) * np.dot(X.T, (h - y))
    return grad

# Example usage
X = ...  # Your feature matrix
y = ...  # Your target vector
theta = np.zeros(X.shape[1])
cost = cost_function(theta, X, y)
grad = gradient(theta, X, y)


from sklearn.naive_bayes import GaussianNB

# Example usage
X_train = ...  # Your training data
y_train = ...  # Corresponding training labels
X_test = ...   # Your test data

naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(X_train, y_train)
predictions = naive_bayes_classifier.predict(X_test)


from sklearn.naive_bayes import GaussianNB

# Example usage
X_train = ...  # Your training data
y_train = ...  # Corresponding training labels
X_test = ...   # Your test data

naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(X_train, y_train)
predictions = naive_bayes_classifier.predict(X_test)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example data
x_values = ...  # X values for the surface
y_values = ...  # Y values for the surface
z_values = ...  # Z values for the surface

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_values, y_values, z_values, cmap='viridis')

# Set labels and show the plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
