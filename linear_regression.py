import numpy as np

class LinearRegression():
    def __init__(self, lr = 0.01, epochs = 1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features) # Initialize weights and bias
        self.bias = 0

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) # Compute gradients
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw # Update weights and bias
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias