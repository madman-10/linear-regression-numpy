# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

class LinearRegression():
    def __init__(self, learning_rate=0.001, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, y):
        #No of training samples, number of features
        self.samples, self.features = X.shape

        #Initializing weights
        self.weights = np.zeros(self.features)
        self.bias = 0
        self.X = X
        self.y = y

        #Gradient Descent Learning
        for i in range(self.iterations):
            self.update_weights()
        
        return self
    
    #helper to update weights
    def update_weights(self):
        y_pred = self.predict(self.X)

        #formula to calculate the gradients
        dw = -(2*(self.X.T).dot(self.y - y_pred)) / self.samples
        db = -2 * np.sum(self.y - y_pred) / self.samples

        #update the weights
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

        return self
    
    def predict(self, X):
        return X.dot(self.weights) + self.bias
    

def mse(y_test, predictions):
    return np.mean((y_test-predictions)**2)


X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1234)


reg = LinearRegression(learning_rate=0.1)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

mse = mse(y_test, predictions)
print(mse)


y_pred_line = reg.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
plt.show()