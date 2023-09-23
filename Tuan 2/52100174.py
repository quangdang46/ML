import numpy as np
import pandas as pd

def gradient_descent(X, y, alpha, num_iterations):
    theta = np.random.randn(X.shape[1])

    for i in range(num_iterations):
        h = X @ theta 
        gradient = 1 / X.shape[0] * X.T @ (h - y)
        theta -= alpha * gradient 

    return theta


# Load 
data = pd.read_csv('house_practice.csv')
X = data[['Size', 'Bedrooms']].values
y = data['Price'].values


mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean) / std

X = np.column_stack((np.ones(X.shape[0]), X))

alpha = 0.01
num_iterations = 1000

theta = gradient_descent(X, y, alpha, num_iterations)

print("Learned Parameters (theta):", theta)

new_data = np.array([[2104, 3]])

new_data = (new_data - mean) / std
new_data = np.insert(new_data, 0, 1)  
predicted_price = np.dot(new_data, theta)
print("Predicted Price:", predicted_price)

