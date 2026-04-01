import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read dataset1.csv
try:
    dataset1 = pd.read_csv('dataset1.csv')
    x = dataset1['x'].values.reshape(-1, 1)
    y = dataset1['y'].values.reshape(-1, 1)
except FileNotFoundError:
    print("dataset1.csv not found. Please ensure the file is in the correct directory.")
    exit()

# normalization
x_min, x_max = x.min(), x.max()
x_normalized = (x - x_min) / (x_max - x_min)

# build matrix & normal equation
def solve_poly_regression(x_input, y_target, degree):
    # create vandermonde matrix
    X = np.hstack([x_input ** i for i in range(degree + 1)])
    # normal equation, w = (X^T X)^(-1) X^T y
    w = np.linalg.solve(X.T @ X, X.T @ y_target)
    return w, X

def get_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# problem 1
# (a) m = 2
degree = 2
w2, X2 = solve_poly_regression(x_normalized, y, degree)
y_pred2 = X2 @ w2   # y = w0 + w1*x + w2*x^2

print(f"Problem 1(a): w0, w1, w2 = \n {w2.flatten()}")
print(f"MSE for m=2: {get_mse(y, y_pred2):.4f}")

# (c) m = 3 & m = 8
degree3 = 3
w3, X3 = solve_poly_regression(x_normalized, y, degree3)
y_pred3 = X3 @ w3
print(f"MSE for m=3: {get_mse(y, y_pred3):.4f}")

# if degree = 4 ~ 7, try overfitting
for i in range(4, 8):
    degrees = i
    w, X = solve_poly_regression(x_normalized, y, degrees)
    y_pred = X @ w
    print(f"MSE for m={i}: {get_mse(y, y_pred):.4f}")

degree8 = 8
w8, X8 = solve_poly_regression(x_normalized, y, degree8)
y_pred8 = X8 @ w8
print(f"MSE for m=8: {get_mse(y, y_pred8):.4f}")

# draw curves


plt.figure(figsize=(10, 6))
plt.scatter(x_normalized, y, s=5, color='gray', alpha=0.5, label='Data points')

# 為了讓曲線平滑，建立均勻分布的測試點
x_line = np.linspace(0, 1, 500).reshape(-1, 1)


# m = 7 的曲線
X_line7 = np.hstack([x_line**i for i in range(7 + 1)])
w7, X7 = solve_poly_regression(x_normalized, y, degrees)
y_pred7 = X7 @ w7
plt.plot(x_line, X_line7 @ w7, 'm-', linewidth=2, label=f'm=7 (MSE: {get_mse(y, y_pred7):.2f})')

plt.title('m = 7, good fit')
plt.xlabel('Normalized x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# m=2 的曲線
"""
X_line2 = np.hstack([x_line**i for i in range(degree + 1)])
plt.plot(x_line, X_line2 @ w2, 'g-', linewidth=2, label=f'm=2 (MSE: {get_mse(y, y_pred2):.2f})')

plt.title('Polynomial Regression Comparison (Problem 1, (a))')
plt.xlabel('Normalized x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

"""

# 比較 m=3 和 m=8 的曲線
"""
X_line3 = np.hstack([x_line**i for i in range(degree3 + 1)])
X_line8 = np.hstack([x_line**i for i in range(degree8 + 1)])

plt.plot(x_line, X_line3 @ w3, 'r-', linewidth = 2, label=f'm=3 (MSE: {get_mse(y, y_pred3):.2f})')
plt.plot(x_line, X_line8 @ w8, 'b--', linewidth=2, label=f'm=8 (MSE: {get_mse(y, y_pred8):.2f})')

plt.title('Polynomial Regression Comparison (Problem 1)')
plt.xlabel('Normalized x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

"""