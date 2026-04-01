import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read dataset2.csv
try:
    dataset2 = pd.read_csv('dataset2.csv')
    train_df = dataset2[dataset2['split'] == 'train']
    val_df = dataset2[dataset2['split'] == 'val']
except FileNotFoundError:
    print("dataset2.csv not found. Please ensure the file is in the correct directory.")
    exit()

# get train data
x_train = train_df['x'].values.reshape(-1, 1)
y_train = train_df['y'].values.reshape(-1, 1)

# get validation data
x_val = val_df['x'].values.reshape(-1, 1)
y_val = val_df['y'].values.reshape(-1, 1)

# build matrix & normal equation
def solve_poly_regression(x_input, y_target, degree):
    # create vandermonde matrix
    X = np.hstack([x_input ** i for i in range(degree + 1)])
    # normal equation, w = (X^T X)^(-1) X^T y
    w = np.linalg.solve(X.T @ X, X.T @ y_target)
    return w

def get_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def get_x_matrix(x_input, degree):
    return np.hstack([x_input ** i for i in range(degree + 1)]) 

### ---------------------
# problem 2
# (a) m = 1 to 11
degrees = range(1, 12)
train_mses = []
val_mses = []

for i in degrees:
    w = solve_poly_regression(x_train, y_train, i)

    X_train_mat = get_x_matrix(x_train, i)
    y_train_pred = X_train_mat @ w
    train_mses.append(np.mean((y_train - y_train_pred) ** 2))

    # use the same w to predict validation set
    X_val_mat = get_x_matrix(x_val, i)
    y_val_pred = X_val_mat @ w
    val_mses.append(np.mean((y_val - y_val_pred) ** 2))

# draw a MSE figure
plt.figure(figsize=(8, 5))
plt.plot(degrees, train_mses, 'o-', label= 'Train MSE')
plt.plot(degrees, val_mses, 's-', label= 'Validation MSE')
plt.yscale('log')       # use log scale for better visualization of MSE differences
plt.xlabel('Polynomial Degree (m)')
plt.ylabel('MSE (log scale)')
plt.title('Train vs Validation MSE')
plt.legend()
plt.grid(True)
plt.show()

# draw a 2x2 subplot of fitted curves for m = 1, 3, 8, 11
plot_degrees = [1, 3, 8, 11]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

x_range = np.linspace(0, 1, 100).reshape(-1, 1)

for i, m in enumerate(plot_degrees):
    w = solve_poly_regression(x_train, y_train, m)

    axes[i].scatter(x_train, y_train, color='blue', label='Train Data')
    axes[i].scatter(x_val, y_val, color='orange', label='Validation Data', alpha=0.5)

    X_line = get_x_matrix(x_range, m)
    y_line = X_line @ w
    axes[i].plot(x_range, y_line, 'r-', label=f'm={m}')

    axes[i].set_title(f'Degree m={m}')
    axes[i].set_ylim(-2, 2)     # limit y range to observe overfitting
    axes[i].legend()

plt.tight_layout()
plt.show()