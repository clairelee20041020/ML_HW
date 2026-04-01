import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

# --- 1. 數據讀取與準備 ---
try:
    dataset2 = pd.read_csv('dataset2.csv')
    train_df = dataset2[dataset2['split'] == 'train']
    val_df = dataset2[dataset2['split'] == 'val']
    
    x_train, y_train = train_df['x'].values.reshape(-1, 1), train_df['y'].values.reshape(-1, 1)
    x_val, y_val = val_df['x'].values.reshape(-1, 1), val_df['y'].values.reshape(-1, 1)
except FileNotFoundError:
    print("dataset2.csv not found."); exit()

def get_x_matrix(x_input, degree):
    return np.hstack([x_input ** i for i in range(degree + 1)]) 

def solve_l2_regression(x_input, y_target, degree, lambd):
    X = get_x_matrix(x_input, degree)
    I = np.eye(degree + 1)
    # L2 Regularization (Ridge): (X^T X + lambda * I)^-1 * X^T * y
    w = np.linalg.solve(X.T @ X + lambd * I, X.T @ y_target)
    return w

# --- 2. 實驗參數設定 ---
fixed_m = 11    # 固定次數為 11，觀察不同 lambda 的效果
lambdas = [0.001, 0.01, 0.1, 1.0]
x_range = np.linspace(0, 1, 200).reshape(-1, 1)

# --- 3. draw L2 regression figure ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(f"L2 Regularization Effect (Fixed Degree m = {fixed_m})", fontsize=16)
axes = axes.flatten()

for i, lambd in enumerate(lambdas):
    # 訓練模型
    w = solve_l2_regression(x_train, y_train, fixed_m, lambd)
    
    # 計算 MSE
    y_train_pred = get_x_matrix(x_train, fixed_m) @ w
    y_val_pred = get_x_matrix(x_val, fixed_m) @ w
    train_mse = np.mean((y_train - y_train_pred) ** 2)
    val_mse = np.mean((y_val - y_val_pred) ** 2)
    
    # 繪製原始數據
    axes[i].scatter(x_train, y_train, color='blue', label='Train Data (n=12)')
    axes[i].scatter(x_val, y_val, color='orange', alpha=0.4, label='Val Data (n=30)')
    
    # 繪製擬合曲線
    y_line = get_x_matrix(x_range, fixed_m) @ w
    axes[i].plot(x_range, y_line, 'r-', linewidth=2, label=f'L2 (λ={lambd})')
    
    # 設定標題與範圍
    axes[i].set_title(f"Lambda = {lambd}\nTrain MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f}")
    axes[i].set_ylim(-1.5, 1.5) # 固定 y 軸範圍才好比較平滑度
    axes[i].legend(loc='upper right', fontsize='small')
    axes[i].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# --- 4. draw L1 regression figure ---
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
fig2.suptitle(f"L1 Regularization (Lasso) Effect (Fixed m = {fixed_m})", fontsize=16)
axes2 = axes2.flatten()

for i, lambd in enumerate(lambdas):
    # 1. 準備特徵矩陣 (注意：Lasso 內部會處理截距，所以我們給 x, x^2...x^m 即可)
    # 但為了統一，我們直接用 get_x_matrix 產出的矩陣，並關閉 Lasso 的 fit_intercept
    X_train_poly = get_x_matrix(x_train, fixed_m)
    
    # 2. 建立並訓練 Lasso 模型
    # alpha 在 sklearn 中等同於題目要求的 lambda
    lasso_model = Lasso(alpha=lambd, max_iter=100000)
    lasso_model.fit(X_train_poly, y_train)
    
    # 3. 取得權重 w (包含截距)
    w_l1 = lasso_model.coef_
    w_l1[0] = lasso_model.intercept_[0] # 替換第一項為截距
    
    # 4. 預測與計算 MSE
    y_train_pred = X_train_poly @ w_l1
    y_val_pred = get_x_matrix(x_val, fixed_m) @ w_l1
    train_mse = np.mean((y_train.flatten() - y_train_pred)**2)
    val_mse = np.mean((y_val.flatten() - y_val_pred)**2)

    # 5. 繪圖
    axes2[i].scatter(x_train, y_train, color='blue', label='Train')
    axes2[i].scatter(x_val, y_val, color='orange', alpha=0.4, label='Val')
    
    y_line = get_x_matrix(x_range, fixed_m) @ w_l1
    axes2[i].plot(x_range, y_line, 'g-', linewidth=2, label=f'L1 (λ={lambd})')
    
    # 統計非零權重的數量 (特徵選擇效果)
    non_zero_count = np.sum(np.abs(w_l1) > 1e-5)
    
    axes2[i].set_title(f"Lambda = {lambd}\nNon-zero weights: {non_zero_count}\nVal MSE: {val_mse:.4f}")
    axes2[i].set_ylim(-1.5, 1.5)
    axes2[i].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

