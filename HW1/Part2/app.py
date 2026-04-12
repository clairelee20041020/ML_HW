import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

# =========================
# 工具函數（你原本的 + 修正）
# =========================

def normalize(X):
    return (X - X.min()) / (X.max() - X.min())

def get_x_matrix(x_input, degree):
    return np.hstack([x_input ** i for i in range(degree + 1)])

def solve_l2_regression(x_input, y_target, degree, lambd):
    X = get_x_matrix(x_input, degree)
    I = np.eye(degree + 1)
    w = np.linalg.pinv(X.T @ X + lambd * I) @ (X.T @ y_target)
    return w

def solve_no_reg(x_input, y_target, degree):
    X = get_x_matrix(x_input, degree)
    return np.linalg.pinv(X.T @ X) @ (X.T @ y_target)

# =========================
# 核心功能
# =========================

def run_regression(x_train, y_train, x_test, y_test, degree, lambd, reg_type):
    x_range = np.linspace(x_train.min(), x_train.max(), 200).reshape(-1, 1)

    fig, ax = plt.subplots()

    ax.scatter(x_train, y_train, color='blue', label='Train')
    ax.scatter(x_test, y_test, color='orange', label='Test', alpha=0.4)

    # 選模型
    if reg_type == "L2":
        w = solve_l2_regression(x_train, y_train, degree, lambd)

        y_train_pred = get_x_matrix(x_train, degree) @ w
        y_test_pred = get_x_matrix(x_test, degree) @ w
        y_line = get_x_matrix(x_range, degree) @ w

    elif reg_type == "L1":
        X_train_poly = get_x_matrix(x_train, degree)
        X_test_poly = get_x_matrix(x_test, degree)

        model = Lasso(alpha=lambd, max_iter=100000)
        model.fit(X_train_poly, y_train)

        y_train_pred = model.predict(X_train_poly)
        y_test_pred = model.predict(X_test_poly)
        y_line = model.predict(get_x_matrix(x_range, degree))

    else:
        w = solve_no_reg(x_train, y_train, degree)

        y_train_pred = get_x_matrix(x_train, degree) @ w
        y_test_pred = get_x_matrix(x_test, degree) @ w
        y_line = get_x_matrix(x_range, degree) @ w

    train_mse = np.mean((y_train - y_train_pred.reshape(-1,1))**2)
    test_mse = np.mean((y_test - y_test_pred.reshape(-1,1))**2)

    color = plt.cm.viridis(lambd)  # λ → 顏色

    ax.plot(
        x_range, y_line,
        color='red',
        label=f'{reg_type} λ={lambd}'
    )
    ax.legend()
    ax.set_title(f"Train MSE: {train_mse:.4f} | Test MSE: {test_mse:.4f}")

    return fig, train_mse, test_mse


def overfitting_curve(x_train, y_train, x_test, y_test):
    train_mse_list = []
    test_mse_list = []

    degrees = range(1, 12)

    for m in degrees:
        w = solve_no_reg(x_train, y_train, m)

        y_train_pred = get_x_matrix(x_train, m) @ w
        y_test_pred = get_x_matrix(x_test, m) @ w

        train_mse = np.mean((y_train - y_train_pred)**2)
        test_mse = np.mean((y_test - y_test_pred)**2)

        train_mse_list.append(train_mse)
        test_mse_list.append(test_mse)

    fig, ax = plt.subplots()
    ax.plot(degrees, train_mse_list, marker='o', label="Train MSE")
    ax.plot(degrees, test_mse_list, marker='o', label="Test MSE")
    ax.set_xlabel("Degree")
    ax.set_ylabel("MSE")
    ax.set_title("Overfitting Curve")
    ax.legend()
    ax.grid()

    return fig


def lambda_curve(x_train, y_train, x_test, y_test, degree):
    lambdas = np.logspace(-3, 1, 20)

    train_mse_list = []
    test_mse_list = []

    for l in lambdas:
        w = solve_l2_regression(x_train, y_train, degree, l)

        y_train_pred = get_x_matrix(x_train, degree) @ w
        y_test_pred = get_x_matrix(x_test, degree) @ w

        train_mse = np.mean((y_train - y_train_pred)**2)
        test_mse = np.mean((y_test - y_test_pred)**2)

        train_mse_list.append(train_mse)
        test_mse_list.append(test_mse)

    fig, ax = plt.subplots()
    ax.plot(lambdas, train_mse_list, label="Train MSE")
    ax.plot(lambdas, test_mse_list, label="Test MSE")
    ax.set_xscale("log")
    ax.set_xlabel("Lambda")
    ax.set_ylabel("MSE")
    ax.set_title("Lambda Curve (L2)")
    ax.legend()
    ax.grid()

    return fig

# =========================
# Streamlit UI
# =========================

st.set_page_config(layout="wide")

st.title("Polynomial Regression Interactive Demo")

df = pd.read_csv("IceCreamData.csv")

col_left, col_right = st.columns([1, 3])

with col_left:
    # ===== 控制桿 =====
    st.subheader("Controls")
    
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
    degree = st.slider("Degree (m)", 1, 12, 3)
    lambd = st.slider("Lambda", 0.001, 1.0, 0.01)

    reg_type = st.selectbox("Model", ["None", "L2", "L1"])
    
    # ===== 數據 =====
    st.write("Dataset Statistics")
    st.write(df.head())

    X = df[['Temperature']].values
    y = df['Revenue'].values.reshape(-1,1)

    # Normalize（重要）
    X = normalize(X)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

with col_right:
    
    col_A, col_B = st.columns([2, 1])
    
    with col_A:
        # ===== 主圖 =====
        st.subheader("Interactive Plot")
        fig, train_mse, test_mse = run_regression(
            x_train, y_train, x_test, y_test, degree, lambd, reg_type
        )

        st.pyplot(fig)
        st.write(f"Train MSE: {train_mse:.4f}")
        st.write(f"Test MSE: {test_mse:.4f}")
    
    with col_B:
        # ===== Overfitting Curve =====
        st.subheader("Overfitting Analysis")
        fig2 = overfitting_curve(x_train, y_train, x_test, y_test)
        st.pyplot(fig2)

        # ===== Lambda Curve =====
        st.subheader("Regularization Effect (L2)")
        fig3 = lambda_curve(x_train, y_train, x_test, y_test, degree)
        st.pyplot(fig3)