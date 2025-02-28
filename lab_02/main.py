import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


SIZE = 100
degrees = [11, 16, 21]
alphas = [0.01, 0.05, 0.1, 0.5, 1, 10]


np.random.seed(0)
x = np.linspace(-10, 10, SIZE)
theta1 = 1
theta2 = 5
theta3 = 2

y_true = theta1 * x + theta2 * np.sin(x) + theta3
noise = np.random.normal(0, 1, SIZE)
y = y_true + noise

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


def train_and_print_model(model, x_train_poly, y_train, model_name):
    model.fit(x_train_poly, y_train)
    print(f"{model_name} coefficients: {model.coef_}")


# Функция для расчета MSE
def calculate_mse(model, x_poly, y_true):
    y_pred = model.predict(x_poly)
    return mean_squared_error(y_true, y_pred)


mse_train = {degree: [] for degree in degrees}
mse_test = {degree: [] for degree in degrees}

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    x_train_poly = poly.fit_transform(x_train.reshape(-1, 1))
    x_test_poly = poly.transform(x_test.reshape(-1, 1))

    model_lr = LinearRegression()
    train_and_print_model(model_lr, x_train_poly, y_train, f"Linear Regression (degree {degree})")
    mse_train[degree].append(calculate_mse(model_lr, x_train_poly, y_train))
    mse_test[degree].append(calculate_mse(model_lr, x_test_poly, y_test))

    for alpha in alphas:
        model_ridge = Ridge(alpha=alpha)
        train_and_print_model(model_ridge, x_train_poly, y_train, f"Ridge (degree {degree}, alpha {alpha})")
        mse_train[degree].append(calculate_mse(model_ridge, x_train_poly, y_train))
        mse_test[degree].append(calculate_mse(model_ridge, x_test_poly, y_test))

    for alpha in alphas:
        model_lasso = Lasso(alpha=alpha, max_iter=10000)
        train_and_print_model(model_lasso, x_train_poly, y_train, f"Lasso (degree {degree}, alpha {alpha})")
        mse_train[degree].append(calculate_mse(model_lasso, x_train_poly, y_train))
        mse_test[degree].append(calculate_mse(model_lasso, x_test_poly, y_test))

for degree in degrees:
    plt.figure(figsize=(12, 6))
    labels = ['Linear Regression'] + [f'Ridge (alpha={alpha})' for alpha in alphas] + [f'Lasso (alpha={alpha})' for
                                                                                       alpha in alphas]
    x_labels = np.arange(len(labels))
    plt.bar(x_labels - 0.2, mse_train[degree], 0.4, label='Train MSE', color='blue')
    plt.bar(x_labels + 0.2, mse_test[degree], 0.4, label='Test MSE', color='orange')
    plt.xticks(x_labels, labels, rotation=45, ha='right')
    plt.xlabel('Модель')
    plt.ylabel('Среднеквадратичная ошибка (MSE)')
    plt.title(f'MSE для полинома степени {degree}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# optimal_degree = 21
# poly = PolynomialFeatures(degree=optimal_degree)
# x_poly = poly.fit_transform(x.reshape(-1, 1))
# model = LinearRegression()
# model.fit(x_poly, y)
# y_pred = model.predict(x_poly)
#
# plt.figure(figsize=(10, 6))
# plt.scatter(x, y, label='Данные с шумом', color='blue', alpha=0.5)
# plt.plot(x, y_true, label='Истинное значение', color='green')
# plt.plot(x, y_pred, color='red', label='Полиномиальная аппроксимация')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title(f'Аппроксимация полиномом степени {optimal_degree} (Linear Regression)')
# plt.legend()
# plt.grid(True)
# plt.show()
