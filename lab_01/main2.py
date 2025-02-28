import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X_START = -2
X_END = 2
L = 50
DEGREE_SIZE = 50

np.random.seed(0)


def runge(x):
    return 1 / (1 + 25 * x ** 2)


x = np.linspace(X_START, X_END, L)

x_train = []
x_test = []

for i in range(1, L + 1):
    x_train.append(4 * (i - 1) / (L - 1) - 2)
    # if i == L:
    #     continue
    x_test.append(4 * (i - 0.5) / (L - 1) - 2)

x_train = np.array(x_train)
x_test = np.array(x_test)

y = []
for i in range(L):
    y.append(runge(x[i]))

y_train = []
for i in range(L):
    y_train.append(runge(x_train[i]))

y_test = []
for i in range(L):
    y_test.append(runge(x_test[i]))

degrees = range(1, DEGREE_SIZE)

mse_train = []
mse_test = []

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    x_train_poly = poly.fit_transform(x_train.reshape(-1, 1))
    x_test_poly = poly.transform(x_test.reshape(-1, 1))

    model = LinearRegression()
    model.fit(x_train_poly, y_train)

    y_train_pred = model.predict(x_train_poly)
    y_test_pred = model.predict(x_test_poly)

    mse_train.append(mean_squared_error(y_train, y_train_pred))
    mse_test.append(mean_squared_error(y_test, y_test_pred))

plt.plot(degrees, mse_train, label='Train MSE')
plt.plot(degrees, mse_test, label='Test MSE')
plt.xlabel('Степень полинома')
plt.ylabel('Среднеквдратичная ошибка (MSE)')
plt.title('MSE vs Polynomial Degree')
plt.legend()
plt.grid(True)
plt.show()

optimal_degree = degrees[np.argmin(mse_test)]
print(f"Оптимальная степень полинома: {optimal_degree}")

poly = PolynomialFeatures(degree=optimal_degree)
x_poly = poly.fit_transform(x.reshape(-1, 1))
model = LinearRegression()
model.fit(x_poly, y)

y_pred = model.predict(x_poly)

plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, label='Обучающие данные', color='blue', alpha=0.5)
plt.scatter(x_test, y_test, label='Контрольные данные', color='green', alpha=0.5)
plt.plot(x, y_pred, color='red', label=f'Полином степени {optimal_degree}')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Аппроксимация полиномом степени {optimal_degree}')
plt.legend()
plt.grid(True)
plt.show()