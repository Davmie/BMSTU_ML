import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

SIZE = 100
DEGREE_SIZE = 15

np.random.seed(0)
x = np.linspace(-10, 10, SIZE)
theta1 = 1
theta2 = 5
theta3 = 2

y_true = theta1 * x + theta2 * np.sin(x) + theta3
noise = np.random.normal(0, 1, SIZE)
y = y_true + noise

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2, random_state=0)

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

plt.scatter(x, y, label='Данные с шумом', color='blue', alpha=0.5)
plt.plot(x, y_true, label='Истинное значение', color='green')
plt.plot(x, y_pred, color='red', label='Полиномиальная аппроксимация')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Аппроксимация полиномом степени {optimal_degree}')
plt.legend()
plt.grid(True)
plt.show()