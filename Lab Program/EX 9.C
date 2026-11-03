import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

X = np.arange(1, 11).reshape(-1, 1)
y = np.array([2, 5, 10, 17, 26, 37, 50, 65, 82, 101])

linear = LinearRegression()
poly = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)

linear.fit(X, y)
poly.fit(X, y)

p1 = linear.predict(X)
p2 = poly.predict(X)

print("Linear Regression MSE:", round(mean_squared_error(y, p1), 2))
print("Polynomial Regression MSE:", round(mean_squared_error(y, p2), 2))
print("Linear prediction at X=11:", round(linear.predict([[11]])[0], 2))
print("Polynomial prediction at X=11:", round(poly.predict([[11]])[0], 2))

plt.scatter(X, y, label="Actual Data")
plt.plot(X, p1, label="Linear Regression")
plt.plot(X, p2, label="Polynomial Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()