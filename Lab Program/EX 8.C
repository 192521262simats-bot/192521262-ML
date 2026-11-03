import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([2, 4, 5, 4, 5, 7, 8, 9])

model = LinearRegression()
model.fit(X, y)

pred = model.predict(X)
print("Slope:", round(model.coef_[0], 2))
print("Intercept:", round(model.intercept_, 2))
print("Prediction for X=9:", round(model.predict([[9]])[0], 2))
print("Mean Squared Error:", round(mean_squared_error(y, pred), 2))
print("R2 Score:", round(r2_score(y, pred), 4))