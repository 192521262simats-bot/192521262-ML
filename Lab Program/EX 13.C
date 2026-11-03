import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.DataFrame({
    'Age': [1, 2, 3, 4, 5, 6, 2, 3, 5, 7, 1, 4],
    'Mileage': [10000, 20000, 30000, 40000, 50000, 60000,
                18000, 35000, 55000, 70000, 8000, 45000],
    'EngineCC': [1200, 1200, 1500, 1200, 1500, 1000,
                 1500, 1200, 1000, 1500, 1200, 1500],
    'Price': [900000, 800000, 750000, 600000, 550000,
              350000, 820000, 680000, 400000, 300000,
              950000, 580000]
})

X = data[['Age', 'Mileage', 'EngineCC']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("MAE:", round(mean_absolute_error(y_test, pred), 2))
print("R2 Score:", round(r2_score(y_test, pred), 4))

car = pd.DataFrame(
    [[3, 25000, 1200]],
    columns=X.columns
)
print("Predicted Car Price: Rs.",
      round(model.predict(car)[0], 2))