import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.DataFrame({
    'Area': [800, 1000, 1200, 1500, 1800, 2000, 900, 1300, 1600, 2200, 1100, 1700],
    'Bedrooms': [2, 2, 3, 3, 3, 4, 2, 3, 3, 4, 2, 3],
    'Age': [10, 8, 5, 4, 3, 2, 12, 6, 4, 1, 9, 3],
    'Price': [2500000, 3000000, 4000000, 5000000,
              6000000, 7500000, 2200000, 4200000,
              5500000, 8500000, 3200000, 5800000]
})

X = data[['Area', 'Bedrooms', 'Age']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100, random_state=42
)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("MAE:", round(mean_absolute_error(y_test, pred), 2))
print("R2 Score:", round(r2_score(y_test, pred), 4))

house = pd.DataFrame(
    [[1400, 3, 5]], columns=X.columns
)
print("Predicted House Price: Rs.",
      round(model.predict(house)[0], 2))