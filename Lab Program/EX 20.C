import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = pd.DataFrame({
    'Month': np.arange(1, 13),
    'Sales': [120, 135, 128, 150, 160, 155,
              175, 180, 190, 185, 210, 220]
})

X = data[['Month']]
y = data['Sales']

model = LinearRegression()
model.fit(X, y)

pred = model.predict(X)

print("Historical Sales:")
print(data.to_string(index=False))

print("\nMean Absolute Error:",
      round(mean_absolute_error(y, pred), 2))

future = pd.DataFrame({'Month': [13, 14, 15]})
future_sales = model.predict(future)

print("\nFuture Sales Prediction:")
for month, sales in zip(future['Month'], future_sales):
    print(f"Month {month}: {sales:.2f} units")

plt.scatter(data['Month'], data['Sales'], label='Actual Sales')
plt.plot(data['Month'], pred, label='Regression Line')
plt.scatter(future['Month'], future_sales,
            marker='x', label='Predicted Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Future Sales Prediction')
plt.legend()
plt.show()