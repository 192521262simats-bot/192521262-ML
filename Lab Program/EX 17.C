import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.DataFrame({
    'RAM': [2, 2, 3, 3, 4, 4, 6, 6, 8, 8, 12, 12,
            3, 4, 6, 8, 2, 6, 8, 12],
    'Battery': [2000, 2500, 3000, 2800, 3500, 4000, 4200,
                4500, 5000, 4800, 5000, 6000, 3200, 3600,
                4300, 5200, 2200, 4600, 5100, 5500],
    'Storage': [16, 32, 32, 64, 64, 128, 128, 128, 256, 256,
                512, 512, 32, 64, 128, 256, 16, 128, 256, 512],
    'PriceRange': [0, 0, 0, 0, 1, 1, 1, 1, 2, 2,
                   3, 3, 0, 1, 1, 2, 0, 1, 2, 3]
})

X = data[['RAM', 'Battery', 'Storage']]
y = data['PriceRange']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100, random_state=42
)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("\nClassification Report:\n",
      classification_report(y_test, pred, zero_division=0))

mobile = pd.DataFrame(
    [[8, 5000, 256]], columns=X.columns
)
print("Predicted Price Category:",
      model.predict(mobile)[0])