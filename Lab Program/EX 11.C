import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.DataFrame({
    'Income': [25000, 30000, 45000, 60000, 80000, 90000,
               20000, 40000, 70000, 55000, 35000, 95000],
    'Debt': [20000, 25000, 15000, 10000, 12000, 8000,
             22000, 18000, 15000, 12000, 20000, 5000],
    'LatePayments': [5, 4, 2, 1, 0, 0, 6, 3, 1, 1, 4, 0],
    'CreditScore': ['Poor', 'Poor', 'Average', 'Good',
                    'Good', 'Good', 'Poor', 'Average',
                    'Good', 'Good', 'Poor', 'Good']
})

X = data[['Income', 'Debt', 'LatePayments']]
y = data['CreditScore']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, pred), 2))
print(classification_report(y_test, pred, zero_division=0))

new_customer = pd.DataFrame(
    [[65000, 10000, 1]],
    columns=X.columns
)
print("Predicted Credit Category:",
      model.predict(new_customer)[0])