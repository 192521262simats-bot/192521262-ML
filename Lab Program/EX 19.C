import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.DataFrame({
    'Income': ['High', 'Medium', 'Low', 'High', 'Medium',
               'Low', 'High', 'Medium', 'Low', 'High',
               'Medium', 'Low', 'High', 'Medium', 'Low'],
    'CreditHistory': ['Good', 'Good', 'Poor', 'Good', 'Poor',
                      'Good', 'Poor', 'Good', 'Poor', 'Good',
                      'Good', 'Good', 'Good', 'Poor', 'Poor'],
    'Employment': ['Stable', 'Stable', 'Unstable', 'Stable',
                   'Unstable', 'Stable', 'Unstable', 'Stable',
                   'Unstable', 'Stable', 'Stable', 'Unstable',
                   'Unstable', 'Stable', 'Unstable'],
    'LoanStatus': ['Approved', 'Approved', 'Rejected',
                   'Approved', 'Rejected', 'Rejected',
                   'Rejected', 'Approved', 'Rejected',
                   'Approved', 'Approved', 'Rejected',
                   'Approved', 'Rejected', 'Rejected']
})

encoders = {}
encoded = pd.DataFrame()

for col in data.columns:
    encoders[col] = LabelEncoder()
    encoded[col] = encoders[col].fit_transform(data[col])

X = encoded[['Income', 'CreditHistory', 'Employment']]
y = encoded['LoanStatus']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = CategoricalNB()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, pred), 2))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
print("\nClassification Report:\n",
      classification_report(
          y_test, pred,
          labels=sorted(y.unique()),
          target_names=encoders['LoanStatus'].inverse_transform(
              sorted(y.unique())
          ),
          zero_division=0
      ))

new_applicant = pd.DataFrame({
    'Income': ['High'],
    'CreditHistory': ['Good'],
    'Employment': ['Stable']
})

new_encoded = pd.DataFrame({
    col: encoders[col].transform(new_applicant[col])
    for col in new_applicant.columns
})

prediction = model.predict(new_encoded)[0]

print("Loan Prediction:",
      encoders['LoanStatus'].inverse_transform([prediction])[0])