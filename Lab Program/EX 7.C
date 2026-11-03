from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target,
    test_size=0.2, random_state=42, stratify=data.target
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=3000)
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
print("First 5 predictions:", pred[:5])