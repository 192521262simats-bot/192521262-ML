from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target,
    test_size=0.2, random_state=42, stratify=iris.target
)

model = GaussianNB()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

sample = [[5.1, 3.5, 1.4, 0.2]]
result = model.predict(sample)[0]

print("Predicted Flower:", iris.target_names[result])