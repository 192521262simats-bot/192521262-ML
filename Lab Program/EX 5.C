# ============================================================
# IMPLEMENTATION OF K-NEAREST NEIGHBOURS (K-NN) IN PYTHON
# Dataset: Iris Flower Classification
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

# STEP 1: Load Dataset

iris = load_iris()

X = iris.data
y = iris.target

class_names = iris.target_names

df = pd.DataFrame(X, columns=iris.feature_names)
df["Species"] = [class_names[i] for i in y]

print("IRIS DATASET")
display(df.head(10))

# STEP 2: Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# STEP 3: Normalize Features

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# STEP 4: Calculate Euclidean Distance

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# STEP 5: Implement K-NN Algorithm

def knn_predict(X_train, y_train, X_test, k):

    predictions = []

    for test_point in X_test:

        distances = []

        # Calculate distance from every training point
        for i in range(len(X_train)):

            distance = euclidean_distance(
                test_point,
                X_train[i]
            )

            distances.append((distance, y_train[i]))

        # Sort distances in ascending order
        distances.sort(key=lambda x: x[0])

        # Select K nearest neighbours
        k_neighbours = distances[:k]

        # Extract neighbour classes
        neighbour_labels = [
            label for distance, label in k_neighbours
        ]

        # Majority voting
        values, counts = np.unique(
            neighbour_labels,
            return_counts=True
        )

        prediction = values[np.argmax(counts)]

        predictions.append(prediction)

    return np.array(predictions)

# STEP 6: Choose K and Make Predictions

k = 5

y_pred = knn_predict(
    X_train,
    y_train,
    X_test,
    k
)

# STEP 7: Evaluate Model

accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print("K-NEAREST NEIGHBOURS RESULTS")
print("===================================")

print("K Value:", k)
print("Actual Labels:", y_test)
print("Predicted Labels:", y_pred)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=class_names
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# STEP 8: Test a New Sample

new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])

new_sample_scaled = scaler.transform(new_sample)

new_prediction = knn_predict(
    X_train,
    y_train,
    new_sample_scaled,
    k
)

print("\n===================================")
print("NEW SAMPLE CLASSIFICATION")
print("===================================")

print("Input Features:", new_sample[0])

print(
    "Predicted Species:",
    class_names[new_prediction[0]]
)

# STEP 9: Compare Different K Values

k_values = range(1, 16)
accuracy_values = []

for k_value in k_values:

    predictions = knn_predict(
        X_train,
        y_train,
        X_test,
        k_value
    )

    score = accuracy_score(y_test, predictions)

    accuracy_values.append(score)

    print(
        f"K = {k_value:2d} | "
        f"Accuracy = {score * 100:.2f}%"
    )

# STEP 10: Plot Accuracy Graph

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    accuracy_values,
    marker='o'
)

plt.title("K-NN Accuracy for Different K Values")
plt.xlabel("Number of Neighbours (K)")
plt.ylabel("Accuracy")
plt.xticks(list(k_values))
plt.grid(True)

plt.show()

print("\nK-NN Algorithm Executed Successfully!")