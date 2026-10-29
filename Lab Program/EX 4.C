# ============================================================
# ARTIFICIAL NEURAL NETWORK USING BACKPROPAGATION
# Dataset: Iris Flower Classification
# Google Colab Implementation
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

# ------------------------------------------------------------
# STEP 1: Load Dataset
# ------------------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
class_names = iris.target_names

print("IRIS DATASET")
print("Total Samples:", X.shape[0])
print("Input Features:", X.shape[1])
print("Output Classes:", len(class_names))
print("Classes:", class_names)

df = pd.DataFrame(X, columns=feature_names)
df["Species"] = [class_names[i] for i in y]

display(df.head(10))

# ------------------------------------------------------------
# STEP 2: Split Dataset into Training and Testing Sets
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# Normalize input features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encode target labels
num_classes = len(class_names)

y_train_onehot = np.eye(num_classes)[y_train]

# ------------------------------------------------------------
# STEP 3: Define Activation Functions
# ------------------------------------------------------------

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(a):
    return a * (1 - a)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_values = np.exp(x)
    return exp_values / np.sum(
        exp_values, axis=1, keepdims=True
    )

# ------------------------------------------------------------
# STEP 4: Initialize ANN Parameters
# Architecture:
# 4 Input Neurons
# 8 Hidden Neurons
# 3 Output Neurons
# ------------------------------------------------------------

np.random.seed(42)

input_neurons = X_train.shape[1]
hidden_neurons = 8
output_neurons = num_classes

W1 = np.random.randn(
    input_neurons, hidden_neurons
) * np.sqrt(2 / input_neurons)

b1 = np.zeros((1, hidden_neurons))

W2 = np.random.randn(
    hidden_neurons, output_neurons
) * np.sqrt(2 / hidden_neurons)

b2 = np.zeros((1, output_neurons))

learning_rate = 0.1
epochs = 2000

loss_history = []

# ------------------------------------------------------------
# STEP 5: Backpropagation Training
# ------------------------------------------------------------

for epoch in range(epochs):

    # FORWARD PROPAGATION

    Z1 = np.dot(X_train, W1) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)

    # Calculate Cross-Entropy Loss

    m = X_train.shape[0]

    loss = -np.mean(
        np.sum(
            y_train_onehot * np.log(A2 + 1e-12),
            axis=1
        )
    )

    loss_history.append(loss)

    # BACKWARD PROPAGATION

    # Output layer error
    dZ2 = (A2 - y_train_onehot) / m

    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    # Hidden layer error
    dA1 = np.dot(dZ2, W2.T)

    dZ1 = dA1 * sigmoid_derivative(A1)

    dW1 = np.dot(X_train.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # UPDATE WEIGHTS AND BIASES

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # Display training progress
    if (epoch + 1) % 200 == 0:
        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {loss:.4f}"
        )

# ------------------------------------------------------------
# STEP 6: Prediction Function
# ------------------------------------------------------------

def predict(X):

    hidden_output = sigmoid(
        np.dot(X, W1) + b1
    )

    output = softmax(
        np.dot(hidden_output, W2) + b2
    )

    return np.argmax(output, axis=1)

# ------------------------------------------------------------
# STEP 7: Test the ANN
# ------------------------------------------------------------

y_pred = predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print("ANN TESTING RESULTS")
print("===================================")

print("Actual Labels:    ", y_test)
print("Predicted Labels: ", y_pred)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

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

# ------------------------------------------------------------
# STEP 8: Test a New Flower Sample
# ------------------------------------------------------------

new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])

new_flower_scaled = scaler.transform(new_flower)

prediction = predict(new_flower_scaled)

print("\n===================================")
print("NEW SAMPLE CLASSIFICATION")
print("===================================")

print("Input Measurements:", new_flower[0])

print(
    "Predicted Flower Species:",
    class_names[prediction[0]]
)

# ------------------------------------------------------------
# STEP 9: Plot Training Loss
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(loss_history)

plt.title("ANN Training Loss using Backpropagation")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True)

plt.show()

print("\nBackpropagation ANN Executed Successfully!")