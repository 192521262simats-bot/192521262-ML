# ============================================================
# DECISION TREE CLASSIFICATION USING ID3 ALGORITHM
# Google Colab Implementation
# ============================================================

import pandas as pd
import math
from collections import Counter

# STEP 1: Create the Play Tennis Dataset

data = {
    'Outlook': [
        'Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain',
        'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain',
        'Sunny', 'Overcast', 'Overcast', 'Rain'
    ],

    'Temperature': [
        'Hot', 'Hot', 'Hot', 'Mild', 'Cool',
        'Cool', 'Cool', 'Mild', 'Cool', 'Mild',
        'Mild', 'Mild', 'Hot', 'Mild'
    ],

    'Humidity': [
        'High', 'High', 'High', 'High', 'Normal',
        'Normal', 'Normal', 'High', 'Normal', 'Normal',
        'Normal', 'High', 'Normal', 'High'
    ],

    'Wind': [
        'Weak', 'Strong', 'Weak', 'Weak', 'Weak',
        'Strong', 'Strong', 'Weak', 'Weak', 'Weak',
        'Strong', 'Strong', 'Weak', 'Strong'
    ],

    'PlayTennis': [
        'No', 'No', 'Yes', 'Yes', 'Yes',
        'No', 'Yes', 'No', 'Yes', 'Yes',
        'Yes', 'Yes', 'Yes', 'No'
    ]
}

df = pd.DataFrame(data)

print("========== TRAINING DATASET ==========")
display(df)

# STEP 2: Calculate Entropy

def entropy(target):
    counts = Counter(target)
    total = len(target)

    result = 0

    for count in counts.values():
        probability = count / total
        result -= probability * math.log2(probability)

    return result


# STEP 3: Calculate Information Gain

def information_gain(dataset, attribute, target):
    total_entropy = entropy(dataset[target])

    weighted_entropy = 0

    for value in dataset[attribute].unique():

        subset = dataset[dataset[attribute] == value]

        weight = len(subset) / len(dataset)

        weighted_entropy += weight * entropy(subset[target])

    return total_entropy - weighted_entropy


# STEP 4: Implement ID3 Algorithm

def id3(dataset, attributes, target):

    labels = dataset[target]

    # If all samples belong to one class
    if len(labels.unique()) == 1:
        return labels.iloc[0]

    # If no attributes remain, return majority class
    if len(attributes) == 0:
        return labels.mode()[0]

    # Find attribute with maximum information gain
    gains = {
        attribute: information_gain(dataset, attribute, target)
        for attribute in attributes
    }

    best_attribute = max(gains, key=gains.get)

    # Stop if no useful information gain
    if gains[best_attribute] <= 0:
        return labels.mode()[0]

    # Create decision node
    tree = {
        'attribute': best_attribute,
        'branches': {},
        'default': labels.mode()[0]
    }

    remaining_attributes = [
        attribute for attribute in attributes
        if attribute != best_attribute
    ]

    # Recursively construct branches
    for value in dataset[best_attribute].unique():

        subset = dataset[
            dataset[best_attribute] == value
        ]

        subtree = id3(
            subset,
            remaining_attributes,
            target
        )

        tree['branches'][value] = subtree

    return tree


# STEP 5: Build Decision Tree

attributes = [
    'Outlook',
    'Temperature',
    'Humidity',
    'Wind'
]

target = 'PlayTennis'

print("\n========== INFORMATION GAIN ==========")

for attribute in attributes:
    gain = information_gain(df, attribute, target)
    print(f"{attribute}: {gain:.4f}")

decision_tree = id3(df, attributes, target)


# STEP 6: Display Decision Tree

def display_tree(tree, indent=""):

    if not isinstance(tree, dict):
        print(indent + "-> " + str(tree))
        return

    attribute = tree['attribute']

    print(indent + "[" + attribute + "]")

    for value, subtree in tree['branches'].items():

        print(indent + "  |")

        print(indent + "  |-- " + str(value))

        display_tree(subtree, indent + "      ")


print("\n========== GENERATED ID3 DECISION TREE ==========")

display_tree(decision_tree)


# STEP 7: Classify a New Sample

def classify(tree, sample):

    # If leaf node, return classification
    if not isinstance(tree, dict):
        return tree

    attribute = tree['attribute']

    value = sample[attribute]

    # Follow corresponding branch
    if value in tree['branches']:
        return classify(tree['branches'][value], sample)

    # Handle unseen attribute values
    return tree['default']


new_sample = {
    'Outlook': 'Sunny',
    'Temperature': 'Cool',
    'Humidity': 'High',
    'Wind': 'Strong'
}

print("\n========== NEW SAMPLE ==========")

for attribute, value in new_sample.items():
    print(f"{attribute}: {value}")

prediction = classify(decision_tree, new_sample)

print("\n========== CLASSIFICATION RESULT ==========")
print("Predicted PlayTennis:", prediction)

if prediction == 'Yes':
    print("Decision: Play Tennis")
else:
    print("Decision: Do Not Play Tennis")

print("\nID3 Algorithm Executed Successfully!")