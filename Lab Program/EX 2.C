# ============================================================
# CANDIDATE-ELIMINATION ALGORITHM USING CSV
# Google Colab Implementation
# ============================================================

import csv
import itertools
import pandas as pd

# STEP 1: Create CSV Dataset Automatically

filename = "/content/training_data.csv"

dataset = [
    ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same", "Yes"],
    ["Sunny", "Warm", "High", "Strong", "Warm", "Same", "Yes"],
    ["Rainy", "Cold", "High", "Strong", "Warm", "Change", "No"],
    ["Sunny", "Warm", "High", "Strong", "Cool", "Change", "Yes"]
]

columns = [
    "Sky", "AirTemp", "Humidity",
    "Wind", "Water", "Forecast", "EnjoySport"
]

with open(filename, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(columns)
    writer.writerows(dataset)

print("CSV file created successfully!\n")

# STEP 2: Read CSV File

df = pd.read_csv(filename)

print("TRAINING DATASET")
print(df.to_string(index=False))

attributes = columns[:-1]
target = columns[-1]

# STEP 3: Generate Attribute Domains

domains = [
    sorted(df[attr].unique().tolist())
    for attr in attributes
]

# STEP 4: Define Hypothesis Functions

def covers(hypothesis, example):
    return all(
        h == "?" or h == value
        for h, value in zip(hypothesis, example)
    )

def more_general(h1, h2):
    return all(
        a == "?" or a == b
        for a, b in zip(h1, h2)
    )

# STEP 5: Generate All Possible Hypotheses

all_hypotheses = list(itertools.product(
    *[domain + ["?"] for domain in domains]
))

# STEP 6: Find All Consistent Hypotheses

positive_examples = df[df[target] == "Yes"][attributes].values.tolist()
negative_examples = df[df[target] == "No"][attributes].values.tolist()

version_space = []

for hypothesis in all_hypotheses:

    positive_correct = all(
        covers(hypothesis, example)
        for example in positive_examples
    )

    negative_correct = all(
        not covers(hypothesis, example)
        for example in negative_examples
    )

    if positive_correct and negative_correct:
        version_space.append(hypothesis)

# STEP 7: Identify Specific Boundary S
# Most specific consistent hypotheses

S = [
    h for h in version_space
    if not any(
        h != other and more_general(h, other)
        for other in version_space
    )
]

# STEP 8: Identify General Boundary G
# Most general consistent hypotheses

G = [
    h for h in version_space
    if not any(
        h != other and more_general(other, h)
        for other in version_space
    )
]

# STEP 9: Display Results

print("\n" + "=" * 65)
print("CANDIDATE-ELIMINATION ALGORITHM OUTPUT")
print("=" * 65)

print("\nSPECIFIC BOUNDARY (S):")
for h in S:
    print(h)

print("\nGENERAL BOUNDARY (G):")
for h in G:
    print(h)

print("\nVERSION SPACE - ALL CONSISTENT HYPOTHESES:")
for i, h in enumerate(version_space, start=1):
    print(f"Hypothesis {i}: {h}")

print("\nTotal Consistent Hypotheses:", len(version_space))

# STEP 10: Display Results as a Table

result_df = pd.DataFrame(
    version_space,
    columns=attributes
)

print("\nVERSION SPACE TABLE:")
display(result_df)