# FIND-S Algorithm Implementation

# Training dataset
# Attributes: Sky, AirTemp, Humidity, Wind, Water, Forecast
# Class: EnjoySport (Yes/No)

training_data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

# Number of attributes
num_attributes = len(training_data[0]) - 1

# Initialize most specific hypothesis
hypothesis = ['0'] * num_attributes

print("FIND-S Algorithm")
print("-" * 50)

print("\nTraining Data:")
for row in training_data:
    print(row)

print("\nInitial Hypothesis:", hypothesis)

# FIND-S Algorithm
for example in training_data:

    # Process only positive examples
    if example[-1] == 'Yes':

        print("\nPositive Example:", example[:-1])

        # First positive example
        if hypothesis[0] == '0':
            hypothesis = example[:-1].copy()

        else:
            # Compare each attribute
            for i in range(num_attributes):

                if hypothesis[i] != example[i]:
                    hypothesis[i] = '?'

        print("Updated Hypothesis:", hypothesis)

    else:
        print("\nNegative Example Ignored:", example[:-1])

# Display final hypothesis
print("\n" + "=" * 50)
print("FINAL MOST SPECIFIC HYPOTHESIS")
print("=" * 50)

print("Hypothesis:", hypothesis)

# Display hypothesis with attribute names
attributes = [
    'Sky', 'AirTemp', 'Humidity',
    'Wind', 'Water', 'Forecast'
]

print("\nFinal Hypothesis Details:")
for attribute, value in zip(attributes, hypothesis):
    print(attribute, ":", value)