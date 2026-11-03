import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target,
    test_size=0.2, random_state=42, stratify=data.target
)

models = {
    'Logistic Regression': make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=3000)
    ),
    'KNN': make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=5)
    ),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, random_state=42
    )
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    results.append({
        'Algorithm': name,
        'Accuracy': round(accuracy_score(y_test, pred), 4),
        'Precision': round(precision_score(y_test, pred), 4),
        'Recall': round(recall_score(y_test, pred), 4),
        'F1 Score': round(f1_score(y_test, pred), 4)
    })

result_df = pd.DataFrame(results)
print(result_df.to_string(index=False))

best = result_df.loc[result_df['F1 Score'].idxmax()]
print("\nBest by F1 Score:", best['Algorithm'])