# project.py
# IRIS SPECIES CLASSIFICATION PROJECT

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

print("\n=== IRIS SPECIES CLASSIFICATION PROJECT ===\n")

# 1. Load dataset
df = pd.read_csv("Iris.csv")
print("Dataset Loaded Successfully!")
print(df.head())

# 2. Basic EDA
print("\nDataset Info:")
print(df.info())

print("\nClass Distribution:")
print(df["Species"].value_counts())

# Pairplot
sns.pairplot(df, hue="Species")
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(7,5))
sns.heatmap(df.drop("Species", axis=1).corr(), annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.show()

# 3. Preprocess
X = df.drop("Species", axis=1)
y = df["Species"]

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "KNN Classifier": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}

results = {}

print("\n=== MODEL PERFORMANCE ===")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

# 5. Compare models
plt.figure(figsize=(7,4))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title("Model Comparison (Accuracy)")
plt.ylabel("Accuracy")
plt.show()

best_model = max(results, key=results.get)
print(f"\nBest model: {best_model} (Accuracy: {results[best_model]:.4f})")
