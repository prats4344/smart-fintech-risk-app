import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv("german_credit.csv")

# Separate features and target
X = df.drop("Risk", axis=1)
y = df["Risk"]

# One-hot encoding for categorical features
X_encoded = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save model AND columns used
with open("models/credit_model.pkl", "wb") as f:
    pickle.dump((model, X_encoded.columns.tolist()), f)

print("✅ Model retrained and saved successfully with feature names!")

