import pickle
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("german_credit.csv")
X = df.drop("Risk", axis=1)
y = df["Risk"]

# One-hot encoding
X = pd.get_dummies(X)

# ✅ Load model and feature names
with open("models/credit_model.pkl", "rb") as f:
    model, feature_names = pickle.load(f)  # Unpack the tuple!

# ✅ Align features to match training
for col in feature_names:
    if col not in X.columns:
        X[col] = 0
X = X[feature_names]  # Reorder columns

# Split into test set
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Predict
y_pred = model.predict(X_test)

# Save metrics to text file
with open("static/metrics.txt", "w", encoding="utf-8") as f:
    f.write(f"Accuracy Score: {accuracy_score(y_test, y_pred):.2f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred)) + "\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))

print("✅ Metrics saved to static/metrics.txt successfully.")
