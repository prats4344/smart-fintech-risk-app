import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json

# Load dataset
df = pd.read_csv("german_credit.csv")

# Separate features and target
X = df.drop("Risk", axis=1)
y = df["Risk"]

# Encode all categorical columns
label_encoders = {}
for column in X.columns:
    if X[column].dtype == 'object':
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column].astype(str))
        label_encoders[column] = le

# Load model
with open("models/credit_model.pkl", "rb") as f:
    model = pickle.load(f)

# Predict
y_pred = model.predict(X)

# Evaluation
acc = accuracy_score(y, y_pred)
cm = confusion_matrix(y, y_pred)
report = classification_report(y, y_pred)

# Print to terminal
print("\n✅ Accuracy Score:", acc)
print("\n✅ Confusion Matrix:\n", cm)
print("\n✅ Classification Report:\n", report)

# Save metrics to JSON
metrics_data = {
    "accuracy": acc,
    "classification_report": report
}

with open("static/metrics.json", "w") as f:
    json.dump(metrics_data, f)

print("\n✅ Metrics saved to static/metrics.json")

