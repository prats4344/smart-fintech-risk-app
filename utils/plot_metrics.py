import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("german_credit.csv")
X = df.drop("Risk", axis=1)
y = df["Risk"]

# Encode categorical
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# Load model
with open("models/credit_model.pkl", "rb") as f:
    model = pickle.load(f)

# Predict
y_pred = model.predict(X)

# Confusion matrix
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Low", "High"], yticklabels=["Low", "High"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("static/confusion_matrix.png")
plt.show()
