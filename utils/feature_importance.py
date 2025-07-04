import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and column names
with open("models/credit_model.pkl", "rb") as f:
    model, model_columns = pickle.load(f)

# Get feature importance (coefficients)
importance = model.coef_[0]
features_df = pd.DataFrame({"Feature": model_columns, "Importance": importance})

# Sort by absolute importance
features_df["Abs_Importance"] = features_df["Importance"].abs()
features_df = features_df.sort_values(by="Abs_Importance", ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=features_df, palette="viridis")
plt.title("Feature Importance for Credit Risk Model")
plt.tight_layout()
plt.savefig("static/feature_importance.png")
print("✅ Feature importance plot saved as static/feature_importance.png")
