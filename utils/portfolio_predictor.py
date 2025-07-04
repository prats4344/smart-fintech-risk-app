import pandas as pd
import pickle

def predict_risk_for_portfolio(file_path):
    # Load model and feature names
    with open("models/credit_model.pkl", "rb") as f:
        model, model_columns = pickle.load(f)

    # Read CSV
    df = pd.read_csv(file_path)

    # Keep original for saving results later
    original_df = df.copy()

    # One-hot encoding
    df = pd.get_dummies(df)

    # Align columns with training columns
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[model_columns]

    # Predict
    predictions = model.predict(df)
    original_df["Predicted Risk"] = ["High Risk ❌" if p == 1 else "Low Risk ✅" for p in predictions]

    # Save results
    output_path = "results/portfolio_results.csv"
    original_df.to_csv(output_path, index=False)

    # Summary
    high_risk = sum(predictions)
    low_risk = len(predictions) - high_risk

    return high_risk, low_risk, output_path

