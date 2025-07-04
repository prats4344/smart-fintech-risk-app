from flask import Flask, render_template, request, send_file
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load model and feature names
with open("models/credit_model.pkl", "rb") as f:
    model, model_columns = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Collect form inputs
            age = int(request.form['age'])
            job = int(request.form['job'])
            housing = request.form['housing']
            saving_accounts = request.form['saving_accounts']
            checking_account = request.form['checking_account']
            credit_amount = int(request.form['credit_amount'])
            duration = int(request.form['duration'])
            purpose = request.form['purpose']

            # Prepare input for model
            input_dict = {
                "Age": age,
                "Job": job,
                "Housing": housing,
                "Saving accounts": saving_accounts,
                "Checking account": checking_account,
                "Credit amount": credit_amount,
                "Duration": duration,
                "Purpose": purpose
            }

            input_df = pd.DataFrame([input_dict])
            input_df = pd.get_dummies(input_df)

            # Match model columns
            for col in model_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[model_columns]

            prediction = model.predict(input_df)[0]
            result = "High Risk ❌" if prediction == 1 else "Low Risk ✅"

            return render_template("result.html", prediction=result)

        except Exception as e:
            return f"An error occurred: {e}"

    return render_template("form.html")

@app.route("/metrics")
def metrics():
    # Load classification report
    try:
        with open("static/metrics.txt", "r") as f:
            metrics_text = f.read()
    except:
        metrics_text = "Classification report not found."

    return render_template("metrics.html", metrics_text=metrics_text)

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        try:
            file = request.files['file']
            if not os.path.exists("uploads"):
                os.makedirs("uploads")
            filepath = os.path.join("uploads", "portfolio.csv")
            file.save(filepath)

            # Load and preprocess portfolio data
            df = pd.read_csv(filepath)
            df_input = df.copy()
            df_input = pd.get_dummies(df_input)

            for col in model_columns:
                if col not in df_input.columns:
                    df_input[col] = 0
            df_input = df_input[model_columns]

            # Predict risk
            predictions = model.predict(df_input)
            df["Predicted Risk"] = ["High Risk ❌" if p == 1 else "Low Risk ✅" for p in predictions]

            high_risk = sum(predictions)
            low_risk = len(predictions) - high_risk

            # Save results
            if not os.path.exists("results"):
                os.makedirs("results")
            df.to_csv("results/portfolio_results.csv", index=False)

            return render_template("upload.html", message="✅ File processed successfully!",
                                   high_risk=high_risk, low_risk=low_risk)

        except Exception as e:
            return f"An error occurred: {e}"

    return render_template("upload.html")

@app.route("/download")
def download():
    path = "results/portfolio_results.csv"
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    else:
        return "Result file not found. Please upload portfolio first."

# ✅ This part is updated for Render deployment compatibility
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
