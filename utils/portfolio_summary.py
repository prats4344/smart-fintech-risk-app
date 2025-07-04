import pandas as pd
import matplotlib.pyplot as plt

def generate_portfolio_pie_chart(csv_path="results/portfolio_results.csv", output_path="static/risk_pie_chart.png"):
    df = pd.read_csv(csv_path)

    # Count High and Low Risk
    risk_counts = df['Predicted Risk'].value_counts()

    # Plot Pie Chart
    plt.figure(figsize=(6,6))
    plt.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', startangle=140, colors=["#e74c3c", "#2ecc71"])
    plt.title("Portfolio Risk Distribution")

    # Save to static
    plt.savefig(output_path)
    plt.close()
