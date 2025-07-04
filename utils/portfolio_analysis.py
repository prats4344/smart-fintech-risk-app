import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_portfolio(file_path):
    df = pd.read_csv(file_path)

    # Count of risk classes
    risk_counts = df['Predicted Risk'].value_counts()

    # Pie chart
    plt.figure(figsize=(6,6))
    risk_counts.plot.pie(autopct='%1.1f%%', colors=['green', 'red'])
    plt.title("Risk Distribution")
    plt.ylabel("")
    plt.savefig("static/risk_pie_chart.png")
    plt.close()

    # Credit Amount vs Risk
    plt.figure(figsize=(8,5))
    sns.boxplot(x='Predicted Risk', y='Credit Amount', data=df, palette='Set2')
    plt.title("Credit Amount by Risk Level")
    plt.savefig("static/credit_vs_risk.png")
    plt.close()

    return {
        "total": len(df),
        "low_risk": (df['Predicted Risk'] == 'Low Risk ✅').sum(),
        "high_risk": (df['Predicted Risk'] == 'High Risk ❌').sum()
    }
