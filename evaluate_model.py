import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# -------------------------------------------------
# Load scored transactions
# -------------------------------------------------
df = pd.read_csv("scored_transactions.csv")

# -------------------------------------------------
# Ensure ground truth labels exist
# -------------------------------------------------
if "is_fraud" not in df.columns:
    print("⚠️ No fraud label column found. Generating dummy labels for testing...")
    # Generate dummy labels: 5% fraud probability
    df["is_fraud"] = np.random.choice([0, 1], size=len(df), p=[0.95, 0.05])
    df.to_csv("scored_transactions.csv", index=False)
    print("✅ Dummy labels added to scored_transactions.csv")

# -------------------------------------------------
# Define fraud prediction (threshold on Fraud_Score)
# -------------------------------------------------
# Example: Fraud if Fraud_Score >= 50
df["predicted"] = (df["Fraud_Score"] >= 50).astype(int)

# -------------------------------------------------
# Calculate metrics
# -------------------------------------------------
y_true = df["is_fraud"]
y_pred = df["predicted"]

acc  = accuracy_score(y_true, y_pred) * 100
prec = precision_score(y_true, y_pred, zero_division=0) * 100
rec  = recall_score(y_true, y_pred, zero_division=0) * 100
f1   = f1_score(y_true, y_pred, zero_division=0) * 100

print("\n📊 Model Evaluation Metrics:")
print(f"Accuracy:  {acc:.2f}%")
print(f"Precision: {prec:.2f}%")
print(f"Recall:    {rec:.2f}%")
print(f"F1 Score:  {f1:.2f}%")

# Save updated CSV with predictions
df.to_csv("transactions_with_predictions.csv", index=False)
print("✅ Predictions saved to transactions_with_predictions.csv")
