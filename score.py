from train_model import score_transactions, train_model
import os

if not os.path.exists("fraud_model.pkl"):
    print("⚠️ No model found. Training first...")
    train_model("transactions.csv")

df = score_transactions("transactions.csv")
print(df.head(10))

df.to_csv("scored_transactions.csv", index=False)
print("✅ scored_transactions.csv created with fraud scores")
