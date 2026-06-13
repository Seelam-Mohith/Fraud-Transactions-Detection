import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib


# -------------------------------------------------
# Function to train the model and save artifacts
# -------------------------------------------------
def train_model(csv_path="transactions.csv"):
    # Load data
    df = pd.read_csv(csv_path)

    # Encode categorical features
    le_name = LabelEncoder()
    le_place = LabelEncoder()

    df["Name_enc"] = le_name.fit_transform(df["Name"])
    df["Place_enc"] = le_place.fit_transform(df["Place"])

    # Extract hour from transaction time
    df["Hour"] = pd.to_datetime(df["Time"], errors="coerce").dt.hour.fillna(0).astype(int)

    # Select features
    features = df[["Amount", "Name_enc", "Place_enc", "Hour"]]

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # Train Isolation Forest
    model = IsolationForest(
        n_estimators=100,
        contamination=0.01,   # ~1% assumed fraud
        random_state=42
    )
    model.fit(X)

    # Save model + encoders + scaler
    joblib.dump(model, "fraud_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(le_name, "le_name.pkl")
    joblib.dump(le_place, "le_place.pkl")

    print("✅ Model training complete. Model and preprocessors saved.")


# -------------------------------------------------
# Helper function for safe label encoding
# -------------------------------------------------
def safe_transform(le, values):
    """Transforms labels with LabelEncoder, assigns -1 for unseen labels."""
    known_classes = set(le.classes_)
    transformed = []
    for v in values:
        if v in known_classes:
            transformed.append(le.transform([v])[0])
        else:
            transformed.append(-1)  # unseen label -> assign -1
    return transformed


# -------------------------------------------------
# Function to score transactions using saved model
# -------------------------------------------------
def score_transactions(csv_path="transactions.csv"):
    # Load model + preprocessors
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    le_name = joblib.load("le_name.pkl")
    le_place = joblib.load("le_place.pkl")

    # Load data
    df = pd.read_csv(csv_path)

    # Apply safe encoding (handles unseen labels)
    df["Name_enc"] = safe_transform(le_name, df["Name"])
    df["Place_enc"] = safe_transform(le_place, df["Place"])
    df["Hour"] = pd.to_datetime(df["Time"], errors="coerce").dt.hour.fillna(0).astype(int)

    # Scale features
    features = df[["Amount", "Name_enc", "Place_enc", "Hour"]]
    X = scaler.transform(features)

    # Get anomaly scores
    scores = model.decision_function(X)

    # Normalize to 0–100 fraud risk scale (higher = riskier)
    fraud_scores = (1 - (scores - scores.min()) / (scores.max() - scores.min())) * 100

    # Attach fraud score
    df["Fraud_Score"] = fraud_scores.round(2)

    # ✅ Save scored file automatically for later evaluation
    df.to_csv("scored_transactions.csv", index=False)

    return df


# -------------------------------------------------
# Script entry point (if run directly)
# -------------------------------------------------
if __name__ == "__main__":
    # Train model
    train_model("transactions.csv")

    # Score transactions right after training
    scored_df = score_transactions("transactions.csv")

    print("✅ scored_transactions.csv created with fraud scores")
