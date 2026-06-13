import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# ======================
# Load model + encoders
# ======================
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
le_name = joblib.load("le_name.pkl")
le_place = joblib.load("le_place.pkl")

# ======================
# Helper: Safe encoding
# ======================
def safe_transform(le, values):
    """Map unseen labels to -1 safely instead of raising ValueError"""
    known_classes = set(le.classes_)
    return [le.transform([v])[0] if v in known_classes else -1 for v in values]

# ======================
# Load & Score Data
# ======================
@st.cache_data
def load_and_score():
    df = pd.read_csv("transactions.csv")

    # Encode categorical features safely
    df["Name_enc"] = safe_transform(le_name, df["Name"])
    df["Place_enc"] = safe_transform(le_place, df["Place"])

    # Time feature
    df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour

    # Prepare features
    features = df[["Amount", "Name_enc", "Place_enc", "Hour"]]
    X = scaler.transform(features)

    # Get anomaly scores
    scores = model.decision_function(X)

    # Convert into 0–100 fraud scores
    fraud_scores = (1 - (scores - scores.min()) / (scores.max() - scores.min())) * 100
    df["Fraud_Score"] = fraud_scores.round(2)

    return df

# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="Bank Fraud Detection", layout="wide")

# ---------- Navbar ----------
st.markdown("""
    <style>
    .navbar {
        background-color: #102E50;
        padding: 12px;
        color: white;
        text-align: center;
        font-size: 30px;
        font-weight: bold;
        border-radius: 8px;
        margin-bottom: 20px;
        margin-top: -3px;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #102E50;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-radius: 8px 8px 0 0;
    }
    </style>
    <div class="navbar">🏦 Vanguard AI - Fraud Detection System</div>
""", unsafe_allow_html=True)

st.title("Statistics")

df = load_and_score()

# ---------- Global Search Bar ----------
global_search = st.text_input("🔍 Global Search (Name, Place, or Date)")
if global_search:
    df = df[df["Name"].str.contains(global_search, case=False) |
            df["Place"].str.contains(global_search, case=False) |
            df["Date"].astype(str).str.contains(global_search, case=False)]

# Navbar-style layout using tabs
tab1, tab2 = st.tabs(["📑 All Transactions", "🚨 Suspicious Transactions"])

# ---------- All Transactions ----------
with tab1:
    st.subheader("All Transactions")
    # Sort by Date & Time
    all_df = df.sort_values(by=["Date", "Time"], ascending=[False, False])
    st.dataframe(all_df, use_container_width=True)

# ---------- Suspicious ----------
with tab2:
    st.subheader("Fraudulent / Suspicious Transactions (Score > 80)")
    fraud_df = df[df["Fraud_Score"] > 80].sort_values(by="Fraud_Score", ascending=False)
    st.dataframe(fraud_df, use_container_width=True)

st.success("✅ Dashboard Loaded Successfully!")

# ---------- Footer ----------
st.markdown("""
    <div class="footer">
        © 2025 Vanguard AI | Fraud Detection Dashboard | Built with Streamlit
    </div>
""", unsafe_allow_html=True)
