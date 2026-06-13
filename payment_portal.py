import streamlit as st
import pandas as pd
import datetime
import joblib
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Vanguard Bank - Payment Portal", layout="centered")

# ======================
# Load Model + Encoders
# ======================
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
le_name = joblib.load("le_name.pkl")
le_place = joblib.load("le_place.pkl")

# ======================
# Navbar
# ======================
st.markdown("""
    <nav style="background-color:#102E50; padding:10px; border-radius:8px;">
        <h3 style="color:white; text-align:center;">💳 Vanguard Bank | Payment Portal</h3>
    </nav>
""", unsafe_allow_html=True)

st.write("### Make a Payment")

# ======================
# Input Form
# ======================
with st.form("payment_form"):
    name = st.text_input("Name")
    place = st.text_input("Place")
    time = st.time_input("Transaction Time", datetime.datetime.now().time())
    amount = st.number_input("Amount", min_value=1.0, step=0.5)

    submitted = st.form_submit_button("Pay Now")

if submitted:
    # Generate today's date
    today = datetime.date.today().strftime("%Y-%m-%d")

    # Create new transaction row
    new_txn = pd.DataFrame([{
        "Date": today,
        "Time": time.strftime("%H:%M:%S"),
        "Name": name,
        "Place": place,
        "Amount": amount
    }])

    # Append to CSV
    try:
        df = pd.read_csv("transactions.csv")
        df = pd.concat([df, new_txn], ignore_index=True)
    except FileNotFoundError:
        df = new_txn

    # ======================
    # Fraud Scoring
    # ======================
# ======================
# Fraud Scoring (Safe for unseen labels)
# ======================

# Handle unseen labels in Name
    df["Name_enc"] = df["Name"].apply(lambda x: le_name.transform([x])[0] if x in le_name.classes_ else -1)

    # Handle unseen labels in Place
    df["Place_enc"] = df["Place"].apply(lambda x: le_place.transform([x])[0] if x in le_place.classes_ else -1)

    # Extract Hour
    df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour

    # Scale + Predict
    features = df[["Amount", "Name_enc", "Place_enc", "Hour"]]
    X = scaler.transform(features)
    scores = model.decision_function(X)
    fraud_scores = (1 - (scores - scores.min()) / (scores.max() - scores.min())) * 100
    df["Fraud_Score"] = fraud_scores.round(2)


    # Save updated transactions with Fraud Score
    df.to_csv("transactions.csv", index=False)

    st.success(f"✅ Payment Successful! Fraud score calculated for {name}.")

# ======================
# Footer
# ======================
st.markdown("""
    <footer style="background-color:#102E50; padding:10px; border-radius:8px; margin-top:20px;">
        <p style="color:white; text-align:center;">© Vanguard Bank</p>
    </footer>
""", unsafe_allow_html=True)
