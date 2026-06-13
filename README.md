# Fraud Detection System

An end-to-end anomaly detection system that scores financial transactions for fraud risk using an unsupervised machine learning model, with a real-time Streamlit dashboard and payment portal.

## Architecture

```
generate_data.py       Synthetic transaction generation (100K rows)
train_model.py         Model training + scoring pipeline
score.py               Standalone scoring script
app.py                 Streamlit fraud dashboard
payment_portal.py      Streamlit payment entry with real-time scoring
evaluate_model.py      Model performance evaluation
```

## Model

Uses **Isolation Forest** (`sklearn.ensemble`) trained on 4 features:
- Amount
- Name (label-encoded)
- Place (label-encoded)
- Hour (extracted from transaction time)

Fraud scores are normalized to a 0-100 scale (higher = riskier).

## Quick Start

```bash
pip install pandas numpy scikit-learn joblib streamlit
python generate_data.py
python train_model.py
streamlit run app.py
```

## Files

| File | Purpose |
|------|---------|
| `generate_data.py` | Generates 100K synthetic transactions |
| `train_model.py` | Trains Isolation Forest, saves model + encoders + scaler |
| `score.py` | Scores existing transactions using saved model |
| `app.py` | Dashboard with all transactions and suspicious filter (score > 80) |
| `payment_portal.py` | Form that appends transactions and scores them in real time |
| `evaluate_model.py` | Evaluates accuracy, precision, recall, F1 at threshold >= 50 |

## Artifacts

- `fraud_model.pkl` - Trained Isolation Forest model
- `scaler.pkl` - Fitted StandardScaler
- `le_name.pkl`, `le_place.pkl` - Label encoders
- `transactions.csv` - Raw transaction data
- `scored_transactions.csv` - Data with computed fraud scores
- `transactions_with_predictions.csv` - Data with fraud predictions
