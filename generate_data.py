import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Parameters
n_samples = 100000
names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Helen"]
places = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Hyderabad", "Kolkata", "Pune", "Jaipur"]
base_date = datetime(2025, 1, 1)

data = []
for i in range(n_samples):
    name = random.choice(names)
    place = random.choice(places)
    amount = np.random.randint(100, 100000)
    days_offset = np.random.randint(0, 365)
    transaction_date = base_date + timedelta(days=days_offset)
    time_offset = timedelta(
        hours=np.random.randint(0, 24),
        minutes=np.random.randint(0, 60),
        seconds=np.random.randint(0, 60)
    )
    transaction_time = (transaction_date + time_offset).time()
    data.append({
        "TransactionID": f"T{i+1:06d}",
        "Name": name,
        "Place": place,
        "Amount": amount,
        "Date": transaction_date.date(),
        "Time": transaction_time
    })

df = pd.DataFrame(data)
df.to_csv("transactions.csv", index=False)

print("transactions.csv created with", n_samples, "rows")
