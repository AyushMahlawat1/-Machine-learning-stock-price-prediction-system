import pandas as pd

# Load and parse dates
df = pd.read_csv("google-stock-price-csv.txt", parse_dates=["Date"])

# Split into train/test (time-based)
train_df = df[df["Date"] < "2024-01-01"]
test_df = df[df["Date"] >= "2024-01-01"]

# Save to CSV
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)