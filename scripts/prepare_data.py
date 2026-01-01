import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------- CONFIG ----------------
INPUT_CSV = r"C:\Users\HP\Desktop\ROPDetectionAIML\ropfinal.csv"   # your existing CSV
OUTPUT_DIR = "data/splits"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42
# ----------------------------------------

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, \
    "Train/Val/Test ratios must sum to 1"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD ------------------
df = pd.read_csv(INPUT_CSV)

# Basic sanity checks
required_cols = {"Source", "ROP Label"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {required_cols}")

df = df.dropna(subset=["Source", "ROP Label"])

# Ensure labels are int 0/1
# Clean ROP Label column (handle strings like " 0", "1 ")
df["ROP Label"] = (
    df["ROP Label"]
    .astype(str)
    .str.strip()
)

# Validate allowed values before casting
if not set(df["ROP Label"].unique()).issubset({"0", "1"}):
    raise ValueError(
        f"Invalid ROP Label values found: {df['ROP Label'].unique()}"
    )

# Convert to integer
df["ROP Label"] = df["ROP Label"].astype(int)


# ---------------- SPLIT -----------------
# First split: train vs temp (val + test)
train_df, temp_df = train_test_split(
    df,
    test_size=(1.0 - TRAIN_RATIO),
    stratify=df["ROP Label"],
    random_state=RANDOM_STATE
)

# Second split: val vs test
val_size_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

val_df, test_df = train_test_split(
    temp_df,
    test_size=(1.0 - val_size_adjusted),
    stratify=temp_df["ROP Label"],
    random_state=RANDOM_STATE
)

# ---------------- SAVE ------------------
train_path = os.path.join(OUTPUT_DIR, "train.csv")
val_path = os.path.join(OUTPUT_DIR, "val.csv")
test_path = os.path.join(OUTPUT_DIR, "test.csv")

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

# ---------------- REPORT ----------------
def report_split(name, df_split):
    counts = df_split["ROP Label"].value_counts().to_dict()
    total = len(df_split)
    print(f"{name}: {total} samples | Class distribution: {counts}")

print("✅ Dataset split completed\n")
report_split("Train", train_df)
report_split("Validation", val_df)
report_split("Test", test_df)

print("\nSaved files:")
print(train_path)
print(val_path)
print(test_path)
