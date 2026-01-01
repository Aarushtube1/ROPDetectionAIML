from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# CONFIG (EDIT ONLY THIS SECTION IF NEEDED)
# --------------------------------------------------

# Absolute path to your cleaned CSV
INPUT_CSV = Path(r"C:\Users\HP\Desktop\ROPDetectionAIML\ropfinal.csv")

# Train / Val / Test ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42

# --------------------------------------------------
# PATH HANDLING (DO NOT EDIT)
# --------------------------------------------------

# Project root = parent of scripts/ (robust)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# VALIDATE RATIOS
# --------------------------------------------------

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, (
    "Train / Val / Test ratios must sum to 1"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_CSV)

# Required columns check
required_cols = {"Source", "ROP Label"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {required_cols}")

# Drop rows with missing values
df = df.dropna(subset=["Source", "ROP Label"])

# --------------------------------------------------
# CLEAN & VALIDATE LABELS
# --------------------------------------------------

# Normalize labels: handle " 0", "1 ", "-1", etc.
df["ROP Label"] = (
    df["ROP Label"]
    .astype(str)
    .str.strip()
)

# Keep only valid binary labels
valid_labels = {"0", "1"}
invalid_mask = ~df["ROP Label"].isin(valid_labels)

if invalid_mask.any():
    print(
        f"⚠️ Dropping {invalid_mask.sum()} rows with invalid ROP labels: "
        f"{df.loc[invalid_mask, 'ROP Label'].unique()}"
    )

df = df[~invalid_mask]

# Convert to integer
df["ROP Label"] = df["ROP Label"].astype(int)

# --------------------------------------------------
# STRATIFIED SPLIT
# --------------------------------------------------

# Train vs (Val + Test)
train_df, temp_df = train_test_split(
    df,
    test_size=(1.0 - TRAIN_RATIO),
    stratify=df["ROP Label"],
    random_state=RANDOM_STATE
)

# Val vs Test
val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

val_df, test_df = train_test_split(
    temp_df,
    test_size=(1.0 - val_ratio_adjusted),
    stratify=temp_df["ROP Label"],
    random_state=RANDOM_STATE
)

# --------------------------------------------------
# SAVE FILES (ABSOLUTE PATHS)
# --------------------------------------------------

train_path = OUTPUT_DIR / "train.csv"
val_path = OUTPUT_DIR / "val.csv"
test_path = OUTPUT_DIR / "test.csv"

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

# --------------------------------------------------
# REPORT
# --------------------------------------------------

def report_split(name, df_split):
    counts = df_split["ROP Label"].value_counts().to_dict()
    total = len(df_split)
    print(f"{name}: {total} samples | Class distribution: {counts}")

print("\n✅ Dataset split completed\n")
report_split("Train", train_df)
report_split("Validation", val_df)
report_split("Test", test_df)

print("\n📁 Saved files:")
print(train_path)
print(val_path)
print(test_path)
