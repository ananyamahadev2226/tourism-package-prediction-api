import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "tourism_project" / "data" / "tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups",
    "ProductPitched", "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
    "Designation", "MonthlyIncome",
]

if not DATA_PATH.exists():
    sys.exit(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
if missing:
    sys.exit(f"Validation FAILED - missing columns: {missing}")
if extra:
    print(f"Note: extra columns (handled in data prep): {extra}")

print("Validation PASSED - all expected columns are present")
print("\nShape:", df.shape)
print("\nColumn dtypes:\n", df.dtypes.to_string())
missing_vals = df.isna().sum()
missing_vals = missing_vals[missing_vals > 0]
print("\nMissing values:", "none" if missing_vals.empty else "\n" + missing_vals.to_string())
print("\nDuplicate rows:", int(df.duplicated().sum()))
print("\nTarget (ProdTaken) distribution:\n", df["ProdTaken"].value_counts(normalize=True).round(3).to_string())
print("\nNumeric summary:\n", df.describe().T.round(2).to_string())
