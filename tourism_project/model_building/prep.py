import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "tourism_project" / "data" / "tourism.csv"

# Load the dataset
df = pd.read_csv(DATA_PATH)

# 1. Drop unnecessary columns
#    'Unnamed: 0' = leftover index, 'CustomerID' = identifier (no predictive value)
df = df.drop(columns=["Unnamed: 0", "CustomerID"], errors="ignore")

# 2. Gender: fix the typo 'Fe Male' -> 'Female'
gender_key = (df["Gender"].str.strip().str.lower().str.replace(r"\s+", "", regex=True))
df["Gender"] = gender_key.map({"male": "Male", "female": "Female"})

# 3. MaritalStatus: data dictionary only defines Single / Married / Divorced,
#    so merge 'Unmarried' into 'Single'
df["MaritalStatus"] = (df["MaritalStatus"].str.strip().str.title()
                       .replace({"Unmarried": "Single"}))

print("\nGender counts after cleaning:\n", df["Gender"].value_counts())
print("\nMaritalStatus counts after cleaning:\n", df["MaritalStatus"].value_counts())
missing = df.isna().sum()
missing = missing[missing > 0]
print("\nMissing values per column:", "none" if missing.empty else "\n" + missing.to_string())

# 5. Train / test split (stratified because only ~19% of customers buy)
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Save locally (the workflow uploads these four files as an artifact)
X_train.to_csv(ROOT / "Xtrain.csv", index=False)
X_test.to_csv(ROOT / "Xtest.csv", index=False)
y_train.to_csv(ROOT / "ytrain.csv", index=False)
y_test.to_csv(ROOT / "ytest.csv", index=False)

print(f"\nXtrain {X_train.shape} | Xtest {X_test.shape}")
print("Buyer rate  train: %.3f  test: %.3f" % (y_train.mean(), y_test.mean()))
print("Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")
