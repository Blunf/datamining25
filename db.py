import os
import pandas as pd
import sqlite3

# -------------------------------------------------------
# 1) Check that data/ exists and list its contents
# -------------------------------------------------------
data_dir = "data"
if not os.path.isdir(data_dir):
    raise FileNotFoundError(f"Expected a folder named '{data_dir}/', but it does not exist.")

print(f"--- Files in '{data_dir}/' ---")
for fname in sorted(os.listdir(data_dir)):
    print(f"  • {fname}")
print("\nMake sure you see all six Telco CSVs above.\n")

# -------------------------------------------------------
# 2) Build accurate file paths (case‐sensitive!)
# -------------------------------------------------------
status_csv       = os.path.join(data_dir, "Telco_customer_churn_status.csv")
services_csv     = os.path.join(data_dir, "Telco_customer_churn_services.csv")
population_csv   = os.path.join(data_dir, "Telco_customer_churn_population.csv")  # read for completeness
location_csv     = os.path.join(data_dir, "Telco_customer_churn_location.csv")
demographics_csv = os.path.join(data_dir, "Telco_customer_churn_demographics.csv")
full_csv         = os.path.join(data_dir, "Telco_customer_churn.csv")

# -------------------------------------------------------
# 3) Read each CSV into pandas DataFrames
# -------------------------------------------------------
df_status       = pd.read_csv(status_csv)
df_services     = pd.read_csv(services_csv)
df_population   = pd.read_csv(population_csv)       # not merged below, but read for reference
df_location     = pd.read_csv(location_csv)
df_demographics = pd.read_csv(demographics_csv)
df_full         = pd.read_csv(full_csv)

# -------------------------------------------------------
# 4) From df_full, select the “core churn columns”
# -------------------------------------------------------
core_cols = [
    "CustomerID", "Gender", "Senior Citizen", "Partner", "Dependents",
    "Phone Service", "Multiple Lines", "Internet Service", "Online Security",
    "Online Backup", "Device Protection", "Tech Support", "Streaming TV",
    "Streaming Movies", "Contract", "Paperless Billing", "Payment Method",
    "Churn Label"
]
df_core = df_full[core_cols].copy()

# -------------------------------------------------------
# 5) Rename “Customer ID” → “CustomerID” in other DataFrames
# -------------------------------------------------------
df_status       = df_status.rename(columns={"Customer ID": "CustomerID"})
df_services     = df_services.rename(columns={"Customer ID": "CustomerID"})
df_location     = df_location.rename(columns={"Customer ID": "CustomerID"})
df_demographics = df_demographics.rename(columns={"Customer ID": "CustomerID"})

# -------------------------------------------------------
# 6) Merge extra fields
# -------------------------------------------------------
df_merged = (
    df_core
    # Demographics: “Age” and “Number of Dependents”
    .merge(
        df_demographics[["CustomerID", "Age", "Number of Dependents"]],
        on="CustomerID", how="left"
    )
    # Services: “Tenure in Months” and “Offer”
    .merge(
        df_services[["CustomerID", "Tenure in Months", "Offer"]],
        on="CustomerID", how="left"
    )
    # Location: “City”, “State”, “Zip Code”
    .merge(
        df_location[["CustomerID", "City", "State", "Zip Code"]],
        on="CustomerID", how="left"
    )
    # Status: “Satisfaction Score”, “CLTV”, “Churn Reason”
    .merge(
        df_status[["CustomerID", "Satisfaction Score", "CLTV", "Churn Reason"]],
        on="CustomerID", how="left"
    )
)

# -------------------------------------------------------
# 7) Create HasInternet = 1 if “Internet Service” != “No”, else 0
# -------------------------------------------------------
df_merged["HasInternet"] = (
    df_merged["Internet Service"]
    .astype(str)
    .str.strip()
    .str.lower()
    .ne("no")      # True if not exactly "no"
).astype(int)

# -------------------------------------------------------
# 8) Zero‐out any “No phone service” cells (any column) and force HasInternet=0
# -------------------------------------------------------
for col in df_merged.columns:
    mask_nops = (
        df_merged[col]
        .astype(str)
        .str.strip()
        .str.lower()
        == "no phone service"
    )
    if mask_nops.any():
        df_merged.loc[mask_nops, col] = 0
        df_merged.loc[mask_nops, "HasInternet"] = 0

# -------------------------------------------------------
# 9) Convert known binary “Yes”/“No” columns to 1/0 and 
#    internet sub‐features (“No internet service” → 0)
# -------------------------------------------------------

# 9a) Gender: “Male” → 1, “Female” → 0
df_merged["Gender"] = (
    df_merged["Gender"]
    .astype(str)
    .str.strip()
    .str.capitalize()
    .map({"Male": 1, "Female": 0})
)

# 9b) Single binary columns (exactly Yes/No → 1/0)
binary_yesno = [
    "Senior Citizen", "Partner", "Dependents",
    "Phone Service", "Paperless Billing", "Churn Label",
]

for col in binary_yesno:
    df_merged[col] = (
        df_merged[col]
        .astype(str)
        .str.strip()
        .str.capitalize()
        .map({"Yes": 1, "No": 0})
    )

# 9c) Internet‐related sub‐features: “Yes” → 1, “No” → 0, “No internet service” → 0
internet_sub = [
    "Online Security", "Online Backup", "Device Protection",
    "Tech Support", "Streaming TV", "Streaming Movies"
]

for col in internet_sub:
    df_merged[col] = (
        df_merged[col]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "yes": 1,
            "no": 0,
            "no internet service": 0
        })
    )

# 9d) Finally, convert “Multiple Lines” separately:
df_merged["Multiple Lines"] = (
    df_merged["Multiple Lines"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "yes": 1,
        "no": 0,
        "no phone service": 0
    })
)

df_merged["Multiple Lines"] = df_merged["Multiple Lines"].fillna(0).astype(int)

# -------------------------------------------------------
# 10) Drop columns you do not need:
#      CustomerID, raw Internet Service, Churn Reason, location fields
# -------------------------------------------------------
to_drop = [
    "CustomerID",
    "Internet Service",    # replaced by HasInternet
    "Churn Reason",        # text
    "City", "State",       # location
    "Zip Code"
]
df_merged = df_merged.drop(columns=[c for c in to_drop if c in df_merged.columns])

# -------------------------------------------------------
# 11) Rename columns to remove spaces (optional)
# -------------------------------------------------------
df_merged = df_merged.rename(columns={
    "Senior Citizen":       "SeniorCitizen",
    "Online Security":      "OnlineSecurity",
    "Online Backup":        "OnlineBackup",
    "Device Protection":    "DeviceProtection",
    "Tech Support":         "TechSupport",
    "Streaming TV":         "StreamingTV",
    "Streaming Movies":     "StreamingMovies",
    "Paperless Billing":    "PaperlessBilling",
    "Payment Method":       "PaymentMethod",
    "Churn Label":          "ChurnLabel",
    "Number of Dependents": "NumDependents",
    "Tenure in Months":     "TenureMonths",
    "Satisfaction Score":   "SatisfactionScore"
})

# -------------------------------------------------------
# 12) Save the final table as CSV and SQLite
# -------------------------------------------------------
os.makedirs(data_dir, exist_ok=True)

csv_out = os.path.join(data_dir, "combined_telco_churn_with_hasInternet.csv")
df_merged.to_csv(csv_out, index=False)
print(f"➡️  CSV saved to: {csv_out}")

db_out = os.path.join(data_dir, "combined_telco_churn_with_hasInternet.db")
conn = sqlite3.connect(db_out)
df_merged.to_sql("churn_data", conn, if_exists="replace", index=False)
conn.close()
print(f"➡️  SQLite DB saved to: {db_out}")

# -------------------------------------------------------
# 13) Display a preview to verify exact repr of values
# -------------------------------------------------------
print("\n--- First 5 rows of the cleaned, merged data ---")
print(df_merged.head(5))

print("\n--- Unique values in 'Multiple Lines' after mapping (with repr) ---")
unique_ml = df_merged["Multiple Lines"].dropna().unique()
print(sorted(repr(v) for v in unique_ml))

print("\n--- Unique values in 'HasInternet' (should be [0, 1]) ---")
print(sorted(df_merged["HasInternet"].dropna().unique()))

print("\n--- Unique values in 'OnlineSecurity' (with repr) ---")
unique_os = df_merged["OnlineSecurity"].dropna().unique()
print(sorted(repr(v) for v in unique_os))
