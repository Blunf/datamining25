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
print("\nIf any Telco CSV is missing or misspelled, fix that before continuing.\n")

# -------------------------------------------------------
# 2) Build accurate file paths (case‐sensitive!)
# -------------------------------------------------------
status_csv       = os.path.join(data_dir, "Telco_customer_churn_status.csv")
services_csv     = os.path.join(data_dir, "Telco_customer_churn_services.csv")
population_csv   = os.path.join(data_dir, "Telco_customer_churn_population.csv")  # not used below but read for completeness
location_csv     = os.path.join(data_dir, "Telco_customer_churn_location.csv")
demographics_csv = os.path.join(data_dir, "Telco_customer_churn_demographics.csv")
full_csv         = os.path.join(data_dir, "Telco_customer_churn.csv")

# -------------------------------------------------------
# 3) Read each CSV into a pandas DataFrame
# -------------------------------------------------------
df_status       = pd.read_csv(status_csv)
df_services     = pd.read_csv(services_csv)
df_population   = pd.read_csv(population_csv)       # (we won’t merge this one, but we read it anyway)
df_location     = pd.read_csv(location_csv)
df_demographics = pd.read_csv(demographics_csv)
df_full         = pd.read_csv(full_csv)

# -------------------------------------------------------
# 4) From df_full, select the “core churn columns” you originally listed
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
# 5) Rename “Customer ID” → “CustomerID” in the other DataFrames for consistent merging
# -------------------------------------------------------
df_status       = df_status.rename(columns={"Customer ID": "CustomerID"})
df_services     = df_services.rename(columns={"Customer ID": "CustomerID"})
df_location     = df_location.rename(columns={"Customer ID": "CustomerID"})
df_demographics = df_demographics.rename(columns={"Customer ID": "CustomerID"})

# -------------------------------------------------------
# 6) Merge extra fields as needed (example fields shown below)
#    – You can adjust which columns you pull from each DataFrame.
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
# 7) Drop the CustomerID column so it isn't used as a feature
# -------------------------------------------------------
df_merged = df_merged.drop(columns=["CustomerID"])

# -------------------------------------------------------
# 8) Convert only the strictly binary “Yes”/“No” columns into 1/0.
#    If a column has more than 2 unique values (e.g., “Multiple Lines”),
#    leave it as text.
# -------------------------------------------------------

# We'll iterate over every column in df_merged and check its unique non-null values.
for col in df_merged.columns:
    # Gather unique non-null string representations
    uniques = set(df_merged[col].dropna().astype(str).str.strip().str.capitalize())
    # If exactly {"Yes", "No"}, convert to 1/0:
    if uniques == {"Yes", "No"}:
        # First, standardize the column to capitalized “Yes”/“No”
        df_merged[col] = (
            df_merged[col]
            .astype(str)
            .str.strip()
            .str.capitalize()
        ).map({"Yes": 1, "No": 0})
    # Otherwise, leave as is (even if it’s text with more categories)

# -------------------------------------------------------
# 9) (Optional) Rename columns to remove spaces so they become easier to reference
# -------------------------------------------------------
df_merged = df_merged.rename(columns={
    "Senior Citizen":        "SeniorCitizen",
    "Internet Service":      "InternetService",
    "Online Security":       "OnlineSecurity",
    "Online Backup":         "OnlineBackup",
    "Device Protection":     "DeviceProtection",
    "Tech Support":          "TechSupport",
    "Streaming TV":          "StreamingTV",
    "Streaming Movies":      "StreamingMovies",
    "Paperless Billing":     "PaperlessBilling",
    "Payment Method":        "PaymentMethod",
    "Churn Label":           "ChurnLabel",
    "Number of Dependents":  "NumDependents",
    "Tenure in Months":      "TenureMonths",
    "Satisfaction Score":    "SatisfactionScore"
})

# -------------------------------------------------------
# 10) Save the final, cleaned DataFrame back to data/
# -------------------------------------------------------
os.makedirs(data_dir, exist_ok=True)

# 10a) CSV output
csv_out = os.path.join(data_dir, "combined_telco_churn_cleaned.csv")
df_merged.to_csv(csv_out, index=False)
print(f"➡️  CSV saved to: {csv_out}")

# 10b) SQLite output
db_out = os.path.join(data_dir, "combined_telco_churn_cleaned.db")
conn = sqlite3.connect(db_out)
df_merged.to_sql("churn_data", conn, if_exists="replace", index=False)
conn.close()
print(f"➡️  SQLite DB saved to: {db_out}")

# -------------------------------------------------------
# 11) Display a preview so you can verify that only strictly binary columns became 1/0,
#     and multi-valued columns (like “Multiple Lines”) remain as text.
# -------------------------------------------------------
print("\n--- First 5 rows of the cleaned, merged data ---")
print(df_merged.head(5))

print("\n--- Unique values in ‘Multiple Lines’ column  ---")
if "Multiple Lines" in df_merged.columns:
    print(sorted(df_merged["Multiple Lines"].dropna().unique()))
else:
    print("Column 'Multiple Lines' not present.")
