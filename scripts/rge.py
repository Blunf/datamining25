import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1) Load data
df = pd.read_csv("../data/combined_telco_churn_with_hasInternet.csv")

# 2) Convert all Yes/No/No phone service/No internet service → 0/1
binary_map = {"yes": 1, "no": 0, "no phone service": 0, "no internet service": 0}
binary_cols = [
    "SeniorCitizen", "Partner", "NumDependents", "Phone Service", "Multiple Lines",
    "Online Security", "Online Backup", "Device Protection", "Tech Support",
    "Streaming TV", "Streaming Movies", "Paperless Billing", "HasInternet"
]
for col in binary_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str).str.strip().str.lower()
            .map(binary_map)
            .fillna(0)
            .astype(int)
        )

# 3) Convert Gender → 0/1 if needed
if "Gender" in df.columns:
    if df["Gender"].dtype == object:
        df["Gender"] = (
            df["Gender"]
            .str.strip().str.capitalize()
            .map({"Male": 1, "Female": 0})
            .fillna(0)
            .astype(int)
        )
    else:
        df["Gender"] = df["Gender"].fillna(0).astype(int)

# 4) One‐hot encode any remaining object columns (drop the first dummy to avoid collinearities)
object_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != "ChurnLabel"]
df = pd.get_dummies(df, columns=object_cols, drop_first=True)

# 5) Split X/y
y = df["ChurnLabel"]
X = df.drop(columns=["ChurnLabel", "CLTV"], errors="ignore")

# 6) Identify which continuous columns to scale
continuous_cols = ["Age", "NumDependents", "TenureMonths", "SatisfactionScore"]

# 7) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# 8) Build pipeline: scale chosen continuous columns, leave all others “passthrough”
preprocessor = ColumnTransformer(
    transformers=[("num", StandardScaler(), [c for c in continuous_cols if c in X_train.columns])],
    remainder="passthrough"
)

pipe = Pipeline([
    ("preproc", preprocessor),
    ("clf", LogisticRegression(
        penalty="l2", C=1.0, solver="liblinear",
        class_weight="balanced", random_state=42
    ))
])

# 9) Fit model
pipe.fit(X_train, y_train)

# 10) Pull out the underlying LogisticRegression step and its coefficients
lr = pipe.named_steps["clf"]
feature_names = X_train.columns.tolist()
coefs = lr.coef_[0]  # shape = (n_features,)

# 11) Build a DataFrame with feature, coefficient, and odds_ratio
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefs,
    "odds_ratio": np.exp(coefs)
})
# 12) Sort by absolute coefficient magnitude
coef_df["abs_coef"] = np.abs(coef_df["coefficient"])
coef_df_sorted = coef_df.sort_values("abs_coef", ascending=False).drop(columns=["abs_coef"]).reset_index(drop=True)

# 13) Show top 10 drivers toward churn (positive β) and top 10 drivers protecting against churn (negative β)
print("Top 10 features driving churn (positive coefficient):")
print(coef_df_sorted[coef_df_sorted["coefficient"] > 0].head(10).to_string(index=False))
print("\nTop 10 features protecting against churn (negative coefficient):")
print(coef_df_sorted[coef_df_sorted["coefficient"] < 0].head(10).to_string(index=False))
