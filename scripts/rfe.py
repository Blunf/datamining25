import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1) Load data and one‐hot encode / binary‐map exactly as before
df = pd.read_csv("data/combined_telco_churn_with_hasInternet.csv")

binary_map = {"yes": 1, "no": 0, "no phone service": 0, "no internet service": 0}
binary_cols = [
    "SeniorCitizen", "Partner", "NumDependents", "Phone Service", "Multiple Lines",
    "Online Security", "Online Backup", "Device Protection", "Tech Support",
    "Streaming TV", "Streaming Movies", "Paperless Billing", "HasInternet"
]
for col in binary_cols:
    if col in df.columns:
        df[col] = (
            df[col].astype(str).str.strip().str.lower()
                 .map(binary_map)
                 .fillna(0)
                 .astype(int)
        )

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

object_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != "ChurnLabel"]
df = pd.get_dummies(df, columns=object_cols, drop_first=True)

# 2) Split into X/y
y = df["ChurnLabel"]
X = df.drop(columns=["ChurnLabel", "CLTV"], errors="ignore")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# 3) Train RF
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42
)
rf.fit(X_train, y_train)

# 4) Extract feature importances and sort descending
feat_imp = pd.DataFrame({
    "feature": X_train.columns,
    "importance": rf.feature_importances_
})
feat_imp_sorted = feat_imp.sort_values("importance", ascending=False).reset_index(drop=True)

# 5) Print the top 15 most important features
print("Top 15 Random Forest feature importances:")
print(feat_imp_sorted.head(15).to_string(index=False))
