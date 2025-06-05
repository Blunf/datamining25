import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 1) Load and preprocess data
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
            df[col]
            .astype(str).str.strip().str.lower()
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

y = df["ChurnLabel"]
X = df.drop(columns=["ChurnLabel", "CLTV"], errors="ignore")
continuous_cols = ["Age", "NumDependents", "TenureMonths", "SatisfactionScore"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# 2) Fit Logistic Regression
preprocessor = ColumnTransformer(
    transformers=[("num", StandardScaler(), [c for c in continuous_cols if c in X_train.columns])],
    remainder="passthrough"
)
log_pipe = Pipeline([
    ("preproc", preprocessor),
    ("clf", LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                               class_weight="balanced", random_state=42))
])
log_pipe.fit(X_train, y_train)
lr = log_pipe.named_steps["clf"]
feature_names = X_train.columns.tolist()
coefs = lr.coef_[0]

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefs
})
coef_df["abs_coef"] = np.abs(coef_df["coefficient"])
coef_df_sorted = coef_df.sort_values("abs_coef", ascending=False).head(10).copy()
coef_df_sorted["coef_norm"] = coef_df_sorted["coefficient"] / coef_df_sorted["coefficient"].abs().max()

# 3) Fit Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_leaf=1,
                            class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)
feat_imp = pd.DataFrame({
    "feature": X_train.columns,
    "importance": rf.feature_importances_
})
feat_imp_sorted = feat_imp.sort_values("importance", ascending=False).head(10).copy()
feat_imp_sorted["imp_norm"] = feat_imp_sorted["importance"] / feat_imp_sorted["importance"].max()

# 4) Prepare for plotting
lr_plot = coef_df_sorted.set_index("feature").loc[:, ["coef_norm", "coefficient"]].sort_values("coef_norm")
rf_plot = feat_imp_sorted.set_index("feature").loc[:, ["imp_norm"]].sort_values("imp_norm")

# 5) Plot
plt.figure(figsize=(14, 7))

# Left subplot: Logistic Regression
plt.subplot(1, 2, 1)
bars = plt.barh(
    lr_plot.index,
    lr_plot["coef_norm"],
    color=lr_plot["coefficient"].apply(lambda x: "firebrick" if x > 0 else "seagreen")
)
for idx, (val, orig) in enumerate(zip(lr_plot["coef_norm"], lr_plot["coefficient"])):
    plt.text(
        val + 0.02 * np.sign(val),
        idx,
        f"{orig:.2f}",
        va="center",
        fontsize=9
    )
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Top 10 LR Coefficients (normalized)")
plt.xlabel("Normalized Coefficient (signed)")
plt.ylabel("Feature")
plt.gca().invert_yaxis()

# Right subplot: Random Forest
plt.subplot(1, 2, 2)
bars2 = plt.barh(rf_plot.index, rf_plot["imp_norm"], color="darkorange")
for idx, val in enumerate(rf_plot["imp_norm"]):
    imp_val = feat_imp_sorted.set_index("feature").loc[rf_plot.index[idx], "importance"]
    plt.text(val + 0.02, idx, f"{imp_val:.3f}", va="center", fontsize=9)
plt.title("Top 10 RF Importances (normalized)")
plt.xlabel("Normalized Importance")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()
