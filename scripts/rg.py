import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

df = pd.read_csv("../data/combined_telco_churn_with_hasInternet.csv")

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

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
print("Logistic Regression")
print("Confusion Matrix:")
print("                Predicted 0    Predicted 1")
print(f"Actual 0    {cm[0,0]:<15} {cm[0,1]}")
print(f"Actual 1    {cm[1,0]:<15} {cm[1,1]}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, y_proba)
print(f"\nROC-AUC: {auc:.4f}")
