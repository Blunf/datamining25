import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

df = pd.read_csv("data/combined_telco_churn_with_hasInternet.csv")
y = df["ChurnLabel"]
X = df.drop(columns=["ChurnLabel"])

object_cols = ["Contract", "PaymentMethod", "Offer"]
X_encoded = pd.get_dummies(X, columns=object_cols, drop_first=True)

continuous_cols = ["Age", "NumDependents", "TenureMonths", "SatisfactionScore", "CLTV"]

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.20, stratify=y, random_state=42
)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

cm_rf = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix:")
print("                Predicted 0    Predicted 1")
print(f"Actual 0    {cm_rf[0,0]:<15} {cm_rf[0,1]}")
print(f"Actual 1    {cm_rf[1,0]:<15} {cm_rf[1,1]}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred_rf, digits=4))

print(f"\nROC-AUC: {roc_auc_score(y_test, y_proba_rf):.4f}")
