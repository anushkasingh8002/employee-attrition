import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# =========================
# 1. LOAD DATA
# =========================

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

print("Dataset shape:", df.shape)


# =========================
# 2. TARGET ENCODING
# =========================

df["Attrition"] = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})


# =========================
# 3. DROP IRRELEVANT COLUMNS
# =========================

drop_columns = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

df = df.drop(columns=drop_columns)


# =========================
# 4. ONE-HOT ENCODING
# =========================

X = df.drop(columns=["Attrition"])
y = df["Attrition"]

X = pd.get_dummies(X, drop_first=True)

# Convert boolean columns to integers
X = X.astype(int)


# =========================
# 5. TRAIN-TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =========================
# 6. SAVE FEATURE COLUMNS
# =========================

columns = list(X.columns)

joblib.dump(columns, "columns.pkl")


# =========================
# 7. LOGISTIC REGRESSION
# =========================

lr_model = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    random_state=42
)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)


# =========================
# 8. DECISION TREE
# =========================

dt_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    class_weight="balanced",
    random_state=42
)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)


# =========================
# 9. RANDOM FOREST
# =========================

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    class_weight="balanced",
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)


# =========================
# 10. EVALUATION
# =========================

models = {
    "Logistic Regression": lr_pred,
    "Decision Tree": dt_pred,
    "Random Forest": rf_pred
}

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

for name, predictions in models.items():

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print(f"\n{name}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


# =========================
# 11. SAVE MODELS
# =========================

joblib.dump(lr_model, "lr.pkl")
joblib.dump(dt_model, "dt.pkl")
joblib.dump(rf_model, "rf.pkl")


# =========================
# 12. MODERN JOB-ROLE PROXIES
# =========================
# The original IBM HR dataset does not contain AI/ML Engineer or
# Software Developer. These application roles therefore use an
# existing dataset role as a transparent proxy. The model itself is
# not being falsely presented as trained on these modern job titles.

job_role_proxy = {
    "AI/ML Engineer": "Research Scientist",
    "Software Developer": "Research Scientist"
}

joblib.dump(job_role_proxy, "job_role_proxy.pkl")


print("\n" + "=" * 60)
print("MODELS SAVED SUCCESSFULLY")
print("=" * 60)

print("\nFeature count:", len(columns))
print("Random Forest features:", len(rf_model.feature_names_in_))

print("\nFiles updated:")
print("- lr.pkl")
print("- dt.pkl")
print("- rf.pkl")
print("- columns.pkl")
print("- job_role_proxy.pkl")
