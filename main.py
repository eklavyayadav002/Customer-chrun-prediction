import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Small synthetic example dataset, included so this project runs offline.
df = pd.DataFrame({
    "tenure": [1,3,5,8,12,18,24,30,36,48,2,6,15,28,42,9,20,33,50,4],
    "monthly_charge": [90,85,80,75,70,65,60,55,50,45,95,88,72,58,48,82,67,53,44,91],
    "contract": ["monthly","monthly","monthly","yearly","yearly","yearly","yearly","two_year","two_year","two_year",
                 "monthly","monthly","yearly","two_year","two_year","monthly","yearly","two_year","two_year","monthly"],
    "support_calls": [5,4,4,3,2,2,1,1,0,0,6,5,3,1,0,4,2,1,0,5],
    "churn": [1,1,1,1,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,1]
})
X, y = df.drop(columns="churn"), df["churn"]
numeric = ["tenure", "monthly_charge", "support_calls"]
categorical = ["contract"]
prep = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])
model = Pipeline([("prep", prep),
                  ("rf", RandomForestClassifier(n_estimators=100, random_state=42,
                                                class_weight="balanced"))])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Customer Churn Prediction (synthetic demo data)")
print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print(classification_report(y_test, pred, zero_division=0))
new_customer = pd.DataFrame([{"tenure": 3, "monthly_charge": 89,
                              "contract": "monthly", "support_calls": 4}])
print("New customer churn (1 = predicted churn):", int(model.predict(new_customer)[0]))
