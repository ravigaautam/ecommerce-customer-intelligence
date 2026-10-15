import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def customer_rfm(df: pd.DataFrame) -> pd.DataFrame:
    known = df.dropna(subset=["CustomerID"])
    purchases = known[(known["IsCancelled"] == 0) & (known["Quantity"] > 0)].copy()
    cols = ["CustomerID", "RecencyDays", "Frequency", "Monetary"]
    if purchases.empty:
        return pd.DataFrame(columns=cols)
    snapshot = purchases["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = purchases.groupby("CustomerID", as_index=False).agg(
        LastPurchase=("InvoiceDate", "max"), Frequency=("InvoiceNo", "nunique"), Monetary=("Revenue", "sum"))
    rfm["RecencyDays"] = (snapshot - rfm["LastPurchase"]).dt.days
    return rfm.drop(columns="LastPurchase")[cols]


def train_cancellation_model(df: pd.DataFrame):
    data = df.copy()
    data["Month"] = data["InvoiceDate"].dt.month
    data["DayOfWeek"] = data["InvoiceDate"].dt.dayofweek
    data["IsWeekend"] = data["DayOfWeek"].isin([5, 6]).astype(int)
    data["CustomerKnown"] = data["CustomerID"].notna().astype(int)
    features = ["Quantity", "UnitPrice", "Month", "DayOfWeek", "IsWeekend", "CustomerKnown", "Country"]
    X, y = data[features], data["IsCancelled"].astype(int)
    if y.nunique() < 2 or y.value_counts().min() < 2:
        return None, {"error": "Not enough examples from both classes for train/test evaluation."}
    numeric = ["Quantity", "UnitPrice", "Month", "DayOfWeek", "IsWeekend", "CustomerKnown"]
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore"))]), ["Country"]),
    ])
    model = Pipeline([("preprocess", prep), ("classifier", RandomForestClassifier(
        n_estimators=160, class_weight="balanced", random_state=42, min_samples_leaf=2, n_jobs=-1))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    pred, prob = model.predict(X_test), model.predict_proba(X_test)[:, 1]
    metrics = {"Accuracy": float(accuracy_score(y_test, pred)),
               "Precision": float(precision_score(y_test, pred, zero_division=0)),
               "Recall": float(recall_score(y_test, pred, zero_division=0)),
               "F1-score": float(f1_score(y_test, pred, zero_division=0)),
               "ROC-AUC": float(roc_auc_score(y_test, prob)),
               "Test rows": int(len(y_test)),
               "Confusion matrix": confusion_matrix(y_test, pred).tolist()}
    return model, metrics
