import numpy as np
import pandas as pd


def make_messy_sample(n_orders: int = 1800, seed: int = 42) -> pd.DataFrame:
    """Create reproducible synthetic transaction data with intentional quality issues."""
    rng = np.random.default_rng(seed)
    products = [f"P{i:03d}" for i in range(1, 61)]
    descriptions = [f"Product {i:03d}" for i in range(1, 61)]
    countries = ["United Kingdom", "Germany", "France", "Spain", "India", "Netherlands"]
    dates = pd.Series(pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n_orders), unit="D"), dtype="object")
    qty = rng.integers(1, 9, n_orders).astype(float)
    prices = np.round(rng.lognormal(2.2, 0.7, n_orders), 2)
    cancelled = rng.choice([0, 1], n_orders, p=[0.91, 0.09])
    invoice = [str(rng.integers(100000, 999999)) for _ in range(n_orders)]
    for i, flag in enumerate(cancelled):
        if flag:
            invoice[i] = "C" + invoice[i]
            qty[i] = -abs(qty[i])
    df = pd.DataFrame({
        "InvoiceNo": invoice, "StockCode": rng.choice(products, n_orders),
        "Description": rng.choice(descriptions, n_orders), "Quantity": qty,
        "InvoiceDate": dates, "UnitPrice": prices,
        "CustomerID": rng.integers(10000, 10280, n_orders).astype(float),
        "Country": rng.choice(countries, n_orders),
    })
    for col, fraction in [("CustomerID", 0.03), ("UnitPrice", 0.015), ("Description", 0.01)]:
        idx = rng.choice(df.index, max(1, int(n_orders * fraction)), replace=False)
        df.loc[idx, col] = np.nan if col != "Description" else "  "
    # A few malformed records are intentional so the cleaning report is meaningful.
    if len(df) > 5:
        df.loc[0, "InvoiceDate"] = "not-a-date"
        df.loc[1, "UnitPrice"] = -2
    return pd.concat([df, df.sample(frac=0.025, random_state=seed)], ignore_index=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "").replace("_", "").replace("-", "").lower() for c in df.columns]
    aliases = {
        "invoiceno": "InvoiceNo", "invoice": "InvoiceNo", "orderid": "InvoiceNo",
        "stockcode": "StockCode", "productid": "StockCode", "description": "Description",
        "productname": "Description", "quantity": "Quantity", "invoicedate": "InvoiceDate",
        "orderdate": "InvoiceDate", "unitprice": "UnitPrice", "price": "UnitPrice",
        "customerid": "CustomerID", "customer": "CustomerID", "country": "Country",
        "iscancelled": "IsCancelled", "cancelled": "IsCancelled"
    }
    return df.rename(columns={c: aliases[c] for c in df.columns if c in aliases})


def clean_transactions(raw: pd.DataFrame):
    df = normalize_columns(raw)
    required = ["InvoiceNo", "Quantity", "InvoiceDate", "UnitPrice"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    for col, default in [("StockCode", "Unknown"), ("Description", "Unknown product"),
                         ("CustomerID", np.nan), ("Country", "Unknown")]:
        if col not in df:
            df[col] = default
    before = len(df)
    missing_before = df.isna().sum().to_dict()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    for col in ["Quantity", "UnitPrice", "CustomerID"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Description"] = df["Description"].astype("string").str.strip()
    df["Country"] = df["Country"].astype("string").str.strip().replace("", "Unknown")
    if "IsCancelled" in df.columns:
        v = df["IsCancelled"].astype(str).str.lower().str.strip()
        df["IsCancelled"] = v.isin(["1", "true", "yes", "y", "cancelled", "canceled"]).astype(int)
    else:
        df["IsCancelled"] = df["InvoiceNo"].str.upper().str.startswith("C").astype(int)
    deduped = df.drop_duplicates()
    df = deduped.dropna(subset=required)
    df = df[(df["UnitPrice"] >= 0) & (df["Quantity"] != 0)]
    df["Description"] = df["Description"].fillna("Unknown product").replace("", "Unknown product")
    df["CustomerID"] = df["CustomerID"].astype("Int64")
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["OrderMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    report = {
        "rows_before": int(before), "rows_after": int(len(df)),
        "duplicates_removed": int(before - len(deduped)),
        "rows_removed_total": int(before - len(df)),
        "missing_values_before": {k: int(v) for k, v in missing_before.items()},
        "missing_customer_ids_after": int(df["CustomerID"].isna().sum()),
    }
    return df.reset_index(drop=True), report
