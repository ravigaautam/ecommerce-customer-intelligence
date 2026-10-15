import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.data import make_messy_sample, clean_transactions
from src.analysis import customer_rfm, train_cancellation_model

st.set_page_config(page_title="E-commerce Customer Intelligence", page_icon="🛍️", layout="wide")
st.title("🛍️ E-commerce Customer Intelligence Platform")
st.caption("Data cleaning • EDA • Feature engineering • Machine learning • SQL • Business insights")
with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload transaction CSV", type=["csv"])
    use_sample = st.checkbox("Use generated messy sample data", value=uploaded is None)
    st.caption("Required columns: InvoiceNo/OrderID, Quantity, InvoiceDate/OrderDate, UnitPrice/Price.")
try:
    if uploaded is not None and not use_sample:
        raw, source = pd.read_csv(uploaded), f"Uploaded: {uploaded.name}"
    else:
        raw, source = make_messy_sample(), "Generated synthetic sample data"
    cleaned, quality = clean_transactions(raw)
except Exception as exc:
    st.error(f"Could not load/clean data: {exc}")
    st.stop()
st.info(f"Data source: **{source}**. Synthetic-data results are for demonstration, not real business claims.")
with st.expander("Data quality report"):
    a, b, c, d = st.columns(4)
    a.metric("Rows before", f"{quality['rows_before']:,}")
    b.metric("Rows after", f"{quality['rows_after']:,}")
    c.metric("Duplicates removed", f"{quality['duplicates_removed']:,}")
    d.metric("Missing customer IDs", f"{quality['missing_customer_ids_after']:,}")
    st.dataframe(pd.DataFrame(quality["missing_values_before"].items(), columns=["Column", "Missing before cleaning"]), use_container_width=True)
with st.sidebar:
    st.header("Dashboard filters")
    country_options = sorted(cleaned["Country"].dropna().astype(str).unique())
    countries = st.multiselect("Country", country_options, default=country_options)
    month_options = sorted(cleaned["OrderMonth"].unique())
    months = st.multiselect("Order month", month_options, default=month_options)
filtered = cleaned[cleaned["Country"].astype(str).isin(countries) & cleaned["OrderMonth"].isin(months)].copy()
if filtered.empty:
    st.warning("No data matches these filters. Choose at least one country and month.")
    st.stop()
sales = filtered[(filtered["IsCancelled"] == 0) & (filtered["Quantity"] > 0)]
revenue = float(sales["Revenue"].sum())
orders = int(sales["InvoiceNo"].nunique())
customers = int(sales["CustomerID"].nunique())
aov = revenue / orders if orders else 0
cancel_rate = float(filtered["IsCancelled"].mean() * 100)
k = st.columns(5)
k[0].metric("Revenue", f"£{revenue:,.2f}")
k[1].metric("Orders", f"{orders:,}")
k[2].metric("Known customers", f"{customers:,}")
k[3].metric("Average order value", f"£{aov:,.2f}")
k[4].metric("Cancelled row rate", f"{cancel_rate:.1f}%")
t_overview, t_eda, t_customer, t_ml, t_data, t_sql = st.tabs(["Overview", "EDA", "Customer Insights", "Machine Learning", "Cleaned Data", "SQL Export"])
with t_overview:
    left, right = st.columns(2)
    monthly = sales.groupby("OrderMonth", as_index=False)["Revenue"].sum().sort_values("OrderMonth")
    with left:
        st.subheader("Monthly revenue trend")
        fig, ax = plt.subplots(figsize=(8, 4)); sns.lineplot(data=monthly, x="OrderMonth", y="Revenue", marker="o", ax=ax)
        ax.tick_params(axis="x", rotation=45); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with right:
        st.subheader("Top products by revenue")
        top = sales.groupby("Description", as_index=False)["Revenue"].sum().nlargest(10, "Revenue").sort_values("Revenue")
        fig, ax = plt.subplots(figsize=(8, 4)); sns.barplot(data=top, x="Revenue", y="Description", ax=ax)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.subheader("Quick business observations")
    if not monthly.empty:
        peak = monthly.loc[monthly["Revenue"].idxmax()]
        st.write(f"- Highest revenue month in this filter: **{peak['OrderMonth']}**.")
    st.write(f"- Current filtered dataset contains **{len(filtered):,} cleaned transaction rows**.")
    st.write("- These are descriptive findings, not evidence of causation.")
with t_eda:
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(7, 4)); sns.histplot(data=sales, x="UnitPrice", bins=30, kde=True, ax=ax)
        ax.set_title("Unit price distribution"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with right:
        country_sales = sales.groupby("Country", as_index=False)["Revenue"].sum().nlargest(10, "Revenue")
        fig, ax = plt.subplots(figsize=(7, 4)); sns.barplot(data=country_sales, x="Revenue", y="Country", ax=ax)
        ax.set_title("Revenue by country"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    numeric = sales[["Quantity", "UnitPrice", "Revenue"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(numeric) > 1:
        fig, ax = plt.subplots(figsize=(6, 4)); sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Numeric correlation matrix"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.subheader("Descriptive statistics")
    st.dataframe(sales.describe(include="all").transpose(), use_container_width=True)
with t_customer:
    st.subheader("RFM customer segmentation")
    rfm = customer_rfm(filtered)
    if rfm.empty:
        st.warning("No identified customers with positive, non-cancelled purchases are available for RFM.")
    else:
        rfm["Segment"] = "Regular"
        rfm.loc[(rfm["RecencyDays"] <= rfm["RecencyDays"].quantile(.33)) & (rfm["Monetary"] >= rfm["Monetary"].quantile(.67)), "Segment"] = "High value"
        rfm.loc[rfm["RecencyDays"] >= rfm["RecencyDays"].quantile(.80), "Segment"] = "At risk"
        segment = rfm.groupby("Segment", as_index=False).agg(Customers=("CustomerID", "nunique"), Revenue=("Monetary", "sum"))
        l, r = st.columns(2)
        l.dataframe(segment, use_container_width=True)
        fig, ax = plt.subplots(figsize=(6, 4)); sns.barplot(data=segment, x="Segment", y="Customers", ax=ax)
        ax.tick_params(axis="x", rotation=20); fig.tight_layout(); r.pyplot(fig); plt.close(fig)
        st.caption("RFM = Recency, Frequency, Monetary value. Thresholds are illustrative and should be validated for business use.")
        st.dataframe(rfm.sort_values("Monetary", ascending=False), use_container_width=True)
with t_ml:
    st.subheader("Random Forest cancellation classification")
    st.write("This demo predicts the cancellation label from transaction attributes. In real deployment, define the prediction time and exclude any information revealed only after cancellation to avoid target leakage.")
    model, metrics = train_cancellation_model(filtered)
    if model is None:
        st.warning(metrics["error"])
    else:
        cols = st.columns(5)
        for col, name in zip(cols, ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]):
            col.metric(name, f"{metrics[name]:.3f}")
        st.write(f"Test rows: **{metrics['Test rows']:,}**")
        fig, ax = plt.subplots(figsize=(5, 4)); sns.heatmap(np.array(metrics["Confusion matrix"]), annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted active", "Predicted cancelled"], yticklabels=["Actual active", "Actual cancelled"], ax=ax)
        ax.set_xlabel("Predicted label"); ax.set_ylabel("Actual label"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.json({k: v for k, v in metrics.items() if k != "Confusion matrix"})
        st.warning("Do not present synthetic-data metrics as real-world results. Validate with a real, time-aware dataset before making claims.")
with t_data:
    st.subheader("Cleaned transactions")
    st.dataframe(filtered, use_container_width=True, height=420)
    st.download_button("Download cleaned CSV", filtered.to_csv(index=False).encode("utf-8"), "cleaned_ecommerce_transactions.csv", "text/csv")
    st.download_button("Download RFM CSV", customer_rfm(filtered).to_csv(index=False).encode("utf-8"), "customer_rfm.csv", "text/csv")
with t_sql:
    st.subheader("Export cleaned data to SQLite")
    st.write("Click to create `ecommerce_transactions.db` in the folder from which Streamlit was launched.")
    if st.button("Create SQLite database"):
        try:
            with sqlite3.connect("ecommerce_transactions.db") as conn:
                filtered.to_sql("ecommerce_transactions", conn, if_exists="replace", index=False)
            st.success("Created ecommerce_transactions.db with table ecommerce_transactions.")
        except Exception as exc:
            st.error(f"Could not create database: {exc}")
    st.code("SELECT Country, ROUND(SUM(Revenue), 2) AS revenue\nFROM ecommerce_transactions\nWHERE IsCancelled = 0 AND Quantity > 0\nGROUP BY Country\nORDER BY revenue DESC;", language="sql")
st.divider()
st.caption("Portfolio project template • Inspect assumptions and validate on real data before deployment.")
