# E-commerce Customer Intelligence & Purchase Analytics

A portfolio-ready learning project using Python, Pandas, NumPy, SQL, scikit-learn, Matplotlib, Seaborn, and Streamlit.

## What is included
- Intentionally messy sample transaction data: missing values, duplicates, invalid dates/prices, and cancellation rows.
- Data cleaning and data-quality report.
- Exploratory Data Analysis (EDA): monthly revenue, top products, country sales, distributions, and correlations.
- Feature engineering: revenue, month/day features, customer-known indicator, and RFM (Recency, Frequency, Monetary) customer metrics.
- Random Forest cancellation classifier with accuracy, precision, recall, F1, ROC-AUC, and confusion matrix.
- Streamlit dashboard with filters, KPIs, charts, cleaned-data preview, and CSV downloads.
- SQLite database export and sample SQL queries.

## Run on Windows PowerShell
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
If activation is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again. Open the local URL Streamlit prints, usually `http://localhost:8501`.

## Data source
The app generates synthetic messy data by default, so it works immediately without a download. Use the sidebar to upload your own transaction CSV. Supported columns include InvoiceNo/OrderID, Quantity, InvoiceDate/OrderDate, UnitPrice/Price, CustomerID, Country, StockCode/ProductID, and Description/ProductName. Required: invoice/order ID, quantity, date, and unit price.

For a real portfolio version, replace the generated data with a real public dataset such as the UCI Online Retail dataset: https://archive.ics.uci.edu/dataset/352/online+retail

## SQL
The cleaned rows are written to `ecommerce_transactions.db` when you click the SQL export button. Example queries are in `sql/queries.sql`.

## Important interview note
The default dataset is synthetic. Never claim its metrics represent real business performance. The cancellation classifier is a learning demonstration; in a production design, define the prediction timestamp and ensure every feature is available before the cancellation event to prevent leakage. Report only the metrics shown by your own run.
