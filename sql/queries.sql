-- Example queries after creating the SQLite database from the dashboard.
SELECT Country, ROUND(SUM(Revenue), 2) AS revenue
FROM ecommerce_transactions
WHERE IsCancelled = 0 AND Quantity > 0
GROUP BY Country ORDER BY revenue DESC;

SELECT OrderMonth, ROUND(SUM(Revenue), 2) AS monthly_revenue
FROM ecommerce_transactions
WHERE IsCancelled = 0 AND Quantity > 0
GROUP BY OrderMonth ORDER BY OrderMonth;

SELECT CustomerID, COUNT(DISTINCT InvoiceNo) AS order_frequency,
       ROUND(SUM(Revenue), 2) AS monetary_value
FROM ecommerce_transactions
WHERE CustomerID IS NOT NULL AND IsCancelled = 0 AND Quantity > 0
GROUP BY CustomerID ORDER BY monetary_value DESC LIMIT 20;
