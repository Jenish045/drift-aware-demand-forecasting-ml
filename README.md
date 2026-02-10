# Drift-Aware Demand Forecasting Using Classical Machine Learning

## Project Objective
The objective of this project is to build a machine learning–oriented demand forecasting pipeline that analyzes historical demand patterns, highlights real-world challenges in time-series data, and prepares the foundation for drift-aware forecasting models. The project emphasizes correct data understanding, aggregation, and evaluation before applying machine learning techniques.

---

## Dataset Description
This project uses the **Historical Product Demand Dataset** from Kaggle.

**Source:** Kaggle  
**Dataset Name:** Historical Product Demand Dataset

### Dataset Columns
- `Date` – Date of demand observation
- `Product_Code` – Unique identifier for each product
- `Warehouse` – Warehouse from which the product was supplied
- `Order_Demand` – Demand quantity for the product

The dataset contains historical daily demand records for multiple products across several warehouses. Demand values are noisy, highly variable, and include intermittent and extreme spikes, making this dataset suitable for real-world demand forecasting analysis.

---

## Exploratory Data Analysis (EDA)

### Key Observations
- Demand data is recorded at the **warehouse level**, requiring aggregation to obtain valid product-level time series.
- Sales distributions are highly skewed, with many low-demand days and occasional extreme spikes.
- High-demand products show **intermittent but very large demand**, while low-demand products exhibit **stable and nearly constant demand**.
- Strong variability exists across products, indicating that a single forecasting strategy may not perform equally well for all items.
- Aggregation at the daily product level is essential before applying forecasting models.

---

## Challenges Identified
- Highly volatile and intermittent demand patterns.
- Presence of extreme outliers due to bulk orders.
- Temporal dependencies that violate assumptions of simple statistical models.
- Risk of model degradation over time due to changing demand behavior (concept drift).

---

## Methodology Overview
1. Data loading and cleaning of real-world demand values.
2. Aggregation of warehouse-level data into product-level daily demand.
3. Exploratory data analysis to understand trend, seasonality, and variability.
4. Comparison of high-demand and low-demand product behavior.
5. (Next phases) Statistical baselines, machine learning models, and drift handling.

---

## Project Roadmap
- **Week 1:** Data understanding and exploratory data analysis  
- **Week 2:** Statistical baseline forecasting models  
- **Week 3:** Feature engineering for supervised machine learning  
- **Week 4:** Classical ML model training and evaluation  
- **Week 5:** Time-aware validation and error analysis  
- **Week 6:** Concept drift detection and retraining strategies  

---

## Tools & Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn (later stages)

---

## Key Learning Outcomes
- Handling messy, real-world time-series data
- Proper aggregation strategies for forecasting
- Understanding demand variability across products
- Building forecasting systems with long-term robustness in mind

---

## Status
✅ Week 1 completed  
🚧 Week 2 in progress