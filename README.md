# 📊 Customer Segmentation (RFM + K‑Means)

An interactive Streamlit app that performs **customer segmentation** using **RFM analysis** (Recency, Frequency, Monetary) and **K‑Means clustering**. You can use a built‑in sample dataset or upload your own transactions, explore distributions, determine the optimal number of clusters, visualize segments, and download the results.

- **Live demo:** https://customer-segmentation-eqpmr72b6iaecr3wyyjx5s.streamlit.app/  
- **Source code:** https://github.com/zenklinov/Customer-Segmentation  
- **Main app file:** `customer-segmentation.py`

---

## ✨ Features

- **Two data sources**
  - Use a **sample dataset** (synthetic transactions).
  - **Upload CSV** with your own transaction data.
- **RFM calculation**
  - Computes Recency (days since last purchase), Frequency (number of invoices), and Monetary (total spend).
- **Exploratory visuals**
  - Histograms (Recency/Frequency/Monetary), summary tables, and dataset info.
- **Clustering**
  - Preprocessing: outlier removal (IQR), **standardization**.
  - **Elbow method** to help choose the optimal number of clusters.
  - **K‑Means** (2–6 clusters) with interactive plots (Plotly scatter, counts, pie chart).
- **Business insights**
  - Automatic labeling of common segments (e.g., **Champions**, **Loyal Customers**, **Recent Customers**, **Lost Customers**, **Need Attention**) with actionable recommendations.
- **Export**
  - Download segmentation results as **CSV**.

---

## 🧠 What is RFM?

- **Recency** – How recently a customer purchased (lower is better).
- **Frequency** – How often they purchase (higher is better).
- **Monetary** – How much they spend (higher is better).

This app computes RFM from your transaction data and applies K‑Means to group customers into data‑driven segments.

---

## 📁 Expected CSV Columns

Your uploaded CSV should contain at least:

- `CustomerID` – Unique customer identifier  
- `InvoiceDate` – Transaction date (parseable as a datetime)  
- `InvoiceNo` – Invoice number  
- `Quantity` – Units purchased  
- `UnitPrice` – Price per unit  

If `TotalAmount` is missing, the app will compute it as `Quantity * UnitPrice`.

> **Note:** You can **map columns** from your CSV to these fields inside the app (so your column names do not have to match exactly).

---

## 🚀 Quick Start (Local)

### 1) Clone the repo
```bash
git clone https://github.com/zenklinov/Customer-Segmentation.git
cd Customer-Segmentation
```

### 2) Create a virtual environment (recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3) Install dependencies
```txt
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
plotly
```
Then install:
```bash
pip install -r requirements.txt
```

_Alternatively_
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn plotly
```

### 4) Run the app
```bash
streamlit run customer-segmentation.py
```
Open the URL printed in the terminal (usually http://localhost:8501).

## 🧭 App Workflow

The sidebar lets you navigate through four sections:

### 1) **Data Source Configuration**
- Choose **Use Sample Dataset** or **Upload Your Data (CSV)**.
- Preview your data, see basic info and summary stats.
- If uploading, map your columns (CustomerID, InvoiceDate, InvoiceNo, Quantity, UnitPrice).
- Proceed to RFM analysis.

### 2) **RFM Analysis**
- The app computes RFM per customer:
  - **Recency**: days since the most recent purchase (relative to max date + 1 day in your data).
  - **Frequency**: count of invoices.
  - **Monetary**: sum of `TotalAmount`.
- View histograms for each metric and a data preview.
- Choose a scoring approach:
  - **K‑Means Clustering (Recommended)** – proceed to preprocessing & clustering.
  - **Manual Scoring (1–4)** – placeholder UI (implementation stub for custom scoring).

### 3) **Clustering Results**
- **Outlier removal**: IQR method (1.5×IQR) for R, F, M.
- **Scaling**: Standardization (`StandardScaler`).
- **Elbow plot**: Inspect inertia for k ∈ [1..9].
- Select **number of clusters (2–6)** and run **K‑Means**.
- Explore results:
  - Cluster size chart and pie chart.
  - **Cluster profile heatmap** (mean R/F/M by cluster).
  - **Interactive scatter** (choose axes, colored by cluster).
- **Download** the full customer‑level results as CSV.

### 4) **Business Insights**
- Automatically assigns human‑readable labels (e.g., **Champions**, **Loyal Customers**, **Recent Customers**, **Lost Customers**, **Need Attention**) based on relative R/F/M.
- For each segment:
  - Shows **metrics** (Recency, Frequency, Monetary).
  - Lists **actionable recommendations** (e.g., win‑back, loyalty, upsell/cross‑sell).
  - Displays **sample customers**.
- Includes high‑level **marketing strategy** suggestions.

---

## 🔍 How RFM is Computed (Conceptual)

For each `CustomerID`:
- `Recency = (reference_date - last(InvoiceDate)).days`, where `reference_date = max(InvoiceDate) + 1 day`
- `Frequency = number of InvoiceNo`
- `Monetary = sum(TotalAmount)`

---

## 🧩 Notes & Limitations

- **Manual Scoring** is currently a placeholder in the UI. If you need it, you can extend the code to assign quartile‑based (or custom) scores for R/F/M.
- **Outlier removal** uses a simple IQR rule; consider tuning the factor or method for your data.
- The **sample dataset** is synthetic and generated for demonstration (randomized transactions across a fixed period). Real data will behave differently.
- For **very large datasets**, you may want to optimize memory usage or switch to chunked processing.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit  
- **Data**: pandas, numpy  
- **Visualization**: matplotlib, seaborn, Plotly  
- **ML**: scikit‑learn (StandardScaler, KMeans)

---

## 🤝 Contributing

Pull requests and issues are welcome:
1. Fork the repository.
2. Create a feature branch.
3. Commit changes with clear messages.
4. Open a PR describing your changes.

---

## 📄 License

Please refer to the repository for licensing information.

---

## 🙌 Acknowledgements

Thanks to the open‑source community behind Streamlit, pandas, scikit‑learn, matplotlib, seaborn, and Plotly.
