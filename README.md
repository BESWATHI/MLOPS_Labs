
# Airflow Lab 1 — Mall Customers K-Means Clustering

### Overview

This project builds and runs a **K-Means clustering pipeline** using **Apache Airflow**.
The workflow automates loading mall customer data, preprocessing it, training and saving a K-Means model, and predicting clusters on test data using the **elbow method** to find the optimal number of clusters.

---

### Updates in This Version

* **Dataset change:**
  The default dataset was replaced with **`mall_customers.csv`** and **`mall_customers_test.csv`**, stored under the `data/` folder.

* **Changes in `lab.py`:**

  * Automatically detects numeric columns instead of hardcoding `BALANCE`, `PURCHASES`, `CREDIT_LIMIT`.
  * Uses `MinMaxScaler` for normalization.
  * Computes the optimal *k* via `KneeLocator` (elbow method).
  * Saves the model, scaler, and feature list together as `model.sav`.
  * Applies the same scaler and features on `mall_customers_test.csv` for prediction.
  * Uses Base64 and Pickle for XCom-safe data transfer between tasks.

---

## ML Model

The pipeline performs **unsupervised customer segmentation** based on features such as age, income, spending score, savings, and family size.

### Steps and Functions

1. **`load_data()`**
   Loads `mall_customers.csv`, serializes it using Pickle, and returns Base64-encoded data for Airflow XCom.

2. **`data_preprocessing(data)`**
   Deserializes data, removes missing values, automatically detects numeric features (`Age`, `Annual_Income_(k$)`, `Spending_Score`, `Savings_(k$)`, `Family_Size`), scales them to [0, 1], and returns scaled data with the fitted scaler and feature list.

3. **`build_save_model(data, filename)`**
   Computes SSE for k = 1–49, determines the elbow (optimal k), fits K-Means, and saves the trained model bundle `{model, scaler, features, k*}` to `model/model.sav`.

4. **`load_model_elbow(filename, sse)`**
   Loads the saved bundle, logs the elbow result, scales `mall_customers_test.csv` using the same scaler and features, and predicts the first cluster label.

---

## Airflow Setup

### Directory Structure

```
MLOPS_Labs/
├── assets/
│   └── Airflow_lab1_DAG.png
├── Lab_1/
│   ├── data/
│   │   ├── mall_customers.csv
│   │   └── mall_customers_test.csv
│   ├── model/
│   │   └── model.sav
│   ├── dags/
│   │   └── airflow1.py
│   ├── src/
│   │   └── __init__.py
│   ├── lab.py
│   └── airflow.cfg
├── docker-compose.yaml
├── .env
└── README.md
```

### Prerequisites

* **Docker Desktop** (≥ 4 GB RAM; 8 GB recommended)
* **Python Packages:** `pandas`, `scikit-learn`, `kneed`, `pickle`

---

## Airflow DAG Script

**DAG Name:** `Airflow_Lab1`

This DAG runs the end-to-end ML workflow with four main tasks:

1. `load_data_task` → calls `load_data()`
2. `data_preprocessing_task` → calls `data_preprocessing()`
3. `build_save_model_task` → calls `build_save_model()`
4. `load_model_task` → calls `load_model_elbow()`

### Task Dependencies

```python
load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task
```

Each task passes its output to the next using Airflow **XCom** (Base64-encoded pickle).

---

## Running Airflow in Docker

### 1. Initialize Airflow

Run this once to set up the metadata database:

```
docker compose up airflow-init
```

### 2. Start Airflow Services

```
docker compose up
```

Wait until you see:

```
app-airflow-webserver-1 | 127.0.0.1 - - [xx/Oct/2025:xx:xx:xx +0000] "GET /health HTTP/1.1" 200 141 "-" "curl/7.74.0"
```

### 3. Access the Airflow Web Interface

Open your browser and go to:

 **[http://localhost:8080](http://localhost:8080)**

**Login credentials** (defined in `.env`):

```
Username: airflow2
Password: airflow2
```

(or whatever credentials you set in your docker-compose file).

---

### 4. Trigger the DAG

* In the Airflow UI, locate **Airflow_Lab1**.
* Turn the toggle **ON** and click **Trigger DAG** to start the pipeline.
* Monitor task progress in the **Graph View** (see screenshot in `assets/Airflow_lab1_DAG.png`).

---

## Outputs

| File                           | Description                                          |
| :----------------------------- | :--------------------------------------------------- |
| `model/model.sav`              | Saved bundle (model + scaler + features + optimal k) |
| `data/mall_customers.csv`      | Training dataset                                     |
| `data/mall_customers_test.csv` | Testing dataset                                      |
| `assets/Airflow_lab1_DAG.png`  | DAG execution screenshot                             |
| Airflow Logs                   | Task execution and model training details            |

---

## Summary

This lab demonstrates a complete **K-Means clustering workflow** orchestrated with **Apache Airflow**.
The updated version enhances the previous lab by:

* Using the **Mall Customers dataset** for realistic segmentation tasks.
* Automatically selecting numeric features for scalability.
* Sharing a single scaler between training and testing stages.

