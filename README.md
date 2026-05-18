# 🛡️ ServerGuard AI — Server Failure Prediction

> A machine learning–powered system that predicts server failure risk in real time using a Logistic Regression model and an interactive browser-based dashboard.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [How It Works](#how-it-works)
  - [Feature Engineering](#feature-engineering)
  - [Model Training](#model-training)
  - [Risk Classification](#risk-classification)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Training the Model](#training-the-model)
  - [Running the Dashboard](#running-the-dashboard)
- [Input Metrics](#input-metrics)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Dashboard Preview](#dashboard-preview)

---

## Overview

**ServerGuard AI** is a predictive maintenance tool designed for DevOps engineers and system administrators. It analyses 10 real-time server health metrics — such as CPU usage, memory, temperature, and network errors — and outputs a failure probability score along with an actionable risk level.

The project consists of two main components:

1. **Python ML Pipeline** (`server_failure_prediction.py`) — trains and serialises a Logistic Regression model.
2. **Interactive Web Dashboard** (`index.html` + `app.js` + `styles.css`) — a browser-based UI that simulates predictions using the model's logic, with live animated gauges and configurable metric sliders.

---

## Features

-  **Real-time failure probability** displayed as an animated circular gauge
-  **Risk classification** across four severity levels: LOW, MEDIUM, HIGH, CRITICAL
-  **Threshold alerts** that flag any metric exceeding its safe operating limit
-  **Engineered features panel** showing derived model inputs (stress score, error rate, etc.)
-  **Preset scenarios** — instantly load a Healthy or Critical server profile
-  **Randomize** — generate random metric combinations for exploration
-  **SMOTE oversampling** to handle class imbalance in training data
-  **Hyperparameter tuning** via `GridSearchCV` with 5-fold cross-validation
-  **Optimal threshold selection** maximising F1 score

---

## Project Structure

```
Server Failure Prediction/
│
├── server_failure_prediction.py   # ML pipeline: training, evaluation & prediction function
├── server_failure_dataset.csv     # Training dataset
├── failure_model.pkl              # Serialised trained Logistic Regression model
├── scaler.pkl                     # Serialised StandardScaler
│
├── index.html                     # Dashboard HTML structure
├── app.js                         # Prediction engine + interactive UI controller
├── styles.css                     # Dashboard styling (light blue-grey theme)
│
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | Scikit-learn — `LogisticRegression` |
| Imbalance Handling | `imbalanced-learn` — SMOTE |
| Hyperparameter Tuning | `GridSearchCV` (ROC-AUC scoring) |
| Data Processing | Pandas, NumPy |
| Model Serialisation | Joblib |
| Dashboard | Vanilla HTML, CSS, JavaScript |
| Fonts | Inter + JetBrains Mono (Google Fonts) |

---

## How It Works

### Feature Engineering

In addition to the 10 raw server metrics, four derived features are computed to improve model performance:

| Engineered Feature | Formula | Purpose |
|---|---|---|
| `stress_score` | Sum of 7 binary threshold flags | Composite overload indicator (0–7) |
| `cpu_x_memory` | `(cpu_usage × memory_usage) / 10000` | Interaction between two high-impact metrics |
| `temp_x_disk` | `(temperature × disk_io) / 10000` | Hardware stress interaction |
| `error_rate` | `error_log_count / (uptime_hours + 1)` | Normalised error frequency over time |
| `latency_per_process` | `response_time_ms / process_count` | Average latency burden per process |

### Model Training

```
Raw CSV → Feature Engineering → Train/Test Split (80/20, stratified)
       → StandardScaler → SMOTE resampling → LogisticRegression
       → GridSearchCV (C, penalty, solver) → Optimal Threshold Selection → .pkl export
```

- **Algorithm**: Logistic Regression with L1/L2 regularisation
- **Class imbalance**: Addressed via `class_weight='balanced'` and SMOTE oversampling
- **Threshold tuning**: Sweeps thresholds from 0.10 to 0.90 in 0.05 steps, selects the one maximising F1 score on the test set

### Risk Classification

| Failure Probability | Risk Level | Recommended Action |
|---|---|---|
| < 30% | 🟢 LOW | Monitor normally |
| 30% – 49% | 🟡 MEDIUM | Schedule maintenance check |
| 50% – 69% | 🔴 HIGH | Inspect within 24 hours |
| ≥ 70% | 🚨 CRITICAL | Immediate intervention required |

---

## Getting Started

### Prerequisites

- Python 3.8+
- `pip` package manager
- A modern web browser (for the dashboard)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/Server-Failure-Prediction-Model.git
cd "Server Failure Prediction"

# Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

# Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn joblib
```

### Training the Model

```bash
python server_failure_prediction.py
```

This will:
1. Load and engineer features from `server_failure_dataset.csv`
2. Train and tune the Logistic Regression model
3. Print the classification report and confusion matrix
4. Save `failure_model.pkl` and `scaler.pkl`
5. Run predictions on a sample healthy and critical server

### Running the Dashboard

No build step or server is required — simply open `index.html` in your browser:

```bash
# Windows
start index.html
or 
python -m http.server 8080
#Dashboard is on http://localhost:8080

# macOS
open index.html

# Linux
xdg-open index.html
```

> **Note:** The dashboard replicates the model's prediction logic in JavaScript, including identical feature engineering and a logistic sigmoid function calibrated to produce results consistent with the trained Python model.

---

## Input Metrics

| Metric | Unit | Safe Threshold |
|---|---|---|
| CPU Usage | % | ≤ 85% |
| Memory Usage | % | ≤ 88% |
| Disk I/O | % | ≤ 80% |
| Temperature | °C | ≤ 82°C |
| Network Errors | /min | ≤ 18 |
| Uptime | hours | — |
| Error Log Count | count | ≤ 45 |
| Process Count | count | ≤ 400 |
| Swap Usage | % | ≤ 78% |
| Response Time | ms | ≤ 1400 ms |

### Example Profiles

**Healthy Server**
```python
cpu_usage=35, memory_usage=45, disk_io=30, temperature=55,
network_errors=3, uptime_hours=720, error_log_count=8,
process_count=150, swap_usage=20, response_time_ms=200
```

**Critical Server**
```python
cpu_usage=96, memory_usage=94, disk_io=88, temperature=91,
network_errors=35, uptime_hours=7200, error_log_count=72,
process_count=470, swap_usage=88, response_time_ms=1750
```

---

## Dataset

The model is trained on `server_failure_dataset.csv`, which contains labelled server telemetry samples with a binary `failure` target column.

Key columns: `server_id`, `cpu_usage`, `memory_usage`, `disk_io`, `temperature`, `network_errors`, `uptime_hours`, `error_log_count`, `process_count`, `swap_usage`, `response_time_ms`, `failure`

---

## Model Performance

The model is evaluated using:
- **Classification Report** (Precision, Recall, F1-score per class)
- **Confusion Matrix**
- **ROC-AUC** (used in GridSearchCV)
- **F1-score** (used for optimal threshold selection)

Run `server_failure_prediction.py` to see the full evaluation output for your dataset.

---

## Dashboard Preview

The **ServerGuard AI** dashboard features:

- **Left panel**: 10 interactive sliders with colour-coded threshold indicators
- **Top-right**: Animated circular gauge showing live failure probability
- **Mid-right**: Risk level badge with recommended action
- **Bottom-right**: Threshold alerts list and engineered features breakdown

---

*Built with Python & Scikit-learn · Dashboard powered by Vanilla JS*
