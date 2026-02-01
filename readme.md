# Hybrid Hospital Forecasting: Managing Scale Heterogeneity

**A robust forecasting pipeline that routes diverse hospital facilities to specialized models (Classical ETS vs. Segmented LightGBM) to balance accuracy and stability.**

---

## 📌 Project Overview
Hospital capacity planning is fundamentally a **scale-heterogeneity problem**. A "one-size-fits-all" machine learning model often fails because the signal-to-noise ratio varies drastically between a rural 50-bed clinic and a metropolitan 1000-bed trauma center.

In this project, I built a hierarchical forecasting pipeline to predict monthly hospital patient volume. My core finding was a **"winner-take-most" dynamic**: while a Segmented LightGBM model achieved state-of-the-art accuracy for small and medium hospitals, it catastrophically failed on large institutions due to over-sensitivity to recent volatility.

**The Solution:** I implemented a **Hybrid Model Routing Strategy** that assigns facilities to algorithms based on their volatility profile:
* **Small/Medium Facilities:** Route to **Segmented LightGBM** (High Accuracy).
* **Large Facilities:** Route to **Classical ETS** (High Robustness).

---

## 🔍 The Core Insight: Why Pure ML Failed
Early in my experiments, I observed that a global LightGBM model (trained on all series) was being "bullied" by high-volume outliers. Even after applying `log1p` transformations and segmenting the data into volume tertiles (Small, Medium, Large), the ML model struggled with the **Large** bucket.

### Failure Mode Analysis
Using SHAP (SHapley Additive exPlanations), I diagnosed the root cause:
* **Small Hospitals**: The model correctly learned to rely on `roll_mean_12` (annual seasonality), which is stable.
* **Large Hospitals**: The model over-indexed on `lag_1` (immediate past month). Because large hospitals are volatile, this caused the model to chase noise, leading to massive RMSE spikes when sudden shifts occurred.



*This classic Bias-Variance tradeoff dictated my architectural pivot: I accepted the higher bias of ETS for large hospitals to avoid the unacceptable variance of the ML model.*

---

## ⚙️ Methodology & Architecture

### 1. Data Integrity & Leakage Prevention
I implemented a strict **Rolling-Origin Backtest** validation scheme.
* **Constraint**: `train_end < test_start` is enforced for every split.
* **Cold Start**: I enforced a minimum 48-month training window to ensure stability before generating the first forecast.

### 2. The Modeling Pipeline
I tested three distinct strategies to arrive at the final hybrid architecture:

| Strategy | Approach | Outcome |
| :--- | :--- | :--- |
| **Baseline** | Seasonal Naive | MAE ~21.8. Provided a "floor" for performance. |
| **Classical** | Exponential Smoothing (ETS) | **Most Robust.** Reliable across all scales (RMSE ~21.8), but rarely the most accurate. |
| **ML Prototype** | Global LightGBM (Quantile) | **Unstable.** Great on average, but prone to massive outlier errors (RMSE ~57.4). |
| **Final Hybrid** | **Segmented Routing** | **Best of Both Worlds.** Uses ML for stable small/medium series and ETS for volatile large series. |

### 3. Automated Interpretation (DeepSeek Integration)
To simulate a real-world production environment where stakeholders need narratives, not just CSVs, I built a lightweight **GenAI Reporting Layer**.
* The pipeline aggregates raw metrics and SHAP values into a context-aware JSON packet.
* I use the **DeepSeek Reasoner** API to generate a "Failure Mode Report" that automatically flags degraded performance buckets without human intervention.

---

## 🚀 Results
The Hybrid approach yields a projected **30-40% improvement** over the pure ETS baseline by capitalizing on ML's strength in low-volatility regimes while mitigating its weakness in high-volatility ones.

| Segment | Best Model | MAE | RMSE | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Small** | Segmented ML | **3.6** | **4.8** | ML dominates (15x better than baseline). |
| **Medium** | Segmented ML | **6.3** | **8.6** | ML dominates. |
| **Large** | Classical ETS | **18.6** | **21.8** | ETS wins. ML failed (RMSE ~89) due to outlier sensitivity. |

---

## 🛠️ Tech Stack
* **Core**: Python 3.10, Pandas, NumPy
* **Modeling**: LightGBM (Gradient Boosting), Statsmodels (ETS)
* **Interpretation**: SHAP (TreeExplainer with additivity override)
* **GenAI**: OpenAI Client (accessing DeepSeek Reasoner)
* **DevOps**: Git for version control (data excluded via `.gitignore`)

---

## 🔮 Limitations & Future Improvements
While the hybrid routing strategy is effective, there are several areas I plan to explore further:

1.  **Hierarchical Reconciliation**: Currently, I forecast at the facility level. A "Bottom-Up" or "MinT" (Minimum Trace) reconciliation approach could enforce consistency between total system capacity and individual facility predictions.
2.  **Dynamic Routing**: Instead of hard-coding the Small/Large split based on volume, I want to implement a meta-learner that routes series based on **spectral entropy** (a measure of forecastability).
3.  **Covariate Integration**: The large hospital model likely suffers from omitted variable bias. Integrating external factors like regional flu trends or holiday schedules could help the ML model capture the signal it currently misses.

---

## 💻 Usage
Cloning the repo (Data not included due to size):
```bash
git clone [https://github.com/aarav2703/Hybrid-Hospital-Forecasting.git](https://github.com/aarav2703/Hybrid-Hospital-Forecasting.git)
cd Hybrid-Hospital-Forecasting
