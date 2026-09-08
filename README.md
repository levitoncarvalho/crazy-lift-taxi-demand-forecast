# 🚕 Crazy Lift Taxi: Hourly Demand Forecast
### *Predicting Airport Taxi Demand One Hour Ahead*

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-02569B?style=for-the-badge&logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2.2-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://crazy-lift-taxi-demand-forecast-carvalholevis.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)

<br>

**[🚀 Try the Live App](https://crazy-lift-taxi-demand-forecastgit.streamlit.app/)** &nbsp;|&nbsp; **[📓 View Full Notebook](https://github.com/levitoncarvalho/crazy-lift-taxi-demand-forecast/blob/main/notebooks/exploration_v1.ipynb)**

</div>

---

> ⚠️ **Disclaimer:** Crazy Lift Taxi is a **fictional taxi service** created exclusively for academic and portfolio purposes. This project was developed as part of a Data Science training program and is intended solely to demonstrate technical skills in time series forecasting, model comparison, and deployment. No real company, customer data, or business relationship is represented here.

---

## 🧩 The Business Problem

> *"The best time to plant a tree was 20 years ago. The second best time is now." (Chinese Proverb)*

**Crazy Lift Taxi** collects historical data on airport taxi orders. To attract more drivers during peak hours, the operations team needs to know, one hour in advance, how many ride requests to expect, so drivers can be positioned before demand spikes instead of reacting after the fact.

The strategic question: **can a model forecast next-hour demand accurately enough to guide real driver positioning?**

The project's success criterion is strict: **test RMSE must not exceed 48**.

### Why Does This Matter?

| Scenario | Impact |
|---|---|
| 🔴 No forecast | Drivers are dispatched reactively, missing peak demand windows |
| 🟡 Inaccurate forecast (RMSE > 48) | Understaffed peaks, idle drivers during quiet hours |
| 🟢 **LightGBM (Test RMSE = 41.65, well under target)** | **Drivers positioned ahead of demand spikes with confidence** |

---

## 📊 Final Results

<div align="center">

### 🏆 All three models beat the business target: LightGBM was the clear winner

</div>

| Model | Test RMSE | Target | Status |
|---|---|---|---|
| Linear Regression (baseline) | 45.22 | ≤ 48 | 🟡 Meets target, weakest fit |
| Random Forest (tuned) | 43.05 | ≤ 48 | 🟡 Meets target, room to improve |
| **LightGBM (tuned, deployed)** | **41.65** | **≤ 48** | ✅ **Best result, well under target** |

> **A test RMSE of 41.65 means the model's hourly demand estimate is off by roughly 42 rides on average**, comfortably inside the 48 ride tolerance the business set as its success bar.

---

## 🧠 End-to-End Data Science Workflow

```
📥 Raw Data (10-min intervals)  →  🔄 Hourly Resampling  →  🔎 Seasonal Decomposition  →  🛠️ Feature Engineering  →  🤖 Modeling (3 algorithms)  →  🎯 Evaluation  →  🚀 Deployment
```

### 1. 🔄 Resampling & Seasonal Analysis

- Raw data arrives in **10 minute intervals**; resampled to **hourly buckets** (`resample('1h').sum()`) as required by the business problem
- **4,416 hourly records** spanning March through August 2018
- `seasonal_decompose` splits the series into trend, seasonality, and residuals, revealing:
  - A **steady upward trend** in total demand over the six month window
  - A **strong daily seasonality**, with sharp drops in the early morning and pronounced spikes at night and in the early hours

### 2. 🛠️ Feature Engineering

Since standard regressors have no built in sense of time, the following features were engineered from the datetime index:

| Feature type | Description |
|---|---|
| Calendar features | `dayofweek`, `hour` (capture daily and weekly seasonality) |
| 24 hourly lags (`lag_1` … `lag_24`) | Order counts from the previous 24 hours |
| Rolling mean (12h) | Recent trend, computed with `shift()` **before** the rolling window to avoid data leakage |

The dataset was split **chronologically** (`shuffle=False`), with the **last 10%** reserved as the test set, never shuffled, to respect the time series nature of the problem.

---

### 3. 🤖 Modeling & Hyperparameter Tuning

- **Three algorithms compared:** Linear Regression (baseline), Random Forest, LightGBM
- **Optimization:** `RandomizedSearchCV` (`n_iter=20`) for Random Forest and LightGBM
- **Cross validation:** `TimeSeriesSplit` (3 splits), which respects chronological order and prevents future data from leaking into training, unlike standard K Fold
- **Primary metric:** RMSE (test set)

---

### 4. 🎯 Final Test Set Evaluation

```
Linear Regression    RMSE = 45.22  ✅  (≤ 48)
Random Forest         RMSE = 43.05  ✅  (≤ 48)
LightGBM               RMSE = 41.65  ✅  (≤ 48, winner)
```

LightGBM's sequential, residual focused learning made it the most capable of tracking the volatility and sharp spikes in early morning demand.

---

### 5. 🚀 Deployment

The tuned LightGBM model was serialized with `joblib`, together with its expected feature order, and deployed as an interactive **Streamlit web app**. Users can edit the last 24 hours of order counts and instantly get a forecast for the next hour.

**[➡️ Try the live app here](https://crazy-lift-taxi-demand-forecastgit.streamlit.app/)**

---

## 🗂️ Project Structure

```text
crazy-lift-taxi-demand-forecast/
│
├── 📂 data/
│   └── 📊 taxi.csv                                 # Raw 10 minute interval dataset
│
├── 📓 notebooks/
│   └── 📓 exploration_v1.ipynb                     # Full analysis + 3 model comparison
│
├── 🤖 models/
│   └── ⚙️ demand_model.joblib                      # Serialized best model + feature order
│
├── 🐍 src/
│   ├── 🐍 __init__.py                              # Package initialization
│   ├── 🐍 data.py                                  # Loading, resampling, feature engineering
│   ├── 🐍 train.py                                 # Tuning, training, saving
│   └── 🐍 predict.py                               # Model loading + next hour inference
│
├── 🌐 app.py                                       # Streamlit interactive app
├── ⚙️ config.py                                    # Centralized configuration
├── 🔁 main.py                                      # Pipeline entry point
├── 📦 requirements.txt                             # Python dependencies
├── 🚫 .gitignore                                   # Ignored files and folders
├── ⚖️ LICENSE                                      # MIT License
└── 📄 README.md                                    # Project documentation
```

---

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/levitoncarvalho/crazy-lift-taxi-demand-forecast.git
cd crazy-lift-taxi-demand-forecast

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app (a trained model already ships in models/)
streamlit run app.py

# Optional: retrain from scratch
python main.py
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.9+ |
| **Data Manipulation** | Pandas, NumPy |
| **Time Series** | Statsmodels (seasonal decomposition) |
| **Machine Learning** | Scikit-Learn, LightGBM |
| **Hyperparameter Tuning** | RandomizedSearchCV + TimeSeriesSplit |
| **Visualization** | Matplotlib |
| **Serialization** | Joblib |
| **Deployment** | Streamlit, Streamlit Community Cloud |

---

## 💡 Key Technical Takeaways

- **`TimeSeriesSplit` instead of standard K Fold:** cross validation on time series data must respect chronological order, or the model ends up training on "future" data it would never have access to in production.
- **`shift()` before `rolling()` prevents leakage:** a rolling mean computed without shifting first would let the current hour's own value influence its own feature, silently inflating validation scores.
- **Seasonal decomposition is worth doing before feature engineering:** visualizing trend and seasonality up front made it obvious that `hour` and `dayofweek` would be essential features, rather than discovering that through trial and error.
- **A shuffled train/test split would have been a critical bug:** `shuffle=False` was non negotiable here; a random split would leak future information into training and produce an unrealistically optimistic RMSE.
- The **modular code structure** (`src/`) keeps feature engineering, tuning, and inference independently testable and reusable between the training pipeline and the Streamlit app.

---

## 👨‍💻 Author

<div align="center">

**Leviton Lima Carvalho**
*Data Scientist | Machine Learning | Python*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-levitoncarvalho-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/levitoncarvalho/)
[![GitHub](https://img.shields.io/badge/GitHub-levitoncarvalho-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/levitoncarvalho)
[![Email](https://img.shields.io/badge/Email-levitoncarvalho@icloud.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:levitoncarvalho@icloud.com)

</div>

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more details.
