# faang-stock-forecasting

## One-Sentence Summary  
This project predicts Apple (AAPL) stock closing prices using Linear Regression, Random Forest, and LSTM models trained on FAANG+ stock market data from Kaggle.

---

## Overview  
This project explores stock price forecasting using traditional machine learning and deep learning models on a FAANG+ stock market dataset from Kaggle. The dataset includes historical stock prices for major technology companies such as Apple, Microsoft, Amazon, Google, Meta, and NVIDIA, along with technical indicators like moving averages, RSI, MACD, and Bollinger Bands. The problem is framed as a time-series regression task where models predict the next-day closing price of Apple (AAPL) using the previous 20 trading days of market data. Linear Regression, Random Forest, and LSTM models are trained and compared using MAE and RMSE. Linear Regression achieved the best performance, while LSTM performed moderately and Random Forest showed the weakest results due to difficulty handling sequential patterns.

---

## Summary of Work Done

### Data Exploration
- **Type:** Tabular time-series dataset (CSV file)  
- **Source:** Kaggle FAANG+ Stock Market Dataset  
- **Companies Included:** AAPL, MSFT, AMZN, GOOGL/GOOG, META, NVDA  
- **Features:** Date, Ticker, Open, High, Low, Close, Volume, SMA_7, SMA_21, EMA_12, EMA_26, RSI_14, MACD, MACD_Signal, Bollinger_Upper, Bollinger_Lower, Daily_Return, Volatility_7d  
- **Target:** Next-day closing price (AAPL model prediction target)  
- **Size:** ~14,000+ rows (varies after filtering and preprocessing)  
- **Modeling Focus:** Apple (AAPL) only  
- **Split:** 80% training, 20% testing using chronological time-series split  

---

### Data Visualization  

- **Stock Price Trends**
    - Major tech stocks show long-term upward trends.
    - Each stock exhibits different levels of volatility over time.
    - Market-wide fluctuations are visible during high volatility periods.

<!-- Stock Price Trend Plot -->
<figure>
<img src="/visualizations/faang_stock_price_trends.png" alt="Stock Price Trends">
<figcaption></figcaption>
</figure>

---

- **Daily Return Distributions**
    - Returns are centered around zero for all stocks.
    - Occasional extreme positive and negative returns exist.
    - Distributions show heavy tails, indicating market volatility.

<!-- Daily Return Distribution Plot -->
<figure>
<img src="/visualizations/faang_daily_return_distribution.png" alt="Daily Returns">
<figcaption></figcaption>
</figure>

---

- **Correlation Heatmap**
    - Stocks are strongly positively correlated across the tech sector.
    - Market movements are highly interconnected between companies.

<!-- Correlation Heatmap -->
<figure>
<img src="/visualizations/faang_correlation_heatmap.png" alt="Correlation Heatmap">
<figcaption></figcaption>
</figure>

---

- **Rolling Volatility Analysis**
    - Volatility changes over time and increases during market events.
    - Some stocks exhibit higher risk than others.
    - Rolling volatility captures dynamic market behavior.

<!-- Rolling Volatility Plot -->
<figure>
<img src="/visualizations/faang_rolling_volatility.png" alt="Volatility">
<figcaption></figcaption>
</figure>

---

## Preprocessing  
- Converted Date column to datetime format  
- Sorted data chronologically  
- Filtered dataset to Apple (AAPL) for modeling  
- Created 20-day sliding window sequences  
- Applied MinMax scaling to normalize features  
- Reshaped data for:
  - Linear Regression & Random Forest (flattened input)
  - LSTM (3D sequential input)

---

## Problem Setup  

This is a time-series regression problem.

- **Input:** Previous 20 trading days of stock features  
- **Output:** Next-day Apple closing price  

### Models Used:
- Linear Regression  
- Random Forest Regressor  
- LSTM Neural Network  

### Model Settings:
- Linear Regression: default scikit-learn settings  
- Random Forest: 100 estimators  
- LSTM:
  - 2 LSTM layers (64, 32 units)
  - Dropout layers (0.2)
  - Adam optimizer
  - 40 epochs
  - Batch size = 32  

---

## Training  
- Implemented in Python using scikit-learn and TensorFlow  
- Chronological train-test split (no data leakage)  
- Traditional ML models trained on flattened sequences  
- LSTM trained on sequential time-series data  
- Minimal hyperparameter tuning applied  

---

## Model Performance  

| Model | MAE | RMSE |
|---|---|---|
| Linear Regression | 2.70 | 3.89 |
| Random Forest | 31.59 | 40.03 |
| LSTM | 14.17 | 16.16 |

---

## Prediction Comparison  

- Linear Regression produced the closest predictions to actual stock prices.  
- LSTM captured general trends but had higher deviation.  
- Random Forest struggled with sequential learning and performed poorly overall.  

---

## Feature Importance  

This project focuses on model comparison rather than interpretability.

Input features used:
- Open  
- High  
- Low  
- Close  

Future work could explore:
- technical indicator importance  
- SHAP analysis  
- feature selection techniques  

---

## Conclusions  

- Linear Regression performed best despite being the simplest model.  
- LSTM showed moderate performance but likely needs more tuning or data.  
- Random Forest struggled with sequential stock behavior.  
- Stock forecasting remains a noisy and difficult regression problem.  

---

## Future Improvements  

- Predict stock returns instead of raw prices  
- Train separate models per company  
- Add more technical indicators  
- Apply hyperparameter tuning  
- Use walk-forward validation  
- Experiment with GRU or Transformer models  
- Try XGBoost / LightGBM  

---

## How to Reproduce Results  

### 1. Clone repository
```bash
git clone <your-repo-url>
```

---

### 2. Download dataset  
https://www.kaggle.com/datasets/vishardmehta/faang-stock-market-data-with-technical-indicators  

Place it in:
```
/data/faang_stock_prices.csv
```

---

### 3. Install dependencies (EDA only)
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

### 4. Run Notebook 1 (Locally)
- `1_data_eda.ipynb`  
Includes:
- data cleaning  
- visualizations  
- exploratory analysis  

---

### 5. Run Notebook 2 (Google Colab REQUIRED)

Open:
https://colab.research.google.com

Steps:
- Upload `2_stock_prediction_models.ipynb`
- Upload `/data/faang_stock_prices.csv`
- Run all cells sequentially  

### Why Colab is required:
- TensorFlow dependency (LSTM model)
- Easier environment setup
- Faster execution (no local installation issues)

---

## Notebook Structure  

- `1_data_eda.ipynb` → Exploratory Data Analysis (local)  
- `2_stock_prediction_models.ipynb` → ML + LSTM modeling (Google Colab)  