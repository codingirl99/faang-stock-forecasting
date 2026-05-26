# faang-stock-forecasting

# faang-stock-forecasting

## One-Sentence Summary  
This project predicts Apple (AAPL) stock closing prices using Linear Regression, Random Forest, and LSTM models trained on a FAANG stock market dataset from Kaggle.

---

## Overview  
This project explores stock price forecasting using traditional machine learning and deep learning models on historical FAANG stock market data from Kaggle. The dataset contains stock prices and technical indicators such as moving averages, RSI, MACD, Bollinger Bands, and volatility metrics. The problem is treated as a time-series regression task where the models predict the next-day closing price of Apple (AAPL) stock using the previous 20 trading days of market data. Linear Regression, Random Forest, and LSTM models are trained and evaluated using MAE and RMSE metrics. Among the tested models, Linear Regression achieved the strongest performance with the lowest prediction error (RMSE ≈ 3.89), while the LSTM model showed moderate forecasting performance and Random Forest produced the highest prediction error.

---

## Summary of Work Done

### Data Exploration
- **Type:** Tabular dataset (CSV file)  
- **Features:** Date, Ticker, Open, High, Low, Close, Volume, SMA_7, SMA_21, EMA_12, EMA_26, RSI_14, MACD, MACD_Signal, Bollinger_Upper, Bollinger_Lower, Daily_Return, Volatility_7d  
- **Target:** Next-day Apple stock closing price  
- **Size:** ~14,964 rows  
- **Stock Used for Modeling:** Apple (AAPL)  
- **Split:** 80% training, 20% testing using chronological time-series splitting  

---

### Data Visualization  

- **Stock Price Trends**
    - Apple, Amazon, Meta, Netflix, and Google all show strong long-term upward growth trends.
    - Some companies exhibit significantly higher volatility during market downturns.
    - Stock prices sharply fluctuate during high-volatility periods.

<!-- Stock Price Trend Plot -->
<figure>
<img src="/visualizations/faang_stock_price_trends.png" alt="FAANG Stock Price Trends">
<figcaption></figcaption>
</figure>

---

- **Daily Return Distributions**
    - Most daily returns cluster close to zero.
    - All FAANG companies show occasional extreme positive and negative returns.
    - Return distributions appear roughly centered but contain heavy tails.

<!-- Daily Return Distribution Plot -->
<figure>
<img src="/visualizations/faang_daily_return_distribution.png" alt="Daily Return Distribution">
<figcaption></figcaption>
</figure>

---

- **Correlation Heatmap**
    - Most FAANG stocks show positive correlations with one another.
    - Correlation strength varies across companies.

<!-- Correlation Heatmap -->
<figure>
<img src="/visualizations/faang_correlation_heatmap.png" alt="FAANG Correlation Heatmap">
<figcaption></figcaption>
</figure>

---

- **Rolling Volatility Analysis**
    - Volatility spikes appear during periods of major market movement.
    - Netflix and Meta exhibit larger volatility swings compared to Apple.
    - Rolling volatility highlights changing market risk over time.

<!-- Rolling Volatility Plot -->
<figure>
<img src="/visualizations/faang_rolling_volatility.png" alt="Rolling Volatility">
<figcaption></figcaption>
</figure>

---

### Preprocessing  
- Converted Date column to datetime format  
- Sorted stock data chronologically  
- Filtered dataset to Apple (AAPL) stock only  
- Created 20-day rolling time-series sequences  
- Applied MinMaxScaler normalization to stock price features  
- Reshaped data for:
  - traditional machine learning models
  - LSTM sequence modeling  

---

### Problem Setup  
This is a time-series regression problem.

- **Input:** Previous 20 trading days of stock features  
- **Output:** Next-day Apple stock closing price  

**Models Used:**
- Linear Regression  
- Random Forest Regressor  
- Long Short-Term Memory (LSTM) Neural Network  

**Settings:**
- Linear Regression: default scikit-learn settings  
- Random Forest: `n_estimators=100`  
- LSTM:
  - 2 LSTM layers
  - 64 and 32 hidden units
  - Dropout regularization
  - Adam optimizer
  - 40 epochs
  - Batch size = 32

---

### Training  
- Implemented using Python, scikit-learn, and TensorFlow  
- Models trained on CPU  
- Traditional ML models used flattened stock sequences  
- LSTM used sequential 3D time-series input data  
- Minimal hyperparameter tuning performed  

---

### Model Performance

| Model | MAE | RMSE |
|---|---|---|
| Linear Regression | 2.70 | 3.89 |
| Random Forest | 31.59 | 40.03 |
| LSTM | 14.17 | 16.16 |

- **Linear Regression** achieved the strongest performance with the lowest prediction error.  
- **LSTM** captured overall stock trends but did not outperform Linear Regression.  
- **Random Forest** struggled with sequential stock forecasting and produced the highest error values.  

---

### Prediction Comparison

<!-- Prediction Comparison Plot -->
<figure>
<img src="/visualizations/aapl_prediction_comparison.png" alt="Prediction Comparison">
<figcaption></figcaption>
</figure>

- Linear Regression predictions remained closest to actual stock prices overall.
- LSTM predictions captured trend direction but showed larger deviations.
- Random Forest predictions displayed the weakest generalization performance.

---

### Feature Importance

This project focused primarily on model comparison rather than feature interpretation.  

The following stock price features were used:
- Open
- High
- Low
- Close

Future work could explore:
- technical indicator importance
- SHAP analysis
- feature selection techniques
- additional engineered financial features

---
 
### Conclusions  
- Traditional machine learning models can perform competitively on stock forecasting tasks.  
- Linear Regression significantly outperformed both Random Forest and LSTM on this dataset.  
- LSTM models may require larger datasets and additional tuning to outperform simpler regression methods.  
- Random Forest struggled to model sequential stock behavior effectively.  
- Stock forecasting remains highly challenging due to market volatility and non-stationary price movements.  

---

### Future Improvements  
- Predict stock returns instead of raw prices  
- Train separate models for each FAANG company  
- Add more technical indicators as input features  
- Apply hyperparameter tuning  
- Implement walk-forward validation  
- Test GRU or Transformer-based architectures  
- Experiment with XGBoost or LightGBM models  

---

## How to Reproduce Results  

1. Clone the repository  

2. Download dataset from Kaggle:  
   https://www.kaggle.com/datasets/vishardmehta/faang-stock-market-data-with-technical-indicators  

3. Place dataset in `/data` folder  

4. Install dependencies:
   - pandas  
   - numpy  
   - matplotlib  
   - seaborn  
   - scikit-learn  
   - tensorflow  

5. Run notebooks in order:
   - `1_data_eda.ipynb` → exploratory data analysis and stock visualizations  
   - `2_stock_prediction_models.ipynb` → preprocessing, Linear Regression, Random Forest, and LSTM modeling  

Note: MUST RUN notebook 2 in Google Colab as TensorFlow is preinstalled, so no additional TensorFlow setup is required.