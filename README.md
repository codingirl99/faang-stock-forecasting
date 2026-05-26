# faang-stock-forecasting

## One-Sentence Summary  
This project predicts Apple (AAPL) stock closing prices using Linear Regression, Random Forest, and LSTM models trained on a FAANG stock market dataset from Kaggle.

---

## Overview  
This project explores stock price forecasting using traditional machine learning and deep learning models on historical FAANG stock market data from Kaggle. The dataset contains stock prices and technical indicators such as moving averages, RSI, MACD, Bollinger Bands, and volatility metrics. The problem is treated as a time-series regression task where the models predict the next-day closing price of Apple (AAPL) stock using the previous 20 trading days of market data. Linear Regression, Random Forest, and LSTM models are trained and evaluated using MAE and RMSE metrics. Among the tested models, Linear Regression achieved the strongest performance with the lowest prediction error (RMSE ≈ 3.89), while the LSTM model showed moderate forecasting performance and Random Forest produced the highest prediction error.

---
## Summary of Work Done

### Data Exploration
**Type:** Tabular dataset (CSV file)
**Features:** Date, Ticker, Open, High, Low, Close, Volume, SMA_7, SMA_21, EMA_12, EMA_26, RSI_14, MACD, MACD_Signal, Bollinger_Upper, Bollinger_Lower, Daily_Return, Volatility_7d
**Target:** Next-day Apple stock closing price
**Size:** ~14,964 rows
**Stock Used for Modeling:** Apple (AAPL)
**Split:** 80% training, 20% testing using chronological time-series splitting

---

### Data Visualization  
- **Stock Price Trends**
    - Apple, Amazon, Meta, Netflix, and Google all show strong long-term upward growth trends.
    - Some companies exhibit significantly higher volatility during market downturns.
    - Stock prices sharply fluctuate during high-volatility periods.
    <!-- Stock Price Trend Plot --> <figure> <img src="/visualizations/faang_stock_price_trends.png" alt="FAANG Stock Price Trends"> <figcaption></figcaption> </figure>

---

### Preprocessing  


---

### Problem Setup  


- **Input:** 
- **Output:** 

**Models Used:**
 

**Settings:**


---

### Training  

---

### Model Performance


---

### Feature Importance

---
 
### Conclusions  


---

### Future Improvements  

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
   - scikit-learn
   - 
5. Run notebooks in order:
   - 1_data_eda.ipynb -> exploratory data analysis 
   - 2_data_visualizations.ipynb -> upload instead to Google Colab for no installation of tensorflow, and run code with .data folder with faang_stock_prices.csv inside