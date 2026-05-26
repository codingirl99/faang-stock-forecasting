# faang-stock-forecasting

## One-Sentence Summary  
This project predicts Apple (AAPL) stock closing prices using Linear Regression, Random Forest, and LSTM models trained on FAANG+ stock market data with technical indicators.

---

## Overview  
This project explores time-series stock price forecasting using machine learning and deep learning techniques. The dataset contains historical stock market data for major technology companies including Apple, Amazon, Google, Meta, Microsoft, and NVIDIA, along with technical indicators such as moving averages, RSI, MACD, and volatility metrics. The goal is to predict the next-day closing price of Apple (AAPL) using the previous 20 days of stock data. Three models—Linear Regression, Random Forest, and LSTM—are trained and compared using MAE and RMSE. Linear Regression performs best, followed by LSTM, while Random Forest performs the weakest due to difficulty handling sequential patterns.

---

## Dataset Information  
- **Source:** Kaggle FAANG+ Stock Market Dataset  
- **Rows:** 14,964  
- **Columns:** 19  
- **Stocks Included:** AAPL, AMZN, GOOGL, META, MSFT, NVDA  
- **Target Variable:** Next_Day_Close  
- **No missing values or duplicates**

---

## Data Visualization  

### Stock Closing Prices Over Time  
- Stocks show long-term upward trends.
- Different companies exhibit different volatility patterns.
- META and MSFT show larger stock closing prices over time compared to other companies present.

<figure>
<img src="/visualizations/faang_stock_closing_prices.png">
</figure>

---

### Daily Return Distributions  
- Returns are centered around zero.
- Heavy tails indicate frequent volatility spikes.
- MSFT having high density and NVDA having lowest.

<figure>
<img src="/visualizations/faang_daily_return_distribution.png">
</figure>

---

### Correlation Heatmap  
- Strong positive correlation between all tech stocks.
- Stocks tend to move together in the market.

<figure>
<img src="/visualizations/faang_correlation_heatmap.png">
</figure>

---

### 20-Day Rolling Volatility  
- Volatility changes over time.
- Some stocks are consistently more volatile than others, including META and NVDA.

<figure>
<img src="/visualizations/faang_rolling_volatility.png">
</figure>

---

### Boxplot of Daily Returns  
- Most returns are concentrated around zero.
- Outliers represent rare market shocks, significantly shown in the META and MSFT boxplots.

<figure>
<img src="/visualizations/faang_boxplot_daily_returns.png">
</figure>

---

## Preprocessing  
- Converted Date column to datetime  
- Sorted dataset by Ticker and Date  
- Created Daily_Return using percentage change  
- Filtered dataset to Apple (AAPL) only for modeling  
- Created 20-day sliding window sequences  
- Applied MinMax scaling to features and target  
- Reshaped data for:
  - Machine learning models (flattened input)
  - LSTM model (3D sequential input)

---

## Problem Setup  

This is a supervised time-series regression problem.

- **Input:** Previous 20 days of Apple stock features (Open, High, Low, Close)  
- **Output:** Next-day Apple closing price  

---

## Models Used  

- Linear Regression  
- Random Forest Regressor  
- LSTM Neural Network  

---

## Model Architecture (LSTM)  
- LSTM layer (64 units)  
- Dropout (0.2)  
- LSTM layer (32 units)  
- Dropout (0.2)  
- Dense output layer  
- Optimizer: Adam  
- Loss: Mean Squared Error  
- Epochs: 40  
- Batch size: 32  

---

## Model Performance  

| Model | MAE | RMSE |
|------|------|------|
| Linear Regression | 2.70 | 3.89 |
| Random Forest | 31.59 | 40.03 |
| LSTM | 14.17 | 16.16 |

---

## Model Comparison Visualization  

- The chart below compares actual vs predicted prices for all models.

<figure>
<img src="/visualizations/faang_model_comparison.png">
</figure>

---

## Key Observations  

- Linear Regression surprisingly performed best despite being the simplest model.
- LSTM captured overall trends but had higher error.
- Random Forest struggled due to inability to model sequential dependencies.
- Stock prices are highly noisy, making prediction inherently difficult.

---

## Prediction Sample (First 20 Test Points)

| Actual | Linear Regression | Random Forest | LSTM |
|------|------------------|--------------|------|
| 185.11 | 182.67 | 181.86 | 187.68 |
| 184.11 | 185.49 | 182.98 | 186.42 |
| 185.92 | 184.76 | 183.39 | 184.95 |
| 187.53 | 186.28 | 186.74 | 184.21 |
| 187.64 | 186.64 | 187.24 | 183.90 |

---

## Conclusions  

- Linear Regression performed best overall in this forecasting task.  
- LSTM provided reasonable trend learning but needs further tuning.  
- Random Forest is not well-suited for sequential stock prediction.  
- Technical indicators alone are not enough to fully capture market behavior.  

---

## Future Improvements  

- Predict returns instead of raw prices  
- Add more feature engineering (lag features, ratios)  
- Use walk-forward validation  
- Try GRU or Transformer models  
- Tune LSTM architecture and hyperparameters  
- Train separate models per stock instead of only AAPL  

---

## How to Reproduce Results  

### 1. Clone Repository
```bash
git clone <your-repo-url>
```

---

### 2. Download Dataset  
https://www.kaggle.com/datasets/vishardmehta/faang-stock-market-data-with-technical-indicators  

Place into:
```
/data/faang_stock_prices.csv
```

---

### 3. Run Notebook 1 (EDA - Local)
Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn
```

Run:
```
1_exploratory_data_analysis.ipynb
```

Includes:
- data inspection  
- visualizations  
- statistical summary  

---

### 4. Run Notebook 2 (Google Colab REQUIRED)

Open:
https://colab.research.google.com

Steps:
- Upload `2_data_preprocessing_and_modeling.ipynb`
- Create new folder called ('data') and upload dataset (`faang_stock_prices.csv`) into folder
- Run all cells sequentially  

### Why Colab is required:
- TensorFlow dependency for LSTM
- Faster execution
- Easier environment setup
```

---

## Notebook Structure  

- `1_exploratory_data_analysis.ipynb` → EDA + visualizations  
- `2_data_preprocessing_and_modeling.ipynb` → ML + LSTM models (Colab)