# Stock Market Analysis & Prediction Program

## Overview

This is a comprehensive Python-based stock market analysis tool that combines data acquisition, statistical hypothesis testing, machine learning predictions, technical analysis, and investment performance metrics. The program provides a complete workflow for analyzing historical stock data and making informed predictions about future price movements.

---

## Table of Contents

1. [Features](#features)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Program Workflow](#program-workflow)
6. [Module Descriptions](#module-descriptions)
7. [Statistical Methodology](#statistical-methodology)
8. [Visualizations](#visualizations)
9. [Metrics Explained](#metrics-explained)
10. [Limitations & Considerations](#limitations--considerations)

---

## Features

### Data Management
- **Automated data download** from Yahoo Finance API
- **Data validation** and error handling
- **CSV export** functionality for offline analysis
- **Date range customization**

### Statistical Analysis
- **Normality testing** (Shapiro-Wilk test)
- **Correlation significance testing** (Pearson correlation)
- **Hypothesis testing** to validate regression assumptions
- **Distribution analysis** (skewness, kurtosis)

### Machine Learning
- **Multiple Linear Regression** model for price prediction
- **Train-test split** with temporal ordering preserved
- **Cross-validation** ready framework
- **Feature engineering** (excludes target from features to prevent data leakage)

### Technical Analysis
- **Simple Moving Averages** (SMA) - customizable windows
- **Golden Cross detection** (bullish signal)
- **Death Cross detection** (bearish signal)
- **Trend analysis** (short-term and long-term)
- **Support/resistance identification**

### Risk & Performance Metrics
- **Return calculations** (daily, cumulative, annualized)
- **Volatility measurements** (daily and annualized)
- **Maximum drawdown** analysis
- **Value at Risk (VaR)** at 95% confidence
- **Sharpe Ratio** (risk-adjusted returns)
- **Sortino Ratio** (downside risk focus)
- **Win rate** and gain/loss ratios

### Visualizations
- **15+ professional charts** covering all aspects of analysis
- **Price and volume plots**
- **Moving average overlays** with trading signals
- **Prediction vs actual comparisons**
- **Returns distribution histograms**
- **Drawdown charts**
- **Rolling performance metrics**

---

## Requirements

### Python Libraries

```python
yfinance          # Yahoo Finance data download
matplotlib        # Data visualization
pandas            # Data manipulation
scikit-learn      # Machine learning models
scipy             # Statistical tests
numpy             # Numerical computations
```

### Installation

```bash
pip install yfinance matplotlib pandas scikit-learn scipy numpy
```

---

## Usage

### Basic Command Line Usage

```bash
python Multiple-Linear-Regression-Analysis.py <TICKER> [START_DATE] [END_DATE]
```

### Examples

```bash
# Download Apple stock from 2010 to present
python Multiple-Linear-Regression-Analysis.py AAPL

# Download Google stock with custom date range
python Multiple-Linear-Regression-Analysis.py GOOGL 2015-01-01 2024-12-31

# Download Tesla stock from 2020
python Multiple-Linear-Regression-Analysis.py TSLA 2020-01-01
```

### Parameters

- **TICKER** (required): Stock ticker symbol (e.g., AAPL, MSFT, TSLA)
- **START_DATE** (optional): Start date in YYYY-MM-DD format (default: 2010-01-01)
- **END_DATE** (optional): End date in YYYY-MM-DD format (default: today)

---

## Program Workflow

The program executes in the following sequence:

```
1. Download Stock Data
   ↓
2. Visualize Price & Volume
   ↓
3. Run Hypothesis Tests
   ↓
4. Train ML Model & Predict
   ↓
5. Calculate Moving Averages
   ↓
6. Analyze Returns & Risk
   ↓
7. Display Results & Charts
```

---

## Module Descriptions

### 1. Data Acquisition Module

**Function:** `download_stock_data(ticker, start, end, filename, save_csv)`

**Purpose:** Connects to Yahoo Finance and downloads historical OHLCV data.

**Features:**
- Error handling for invalid tickers
- Empty data validation
- Automatic CSV saving
- Date range verification
- Progress reporting

**Returns:** pandas DataFrame with columns:
- Date (index)
- Open
- High
- Low
- Close
- Volume

**Example Output:**
```
Downloading data for AAPL...
Downloaded 3728 rows from 2010-01-04 to 2024-12-20
Saved dataset to AAPL.csv
```

---

### 2. Data Visualization Module

**Function:** `plot_stock_data(data, ticker)`

**Purpose:** Creates professional dual-panel charts showing price and volume trends.

**Features:**
- Currency formatting ($) on price axis
- Volume displayed in millions (M)
- Date formatting (YYYY-MM)
- Grid lines for readability
- Color-coded styling
- Rotated date labels

**Outputs:**
- **Top panel:** Closing price over time
- **Bottom panel:** Trading volume bars

---

### 3. Hypothesis Testing Module

**Function:** `run_hypothesis_tests(data, ticker)`

**Purpose:** Validates statistical assumptions required for multiple linear regression.

#### Tests Performed:

**A. Normality Test (Shapiro-Wilk)**
- **Null Hypothesis (H₀):** Closing prices are normally distributed
- **Significance Level:** α = 0.05
- **Interpretation:**
  - p > 0.05: Fail to reject H₀ (prices appear normal)
  - p < 0.05: Reject H₀ (prices are NOT normal)

**B. Correlation Significance Tests (Pearson)**
- Tests each feature against next-day closing price
- Features tested: Open, High, Low, Volume
- **Null Hypothesis (H₀):** No linear correlation exists
- **Significance Level:** α = 0.05

**Why This Matters:**
- Linear regression assumes linear relationships between variables
- Non-normal distributions may require transformations
- Insignificant correlations suggest features have no predictive power
- Helps identify which features to include/exclude

**Example Output:**
```
=== Hypothesis Testing for AAPL Multiple Linear Regression ===

1. Normality Test (Shapiro-Wilk) on Close Prices:
   Test Statistic = 0.9234
   p-value = 0.000001
   ➤ Reject H₀: Close prices are NOT normally distributed.

2. Pearson Correlation Significance (Features vs. Next-Day Close):
   Testing 3727 observations

   Open    : r =  0.9998, p = 0.000000  ✓ Significant
   High    : r =  0.9997, p = 0.000000  ✓ Significant
   Low     : r =  0.9998, p = 0.000000  ✓ Significant
   Volume  : r = -0.0543, p = 0.001234  ✓ Significant

3. Summary:
   Significant predictors (p < 0.05): Open, High, Low, Volume
```

---

### 4. Machine Learning Prediction Module

**Function:** `predict_future_price(data, ticker)`

**Purpose:** Trains a Multiple Linear Regression model to predict next-day closing prices.

#### Model Architecture:

**Features (X):**
- Open price
- High price
- Low price
- Volume

**Target (y):**
- Next day's closing price (shifted by -1)

**Key Design Decisions:**
- **No data leakage:** Current closing price is excluded from features
- **Temporal split:** Train-test split maintains time order (no shuffling)
- **Split ratio:** 80% training, 20% testing

#### Training Process:

```python
# Prepare features and target
X = [Open, High, Low, Volume]
y = Close.shift(-1)  # Tomorrow's close

# Split while preserving time order
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Train linear regression
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
```

#### Evaluation Metrics:

**1. RMSE (Root Mean Squared Error)**
- Measures average prediction error in dollars
- Penalizes large errors more heavily
- Lower is better

**2. R² Score (Coefficient of Determination)**
- Explains what % of variance the model captures
- Range: -∞ to 1.0
- 1.0 = perfect predictions
- 0.0 = model explains nothing
- Negative = worse than predicting the mean

**3. MAE (Mean Absolute Error)**
- Average absolute error in dollars
- Very interpretable: "Model is off by $X on average"
- Lower is better

#### Feature Coefficients:

The model learns coefficients that show how each feature impacts the prediction:

```
Feature Coefficients:
  Open    :     0.2500
  High    :     0.3500
  Low     :     0.3800
  Volume  :    -0.0001
  Intercept: 2.5000
```

**Interpretation:** 
- High and Low prices have the strongest influence
- Volume has minimal impact (very small coefficient)

#### Next-Day Prediction:

Uses the most recent data to predict tomorrow's closing price:

```
Prediction for Next Trading Day:
  Current Close Price:    $180.50
  Predicted Close Price:  $181.75
  Expected Change:        +$1.25 (+0.69%)
```

---

### 5. Moving Average Analysis Module

**Function:** `calculate_moving_averages(data, ticker, windows=[20, 50, 200])`

**Purpose:** Performs technical analysis using Simple Moving Averages (SMA).

#### Moving Averages Calculated:

**1. 20-Day SMA (Short-term)**
- Captures recent price momentum
- Used for short-term trading signals

**2. 50-Day SMA (Medium-term)**
- Represents intermediate trend
- Common for swing trading

**3. 200-Day SMA (Long-term)**
- Indicates overall market trend
- Used for position trading and investing

#### Trading Signals:

**Golden Cross (Bullish Signal):**
- Occurs when 50-day SMA crosses above 200-day SMA
- Indicates potential upward trend
- Often used as a buy signal

**Death Cross (Bearish Signal):**
- Occurs when 50-day SMA crosses below 200-day SMA
- Indicates potential downward trend
- Often used as a sell signal

#### Trend Analysis:

**Long-term Trend:**
- Bullish if 50-day > 200-day
- Bearish if 50-day < 200-day

**Short-term Trend:**
- Bullish if 20-day > 50-day
- Bearish if 20-day < 50-day

#### Example Output:

```
=== Moving Average Analysis for AAPL ===

Golden Crosses (Bullish): 3
Death Crosses (Bearish):  2
Last Golden Cross: 2023-11-15
Last Death Cross:  2023-08-22

Current Price Position:
Current Close: $180.50
  20-day SMA: $178.25 (above by 1.26%)
  50-day SMA: $175.80 (above by 2.67%)
  200-day SMA: $168.50 (above by 7.12%)

Trend Analysis:
  📈 Long-term trend: BULLISH (50-day > 200-day)
  📈 Short-term trend: BULLISH (20-day > 50-day)
```

---

### 6. Returns & Risk Analysis Module

**Function:** `calculate_returns(data, ticker, risk_free_rate=0.04)`

**Purpose:** Calculates comprehensive investment performance and risk metrics.

#### Return Metrics:

**1. Daily Returns**
```python
Daily_Return = (Close_today - Close_yesterday) / Close_yesterday
```

**2. Cumulative Returns**
```python
Cumulative_Return = (1 + Daily_Return₁) × (1 + Daily_Return₂) × ... - 1
```

**3. Annualized Returns**
```python
Annualized_Return = (End_Price / Start_Price)^(1 / Years) - 1
```

#### Risk Metrics:

**1. Daily Volatility**
- Standard deviation of daily returns
- Measures day-to-day price fluctuations

**2. Annualized Volatility**
```python
Annualized_Volatility = Daily_Volatility × √252
```
- 252 = typical trading days per year
- Higher volatility = higher risk

**3. Maximum Drawdown**
- Largest peak-to-trough decline
- Measures worst-case loss scenario
- Example: -30% means investment lost 30% from peak before recovering

**4. Value at Risk (VaR)**
- 95% confidence level
- "On 95% of days, losses won't exceed X%"
- Example: VaR = -2.5% means 5% chance of losing more than 2.5% in a day

#### Risk-Adjusted Returns:

**1. Sharpe Ratio**
```python
Sharpe_Ratio = (Annualized_Return - Risk_Free_Rate) / Annualized_Volatility
```

**Interpretation:**
- Measures return per unit of risk
- Higher is better
- < 1.0 = Poor
- 1.0 - 2.0 = Good
- 2.0 - 3.0 = Very Good
- > 3.0 = Excellent

**2. Sortino Ratio**
```python
Sortino_Ratio = (Annualized_Return - Risk_Free_Rate) / Downside_Volatility
```

**Interpretation:**
- Similar to Sharpe but only considers downside volatility
- Better for investors who care more about losses than gains
- Higher values indicate better risk-adjusted returns

#### Trading Statistics:

- **Best Day:** Highest single-day return
- **Worst Day:** Lowest single-day return
- **Win Rate:** % of days with positive returns
- **Average Gain:** Mean return on positive days
- **Average Loss:** Mean return on negative days
- **Gain/Loss Ratio:** Average gain ÷ Average loss

#### Distribution Analysis:

- **Mean:** Average daily return
- **Median:** Middle value of returns
- **Skewness:** Asymmetry of distribution
  - Negative = more extreme losses than gains
  - Positive = more extreme gains than losses
- **Kurtosis:** "Tail thickness"
  - High = more extreme events than normal distribution

---

## Visualizations

### Total: 15 Charts Across All Modules

#### Module 2: Data Visualization (2 charts)
1. **Closing Price Over Time**
   - Line chart with currency formatting
2. **Trading Volume**
   - Bar chart in millions

#### Module 4: ML Predictions (2 charts)
3. **Predicted vs Actual Scatter Plot**
   - Shows prediction accuracy
   - Includes perfect prediction line
4. **Time Series Predictions**
   - Overlays predictions on actual test data

#### Module 5: Moving Averages (2 charts)
5. **Price with Moving Averages**
   - Overlays 20, 50, 200-day SMAs
   - Marks Golden/Death crosses
6. **Distance from 50-day MA**
   - Shows overbought/oversold conditions

#### Module 6: Returns Analysis (5 charts)
7. **Cumulative Returns**
   - Shows total investment growth
8. **Daily Returns Bar Chart**
   - Green (positive) and red (negative) bars
9. **Returns Distribution Histogram**
   - Shows frequency of different return levels
10. **Drawdown Chart**
    - Visualizes risk periods
11. **Rolling 30-Day Returns**
    - Shows momentum over time

---

## Metrics Explained

### Performance Metrics

| Metric | Formula | Good Value | Interpretation |
|--------|---------|------------|----------------|
| **Total Return** | (End - Start) / Start | > 0% | Overall gain/loss |
| **Annualized Return** | (End/Start)^(1/years) - 1 | > 10% | Yearly average return |
| **RMSE** | √(Σ(actual - pred)² / n) | Lower | Avg prediction error |
| **R² Score** | 1 - (RSS / TSS) | > 0.7 | % variance explained |
| **MAE** | Σ\|actual - pred\| / n | Lower | Avg absolute error |

### Risk Metrics

| Metric | Formula | Good Value | Interpretation |
|--------|---------|------------|----------------|
| **Volatility** | σ(returns) × √252 | < 20% | Annual price variability |
| **Max Drawdown** | Min(cumulative - peak) | > -20% | Worst loss from peak |
| **VaR (95%)** | Percentile(returns, 5) | > -3% | 5% worst-case daily loss |
| **Sharpe Ratio** | (Return - RF) / σ | > 1.5 | Risk-adjusted return |
| **Sortino Ratio** | (Return - RF) / σ_down | > 2.0 | Downside risk-adjusted |

### Trading Metrics

| Metric | Calculation | Good Value | Interpretation |
|--------|-------------|------------|----------------|
| **Win Rate** | Positive days / Total days | > 55% | % of profitable days |
| **Gain/Loss Ratio** | Avg Gain / Avg Loss | > 1.5 | Wins vs losses magnitude |
| **Best Day** | Max(daily returns) | N/A | Largest single-day gain |
| **Worst Day** | Min(daily returns) | N/A | Largest single-day loss |

---

## Limitations & Considerations

### Statistical Limitations

1. **Normality Assumption Violations**
   - Financial data rarely follows normal distribution
   - Returns often have "fat tails" (more extreme events)
   - Linear regression assumes normality of residuals

2. **Autocorrelation**
   - Stock prices are time-series data
   - Yesterday's price influences today's price
   - Violates independence assumption of linear regression

3. **Non-linearity**
   - Relationships between features may be non-linear
   - Volume impact varies at different price levels
   - Market regimes change over time

### Model Limitations

1. **Linear Regression Simplicity**
   - Assumes linear relationships
   - Cannot capture complex market dynamics
   - No consideration of market sentiment, news, or events

2. **Data Leakage Prevention**
   - Current close price excluded from features
   - Prevents unrealistic prediction accuracy
   - Real-world predictions must use only prior information

3. **Feature Engineering**
   - Limited to OHLCV data
   - No technical indicators as features
   - No fundamental analysis included

### Market Considerations

1. **Past Performance ≠ Future Results**
   - Historical patterns may not repeat
   - Market conditions change
   - Black swan events unpredictable

2. **Transaction Costs Ignored**
   - No consideration of:
     - Brokerage fees
     - Bid-ask spreads
     - Slippage
     - Taxes

3. **Liquidity Assumptions**
   - Assumes you can buy/sell at any price
   - May not hold for low-volume stocks
   - Large orders can move the market

### Technical Analysis Limitations

1. **Moving Averages Lag**
   - Based on historical data
   - Signals appear after trend begins
   - Can give false signals in choppy markets

2. **Overfitting Risk**
   - Parameters (20, 50, 200) are conventional
   - May not be optimal for all stocks
   - Past optimal parameters may not work in future

---

## Best Practices

### For Using This Program

1. **Use Multiple Timeframes**
   - Analyze different date ranges
   - Compare recent vs long-term performance
   - Watch for regime changes

2. **Combine Multiple Indicators**
   - Don't rely on single metric
   - Use ML predictions + technical analysis + risk metrics
   - Consider fundamental analysis separately

3. **Regular Retraining**
   - Market conditions evolve
   - Retrain models periodically
   - Monitor prediction accuracy over time

4. **Risk Management**
   - Never invest based solely on this tool
   - Use stop-losses
   - Diversify portfolio
   - Only risk what you can afford to lose

5. **Backtesting**
   - Test strategies on historical data
   - Walk-forward analysis
   - Consider multiple market conditions

---

## Future Enhancements

### Potential Improvements

1. **Additional Models**
   - Random Forest
   - LSTM (Long Short-Term Memory networks)
   - ARIMA for time series
   - Ensemble methods

2. **More Features**
   - Technical indicators (RSI, MACD, Bollinger Bands)
   - Sentiment analysis from news
   - Fundamental ratios (P/E, EPS)
   - Market indices correlation

3. **Advanced Analysis**
   - Portfolio optimization
   - Monte Carlo simulation
   - Regime detection
   - Multi-asset correlation

4. **Automation**
   - Scheduled data updates
   - Alert system for trading signals
   - Automated backtesting framework
   - API integration for live trading

---

## Conclusion

This program provides a comprehensive framework for stock market analysis, combining statistical rigor with practical trading insights. While it offers valuable analytical tools, users should remember that all financial predictions carry inherent uncertainty and should be used as part of a broader investment strategy that includes proper risk management and diversification.

**Disclaimer:** This tool is for educational and research purposes only. It does not constitute financial advice. Always consult with a qualified financial advisor before making investment decisions.
