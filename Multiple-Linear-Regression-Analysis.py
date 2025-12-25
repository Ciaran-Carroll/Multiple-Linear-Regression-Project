## Current workflow
## 1. Download stock data
## 2. Plot price and volume
## 3. Analyze moving averages
## 4. Calculate returns and risk metrics
## 5. Run hypothesis tests
## 6. Compare multiple ML models
## 7. Predict future prices




## 1. Download stock data from yFinance
## 2. Plot hystorical prices with the corresponding volume data
## 3. Runs hypothesis Test on the dataset
## 4. Trains a Multiple Linear Regression model on the dataset
## 5. Plot predictions

## Features:
## - Automatic feature scaling where needed
## - Cross-validation (5-fold) for robust evaluation
## - Multiple metrics: RMSE, R², MAE, training time
## - Best model selection automatically
## - Next-day prediction using the best model
## - Comprehensive visualizations (5 plots)
## Visualizations Include:
##
## RMSE comparison (lower is better)
## R² comparison (higher is better)
## MAE comparison (lower is better)
## Training time comparison
## Predictions vs. actual for top 3 models

import yfinance as yf
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from matplotlib.ticker import FuncFormatter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
from scipy.stats import shapiro, pearsonr
import numpy as np
from scipy import stats


## Data Acquistion
##
## The program connects to Yahoo Finance (yFinance) and downloads historical
##    (Open, High, Low, Close, Volume) data for your specified stock ticker
##    and date range. Data is automatically validated, cleaned, and optionally
##    saved to CSV for future reference.

def download_stock_data(ticker, start="2010-01-01", end=None, filename=None, save_csv=True):
    """
    Downloads historical stock price data for a given ticker symbol.
    Parameters:
        ticker (str): Stock ticker (e.g., 'AAPL', 'GOOGL')
        start (str): Start date in YYYY-MM-DD format
        end (str): End date in YYYY-MM-DD format (None = today)
        filename (str): Output CSV filename (default = '<ticker>.csv')
        save_csv (bool): Whether to save data to CSV (default = True)

    Features:
        Handles errors
        Checks for empty data
        Optionally saves to csv

    Returns:
        DataFrame: Historical data or None if error
    """
    print(f"Downloading data for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start, end=end)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

    if data.empty:
        print("No data found. Check the ticker symbol.")
        return None

    print(f"Downloaded {len(data)} rows from {data.index[0].date()} to {data.index[-1].date()}")

    if save_csv:
        if not filename:
            filename = f"{ticker}.csv"
        data.to_csv(filename)
        print(f"Saved dataset to {filename}")

    return data

## Stock Data Visualisation

def plot_stock_data(data, ticker):
    """
    Plots the Close price and Volume of the stock dataset.
    Includes properly formatted date axes to provide immediate insight
    into price trends and trading activity.

    Parameters:
        data (DataFrame): Stock data with 'Close' and 'Volume' columns
        ticker (str): Stock ticker symbol for the title

    Features:
        Two subplots: closing price (top) and volume (bottom)
        Currency formatting on y-axis ($)
        Volume displayed in millions (M)
        Date formatting on x-axis (YYYY-MM)
        Rotated date labels for readability
        Color-coded and styled lines
        Detailed dataset info printed
        Professional appearance with grid and proper sizing
    """
    if data is None or data.empty:
        print("No data to plot.")
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                     gridspec_kw={'height_ratios': [3, 1]})

    # Plot closing price
    ax1.plot(data.index, data["Close"], color='#1f77b4', linewidth=2)
    ax1.set_title(f"{ticker} Closing Price Over Time", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Closing Price (USD)", fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Format y-axis as currency
    def currency(x, pos):
        return f'${x:,.0f}'
    ax1.yaxis.set_major_formatter(FuncFormatter(currency))

    # Plot volume
    ax2.bar(data.index, data["Volume"], color='gray', alpha=0.5, width=1)
    ax2.set_ylabel("Volume", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Format volume axis
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x/1e6:.0f}M'))

    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())

    # Rotate date labels
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

    # Print dataset info
    print(f"Dataset contains {len(data)} days from {data.index[0].date()} to {data.index[-1].date()}")


## Multiple linear regression relies on several assumptions
## 1. Linearity - Each independent variable should have a linear relationship with the dependent variable.
## 2. Normality - Residuals (or the target distribution) should follow a normal distribution.
## 3. No multicolinearity - Features should not be too highly correlated with each other.
## 4. Independence - Observations should be independent (especially important in time series).

## Stock data often violates statistical assumptions:
## - Prices are autocorrelated (yesterday predicts today).
## - Volume and close price may be non-linearly related.
## - Financial data rarely follows a perfectly normal distribution.

## Hypothesis testing tells you:
## - Is it safe to use linear regression?
## - Which features should be included or removed?
## - Are transformations (log, differencing) needed?
## This is especially important before training a predictive model.


## Hypothesis testing helps you identify non-linear relationships, detect multicolinearity, checks if the target is
## normally distributed, understand whether linear regression is statistically appropriate and know whether data
## transformations are needed

## Key improvements made:
## - Fixed deprecated method - Uses dropna() instead of fillna(method="bfill")
## - Removed data leakage - Excludes "Close" from features when predicting next-day close
## - Better data alignment - Properly handles the shifted target variable
## - Returns results - Returns a DataFrame with test results for further analysis
## - Clearer output - Better formatting with checkmarks and summary
## - Added observation count - Shows how many data points were tested








## The function properly tests whether Open, High, Low, and Volume can predict the next day's closing price!

## Step 3: Hypothesis Testing
## Before applying machine learning, the program validates key assumptions required
## for regression analysis:
## 1 - Normality Test (Shapiro-Wilk)
##      Tests whether closing prices follow a normal distribution. Financial
##      data often violates this assumption, which the program flags for
##      consideration when interpreting model results.
## 2 - Correlation Significance Testing (Pearson)
##     Tests each feature (Open, High, Low, Volume) for significant linear
##     correlation with next-day closing price. This identifies which features
##     are statistically meaningful predictors (p < 0.05).
## Results inform whether linear models are appropriarte for price prediction
## or if a non-linear model would peform better.

def run_hypothesis_tests(data, ticker):
    """
    Performs hypothesis testing required before running multiple linear regression:
    1. Normality test for target variable (Close)
    2. Pearson correlation significance for all features vs. target

    Parameters:
        data (DataFrame): Stock data with OHLCV (Open, High, Low, Close and Volume) columns
        ticker (str): Stock ticker symbol

    Returns:
        DataFrame: Test Results
    """

    print(f"\n=== Hypothesis Testing for {ticker} Multiple Linear Regression ===")

    # 1. Normality Test on Close Prices
    close_prices = data["Close"].dropna()
    stat, p_value = shapiro(close_prices)

    print("\n1. Normality Test (Shapiro-Wilk) on Close Prices:")
    print(f"   Test Statistic = {stat:.4f}")
    print(f"   p-value = {p_value:.6f}")

    if p_value > 0.05:
        print("   ➤ Fail to reject H₀: Close prices appear normally distributed.")
    else:
        print("   ➤ Reject H₀: Close prices are NOT normally distributed.")
        print("     (Common in financial time series — consider log-transform or differencing.)")

    # 2. Correlation Significance Tests
    print("\n2. Pearson Correlation Significance (Features vs. Next-Day Close):")

    # Prepare features and target
    features = ["Open", "High", "Low", "Volume"]  # Exclude Close to avoid data leakage
    target = data["Close"].shift(-1)  # Next-day close (prediction target)

    # Create aligned dataframe (drop last row where target is NaN)
    df_test = data[features].copy()
    df_test["Target"] = target
    df_test = df_test.dropna()

    print(f"   Testing {len(df_test)} observations\n")

    results = []
    for feature in features:
        x = df_test[feature]
        y = df_test["Target"]
        r, p = pearsonr(x, y)

        significance = "✓ Significant" if p < 0.05 else "✗ Not significant"
        results.append({
            'Feature': feature,
            'Correlation': r,
            'P-value': p,
            'Significance': significance
        })

        print(f"   {feature:8s}: r = {r:7.4f}, p = {p:.6f}  {significance}")

    # 3. Summary
    print("\n3. Summary:")
    significant_features = [r['Feature'] for r in results if r['P-value'] < 0.05]

    if significant_features:
        print(f"   Significant predictors (p < 0.05): {', '.join(significant_features)}")
    else:
        print("   No features show significant linear correlation with next-day close.")
        print("   Consider feature engineering or non-linear models.")

    print("\nHypothesis testing complete.\n")

    return pd.DataFrame(results)

## Key improvements
## - Fixed data leakage - Removed "Close" from features (can't use today's close to predict tomorrow's)
## - Better metrics - Added RMSE, R², and MAE for comprehensive evaluation
## - Feature coefficients - Shows which features are most important
## - Visualizations - Two plots showing prediction quality
## - More informative output - Shows expected price change and percentage
## - Returns model and metrics - Allows further analysis
## - Better formatting - Clearer, more professional output

## The function now provides a much clearer picture of model performance and makes more realistic predictions!

def predict_future_price(data, ticker):
    """
    Uses multiple linear regression to predict the next day's closing price.

    Parameters:
        data (DataFrame): Stock data with OHLCV columns
        ticker (str): Stock ticker symbol

    Returns:
        tuple: (predicted_price, model, metrics_dict)
    """
    if data is None or data.empty:
        print("No data for prediction.")
        return None, None, None

    print(f"\n=== Training Multiple Linear Regression Model for {ticker} ===\n")

    # Features (Open, High, Low, Volume) - excluding Close to avoid data leakage
    features = data[["Open", "High", "Low", "Volume"]].copy()

    # Target (next day's Close)
    target = data["Close"].shift(-1)

    # Drop last row (NaN target)
    features = features[:-1]
    target = target[:-1]

    print(f"Training set size: {len(features)} observations")
    print(f"Features used: {list(features.columns)}\n")

    # Split data (80/20, maintaining time order)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, shuffle=False
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Calculate metrics
    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)

    # Print model performance
    print("Model Performance:")
    print(f"  Train RMSE: ${train_rmse:.2f}")
    print(f"  Test RMSE:  ${test_rmse:.2f}")
    print(f"  Train R²:   {train_r2:.4f}")
    print(f"  Test R²:    {test_r2:.4f}")
    print(f"  Test MAE:   ${test_mae:.2f}")

    # Print feature coefficients
    print("\nFeature Coefficients:")
    for feature, coef in zip(features.columns, model.coef_):
        print(f"  {feature:8s}: {coef:10.4f}")
    print(f"  Intercept: {model.intercept_:.4f}")

    # Visualize predictions vs actual
    plt.figure(figsize=(14, 6))

    # Plot 1: Test predictions vs actual
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, test_predictions, alpha=0.5, s=20)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', lw=2, label='Perfect prediction')
    plt.xlabel('Actual Price ($)', fontsize=11)
    plt.ylabel('Predicted Price ($)', fontsize=11)
    plt.title(f'{ticker} - Predicted vs Actual (Test Set)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Time series of predictions
    plt.subplot(1, 2, 2)
    test_dates = y_test.index
    plt.plot(test_dates, y_test.values, label='Actual', linewidth=2, alpha=0.7)
    plt.plot(test_dates, test_predictions, label='Predicted', linewidth=2, alpha=0.7)
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Price ($)', fontsize=11)
    plt.title(f'{ticker} - Test Set Predictions Over Time', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # Predict the next day's price using the final row of available data
    print("\n" + "="*60)
    next_input = data[["Open", "High", "Low", "Volume"]].iloc[-1:]
    next_day_price = model.predict(next_input)[0]

    last_close = data["Close"].iloc[-1]
    price_change = next_day_price - last_close
    percent_change = (price_change / last_close) * 100

    print(f"\nPrediction for Next Trading Day:")
    print(f"  Current Close Price:    ${last_close:.2f}")
    print(f"  Predicted Close Price:  ${next_day_price:.2f}")
    print(f"  Expected Change:        ${price_change:+.2f} ({percent_change:+.2f}%)")
    print("="*60 + "\n")

    # Store metrics
    metrics = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mae': test_mae
    }

    return next_day_price, model, metrics

def plot_predictions(y_test, predictions, ticker):
    """
    Plots actual vs predicted closing prices for the test set.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label="Actual Close")
    plt.plot(y_test.index, predictions, label="Predicted Close", linestyle="--")

    plt.title(f"{ticker} — Actual vs Predicted Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def predict_future_price(data, ticker):
    """
    Uses multiple linear regression to predict the next day's closing price.

    Parameters:
        data (DataFrame): Stock data with OHLCV columns
        ticker (str): Stock ticker symbol

    Returns:
        tuple: (predicted_price, model, metrics_dict)
    """
    if data is None or data.empty:
        print("No data for prediction.")
        return None, None, None

    print(f"\n=== Training Multiple Linear Regression Model for {ticker} ===\n")

    # Features (Open, High, Low, Volume) - excluding Close to avoid data leakage
    features = data[["Open", "High", "Low", "Volume"]].copy()

    # Target (next day's Close)
    target = data["Close"].shift(-1)

    # Drop last row (NaN target)
    features = features[:-1]
    target = target[:-1]

    print(f"Training set size: {len(features)} observations")
    print(f"Features used: {list(features.columns)}\n")

    # Split data (80/20, maintaining time order)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, shuffle=False
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Calculate metrics
    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)

    # Print model performance
    print("Model Performance:")
    print(f"  Train RMSE: ${train_rmse:.2f}")
    print(f"  Test RMSE:  ${test_rmse:.2f}")
    print(f"  Train R²:   {train_r2:.4f}")
    print(f"  Test R²:    {test_r2:.4f}")
    print(f"  Test MAE:   ${test_mae:.2f}")

    # Print feature coefficients
    print("\nFeature Coefficients:")
    for feature, coef in zip(features.columns, model.coef_):
        print(f"  {feature:8s}: {coef:10.4f}")
    print(f"  Intercept: {model.intercept_:.4f}")

    # Visualise predictions vs actual
    plt.figure(figsize=(14, 6))

    # Plot 1: Test predictions vs actual
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, test_predictions, alpha=0.5, s=20)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', lw=2, label='Perfect prediction')
    plt.xlabel('Actual Price ($)', fontsize=11)
    plt.ylabel('Predicted Price ($)', fontsize=11)
    plt.title(f'{ticker} - Predicted vs Actual (Test Set)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Time series of predictions
    plt.subplot(1, 2, 2)
    test_dates = y_test.index
    plt.plot(test_dates, y_test.values, label='Actual', linewidth=2, alpha=0.7)
    plt.plot(test_dates, test_predictions, label='Predicted', linewidth=2, alpha=0.7)
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Price ($)', fontsize=11)
    plt.title(f'{ticker} - Test Set Predictions Over Time', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # Predict the next day's price using the final row of available data
    print("\n" + "="*60)
    next_input = data[["Open", "High", "Low", "Volume"]].iloc[-1:]
    next_day_price = model.predict(next_input)[0]

    last_close = data["Close"].iloc[-1]
    price_change = next_day_price - last_close
    percent_change = (price_change / last_close) * 100

    print(f"\nPrediction for Next Trading Day:")
    print(f"  Current Close Price:    ${last_close:.2f}")
    print(f"  Predicted Close Price:  ${next_day_price:.2f}")
    print(f"  Expected Change:        ${price_change:+.2f} ({percent_change:+.2f}%)")
    print("="*60 + "\n")

    # Store metrics
    metrics = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mae': test_mae
    }

    return next_day_price, model, metrics

## 1. R² (Coefficient of Determination)
##
## Measures how much of the variation in closing prices the model explains.
##
## 1.0 = perfect
## 0.0 = explains nothing
##
## Can be negative (model is worse than predicting the mean!)
##
## 2. RMSE (Root Mean Squared Error)
##
## Penalizes large errors more heavily.
## Useful for financial data where spikes matter.
##
## 3. MAE (Mean Absolute Error)
##
## Average size of prediction error in USD.
## Very interpretable:
##
## MAE = 2.50 means “on average the model is $2.50 off”.
##
## 4. Residual Plot
##
## Shows:
## - Bias
## - Trend violations
## - Heteroscedasticity
## - Non-linearity
##
## If residuals look like random noise → model assumptions satisfied.


## Key Features:
##
## Multiple Moving Averages - Calculates 20, 50, and 200-day SMAs (customizable)
## Golden Cross & Death Cross Detection
##
## Golden Cross: 50-day MA crosses above 200-day MA (bullish signal)
## Death Cross: 50-day MA crosses below 200-day MA (bearish signal)
##
##
## Current Position Analysis - Shows how current price compares to each MA
## Trend Analysis - Identifies short-term and long-term trends
## Visualizations:
##
## Main chart with price and all MAs
## Trading signals marked with green (buy) and red (sell) triangles
## Distance from 50-day MA subplot showing overbought/oversold conditions

def calculate_moving_averages(data, ticker, windows=[20, 50, 200]):
    """
    Calculate and plot simple moving averages (SMA) with buy/sell signals.

    Parameters:
        data (DataFrame): Stock data with 'Close' column
        ticker (str): Stock ticker symbol
        windows (list): List of MA window periods (default: [20, 50, 200])

    Returns:
        DataFrame: Data with added MA columns and signals
    """
    if data is None or data.empty:
        print("No data for moving average analysis.")
        return None

    print(f"\n=== Moving Average Analysis for {ticker} ===\n")

    # Create a copy to avoid modifying original data
    df = data.copy()

    # Calculate moving averages
    for window in windows:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()

    # Identify Golden Cross and Death Cross (if 50 and 200 MA exist)
    if 50 in windows and 200 in windows:
        df['SMA_50_above_200'] = df['SMA_50'] > df['SMA_200']
        df['Golden_Cross'] = (df['SMA_50_above_200'] == True) & (df['SMA_50_above_200'].shift(1) == False)
        df['Death_Cross'] = (df['SMA_50_above_200'] == False) & (df['SMA_50_above_200'].shift(1) == True)

        # Count signals
        golden_crosses = df['Golden_Cross'].sum()
        death_crosses = df['Death_Cross'].sum()

        print(f"Golden Crosses (Bullish): {golden_crosses}")
        print(f"Death Crosses (Bearish):  {death_crosses}")

        # Most recent signal
        if golden_crosses > 0:
            last_golden = df[df['Golden_Cross']].index[-1]
            print(f"Last Golden Cross: {last_golden.date()}")
        if death_crosses > 0:
            last_death = df[df['Death_Cross']].index[-1]
            print(f"Last Death Cross:  {last_death.date()}")

    # Current position relative to MAs
    print("\nCurrent Price Position:")
    current_price = df['Close'].iloc[-1]
    print(f"Current Close: ${current_price:.2f}")

    for window in windows:
        ma_value = df[f'SMA_{window}'].iloc[-1]
        if pd.notna(ma_value):
            diff = current_price - ma_value
            pct_diff = (diff / ma_value) * 100
            position = "above" if diff > 0 else "below"
            print(f"  {window}-day SMA: ${ma_value:.2f} ({position} by {abs(pct_diff):.2f}%)")

    # Determine trend
    print("\nTrend Analysis:")
    if 50 in windows and 200 in windows:
        if df['SMA_50'].iloc[-1] > df['SMA_200'].iloc[-1]:
            print("  📈 Long-term trend: BULLISH (50-day > 200-day)")
        else:
            print("  📉 Long-term trend: BEARISH (50-day < 200-day)")

    if 20 in windows and 50 in windows:
        if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1]:
            print("  📈 Short-term trend: BULLISH (20-day > 50-day)")
        else:
            print("  📉 Short-term trend: BEARISH (20-day < 50-day)")

    # Visualize
    print("\nGenerating visualisation...")
    plot_moving_averages(df, ticker, windows)

    return df


def plot_moving_averages(df, ticker, windows):
    """
    Plot closing price with moving averages and trading signals.

    Parameters:
        df (DataFrame): Data with Close and SMA columns
        ticker (str): Stock ticker symbol
        windows (list): List of MA window periods
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                     gridspec_kw={'height_ratios': [3, 1]})

    # Plot 1: Price and Moving Averages
    ax1.plot(df.index, df['Close'], label='Close Price',
             color='black', linewidth=2, alpha=0.7)

    colors = ['blue', 'orange', 'red', 'green', 'purple']
    for i, window in enumerate(windows):
        color = colors[i % len(colors)]
        ax1.plot(df.index, df[f'SMA_{window}'],
                label=f'{window}-day SMA',
                color=color, linewidth=1.5, alpha=0.8)

    # Add Golden Cross and Death Cross markers
    if 'Golden_Cross' in df.columns and 'Death_Cross' in df.columns:
        golden_cross_dates = df[df['Golden_Cross']].index
        death_cross_dates = df[df['Death_Cross']].index

        if len(golden_cross_dates) > 0:
            ax1.scatter(golden_cross_dates,
                       df.loc[golden_cross_dates, 'Close'],
                       color='green', marker='^', s=200,
                       label='Golden Cross', zorder=5, edgecolors='darkgreen', linewidths=2)

        if len(death_cross_dates) > 0:
            ax1.scatter(death_cross_dates,
                       df.loc[death_cross_dates, 'Close'],
                       color='red', marker='v', s=200,
                       label='Death Cross', zorder=5, edgecolors='darkred', linewidths=2)

    ax1.set_title(f'{ticker} - Price with Moving Averages', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Distance from 50-day MA (if available)
    if 50 in windows:
        distance_pct = ((df['Close'] - df['SMA_50']) / df['SMA_50']) * 100
        ax2.plot(df.index, distance_pct, color='blue', linewidth=1.5)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.fill_between(df.index, distance_pct, 0,
                         where=(distance_pct >= 0), alpha=0.3, color='green', label='Above MA')
        ax2.fill_between(df.index, distance_pct, 0,
                         where=(distance_pct < 0), alpha=0.3, color='red', label='Below MA')
        ax2.set_title('Distance from 50-day MA (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Deviation (%)', fontsize=11)
        ax2.set_xlabel('Date', fontsize=11)
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("Visualisation complete.\n")

###########################################################
##
## Key Metrics Calculated:
##
## Return Metrics
##
## Total return
## Annualized return
## Daily and cumulative returns
##
##
## Risk Metrics
##
## Daily and annualized volatility
## Maximum drawdown (worst peak-to-trough decline)
## Value at Risk (VaR) at 95% confidence
##
##
## Risk-Adjusted Returns
##
## Sharpe Ratio (most common measure)
## Sortino Ratio (focuses on downside risk)
##
##
## Trading Statistics
##
## Best and worst single days
## Win rate (% of positive days)
## Average gain vs. average loss
## Gain/loss ratio
##
##
## Distribution Analysis
##
## Mean, median, skewness, kurtosis
##
##
##
## Visualizations (5 plots):
##
## Cumulative returns over time
## Daily returns bar chart
## Returns distribution histogram
## Drawdown chart (shows risk periods)
## Rolling 30-day returns

def calculate_returns(data, ticker, risk_free_rate=0.04):
    """
    Calculate daily, cumulative, and annualized returns with risk metrics.

    Parameters:
        data (DataFrame): Stock data with 'Close' column
        ticker (str): Stock ticker symbol
        risk_free_rate (float): Annual risk-free rate for Sharpe ratio (default: 4%)

    Returns:
        DataFrame: Data with added return columns and metrics dictionary
    """
    if data is None or data.empty:
        print("No data for returns analysis.")
        return None, None

    print(f"\n=== Returns Analysis for {ticker} ===\n")

    # Create a copy
    df = data.copy()

    # Calculate daily returns
    df['Daily_Return'] = df['Close'].pct_change()

    # Calculate cumulative returns
    df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1

    # Calculate log returns (for more accurate statistics)
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Remove NaN values for calculations
    returns = df['Daily_Return'].dropna()

    # --- Calculate Key Metrics ---

    # Total return
    total_return = df['Cumulative_Return'].iloc[-1]

    # Annualized return
    trading_days = len(returns)
    years = trading_days / 252  # 252 trading days per year
    start_price = df['Close'].iloc[0]
    end_price = df['Close'].iloc[-1]
    annualised_return = (end_price / start_price) ** (1 / years) - 1

    # Volatility (annualized standard deviation)
    daily_volatility = returns.std()
    annualised_volatility = daily_volatility * np.sqrt(252)

    # Sharpe Ratio (risk-adjusted return)
    excess_return = annualised_return - risk_free_rate
    sharpe_ratio = excess_return / annualised_volatility if annualised_volatility > 0 else 0

    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Best and Worst Days
    best_day = returns.max()
    best_day_date = returns.idxmax()
    worst_day = returns.min()
    worst_day_date = returns.idxmin()

    # Win Rate
    positive_days = (returns > 0).sum()
    total_days = len(returns)
    win_rate = positive_days / total_days

    # Average gain/loss
    avg_gain = returns[returns > 0].mean()
    avg_loss = returns[returns < 0].mean()

    # Sortino Ratio (downside risk-adjusted return)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino_ratio = excess_return / downside_std if downside_std > 0 else 0

    # Value at Risk (VaR) - 95% confidence
    var_95 = np.percentile(returns, 5)

    # --- Print Results ---

    print("=" * 60)
    print("RETURN METRICS")
    print("=" * 60)
    print(f"Total Return:              {total_return:.2%}")
    print(f"Annualised Return:         {annualised_return:.2%}")
    print(f"Start Price:               ${start_price:.2f}")
    print(f"End Price:                 ${end_price:.2f}")
    print(f"Time Period:               {years:.2f} years ({trading_days} trading days)")

    print("\n" + "=" * 60)
    print("RISK METRICS")
    print("=" * 60)
    print(f"Daily Volatility:          {daily_volatility:.4f} ({daily_volatility:.2%})")
    print(f"Annualised Volatility:     {annualised_volatility:.2%}")
    print(f"Maximum Drawdown:          {max_drawdown:.2%}")
    print(f"Value at Risk (95%):       {var_95:.2%}")

    print("\n" + "=" * 60)
    print("RISK-ADJUSTED RETURNS")
    print("=" * 60)
    print(f"Sharpe Ratio:              {sharpe_ratio:.4f}")
    print(f"Sortino Ratio:             {sortino_ratio:.4f}")
    print(f"  (Risk-free rate: {risk_free_rate:.1%})")

    print("\n" + "=" * 60)
    print("TRADING STATISTICS")
    print("=" * 60)
    print(f"Best Day:                  {best_day:.2%} on {best_day_date.date()}")
    print(f"Worst Day:                 {worst_day:.2%} on {worst_day_date.date()}")
    print(f"Win Rate:                  {win_rate:.2%} ({positive_days}/{total_days} days)")
    print(f"Average Gain:              {avg_gain:.2%}")
    print(f"Average Loss:              {avg_loss:.2%}")
    print(f"Gain/Loss Ratio:           {abs(avg_gain/avg_loss):.2f}x" if avg_loss != 0 else "N/A")
    print("=" * 60)

    # --- Distribution Analysis ---
    print("\nRETURNS DISTRIBUTION:")
    print(f"  Mean:    {returns.mean():.4f} ({returns.mean():.2%})")
    print(f"  Median:  {returns.median():.4f} ({returns.median():.2%})")
    print(f"  Skewness: {stats.skew(returns):.4f}")
    print(f"  Kurtosis: {stats.kurtosis(returns):.4f}")

    # Visualize
    print("\nGenerating visualisations...")
    plot_returns_analysis(df, ticker, returns, max_drawdown)

    # Store metrics
    metrics = {
        'total_return': total_return,
        'annualised_return': annualised_return,
        'annualised_volatility': annualised_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'var_95': var_95
    }

    return df, metrics


def plot_returns_analysis(df, ticker, returns, max_drawdown):
    """
    Create comprehensive visualisation of returns analysis.

    Parameters:
        df (DataFrame): Data with return columns
        ticker (str): Stock ticker symbol
        returns (Series): Daily returns
        max_drawdown (float): Maximum drawdown value
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Cumulative Returns
    ax1 = fig.add_subplot(gs[0, :])
    cumulative_pct = df['Cumulative_Return'] * 100
    ax1.plot(df.index, cumulative_pct, color='#2E86AB', linewidth=2)
    ax1.fill_between(df.index, cumulative_pct, 0, alpha=0.3, color='#2E86AB')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.set_title(f'{ticker} - Cumulative Returns Over Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Return (%)', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Add annotation for total return
    total_return_pct = cumulative_pct.iloc[-1]
    ax1.annotate(f'Total: {total_return_pct:.1f}%',
                xy=(df.index[-1], total_return_pct),
                xytext=(-80, 20), textcoords='offset points',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Plot 2: Daily Returns
    ax2 = fig.add_subplot(gs[1, 0])
    colors = ['green' if x > 0 else 'red' for x in returns]
    ax2.bar(returns.index, returns * 100, color=colors, alpha=0.6, width=1)
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.set_title('Daily Returns', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Return (%)', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Returns Distribution (Histogram)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(returns * 100, bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
    ax3.axvline(returns.mean() * 100, color='red', linestyle='--',
                linewidth=2, label=f'Mean: {returns.mean()*100:.2f}%')
    ax3.axvline(0, color='black', linestyle='-', linewidth=1)
    ax3.set_title('Returns Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Daily Return (%)', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Drawdown
    ax4 = fig.add_subplot(gs[2, 0])
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    ax4.fill_between(drawdown.index, drawdown * 100, 0, color='red', alpha=0.5)
    ax4.plot(drawdown.index, drawdown * 100, color='darkred', linewidth=1.5)
    ax4.set_title(f'Drawdown (Max: {max_drawdown:.2%})', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Drawdown (%)', fontsize=10)
    ax4.set_xlabel('Date', fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Rolling 30-day Returns
    ax5 = fig.add_subplot(gs[2, 1])
    rolling_returns = df['Daily_Return'].rolling(window=30).sum() * 100
    ax5.plot(rolling_returns.index, rolling_returns, color='#F18F01', linewidth=2)
    ax5.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax5.fill_between(rolling_returns.index, rolling_returns, 0,
                     where=(rolling_returns >= 0), alpha=0.3, color='green')
    ax5.fill_between(rolling_returns.index, rolling_returns, 0,
                     where=(rolling_returns < 0), alpha=0.3, color='red')
    ax5.set_title('Rolling 30-Day Returns', fontsize=12, fontweight='bold')
    ax5.set_ylabel('30-Day Return (%)', fontsize=10)
    ax5.set_xlabel('Date', fontsize=10)
    ax5.grid(True, alpha=0.3)

    plt.suptitle(f'{ticker} - Comprehensive Returns Analysis',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.show()

    print("Visualisation complete.\n")



#def evaluation_regression_model(y_test, predictions, ticker):
#    """
#    Evaluates the regression model using:
#    - R² Score
#    - RMSE (Root Mean Squared Error)
#    - MAE (Mean Absolute Error)
#    - Residual plot
##    """
#
#    print("\n=== Regression Model Validation ===")
#
#    # --- Evaluation Metrics ---
#    r2 = r2_score(y_test, predictions)
#    rmse = mean_squared_error(y_test, predictions)
#    mae = mean_absolute_error(y_test, predictions)
#
#    print(f"R² Score: {r2:.4f}")
#    print(f"RMSE: {rmse:.4f}")
#    print(f"MAE:  {mae:.4f}")
#
#    # --- Residual Plot ---
#    residuals = y_test - predictions
#
#    plt.figure(figsize=(10, 5))
#    plt.scatter(y_test.index, residuals)
#    plt.axhline(0, color='red', linestyle='--')
#    plt.title(f"{ticker} — Residual Plot")
#    plt.xlabel("Date")
#    plt.ylabel("Residual (Actual - Prediction)")
#    plt.grid(True)
#    plt.tight_layout()
#    plt.show()

#    print("\nEvaluation complete.\n")


## Moving Average Analysis - Very practical for trading
## Returns Analysis - Essential for investment decisions
## Model Comparison - Improves prediction accuracy
## Data Summary Dashboard - Great overview of the stock

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_stock.py <TICKER> [START] [END]")
        print("Example: python download_stock.py AAPL 2015-01-01 2024-12-31")
        sys.exit(1)

    ticker = sys.argv[1]
    start = sys.argv[2] if len(sys.argv) > 2 else "2010-01-01"
    end = sys.argv[3] if len(sys.argv) > 3 else None

    data = download_stock_data(ticker, start, end)
    plot_stock_data(data, ticker)
    run_hypothesis_tests(data, ticker)

    # Contains the DataFrame with the test results for further analysis
    results_df = run_hypothesis_tests(data, ticker)

    predicted_price, model, metrics = predict_future_price(data, ticker)

    # With default windows (20, 50, 200-day MAs)
    df_with_mas = calculate_moving_averages(data, ticker)

    # Custom windows
    df_with_mas = calculate_moving_averages(data, ticker, windows=[10, 30, 100])

    # Basic usage with default risk-free rate (4%)
    df_returns, metrics = calculate_returns(data, ticker)

    # Custom risk-free rate (e.g., 5%)
    df_returns, metrics = calculate_returns(data, ticker, risk_free_rate=0.05)

    # Access specific metrics
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")


    ##predict_future_price(data, ticker)
    #next_day, y_test, predictions = predict_future_price(data, ticker)
    #plot_predictions(y_test, predictions, ticker)

    #next_day_price, y_test, predictions = predict_future_price(data, ticker)
    #plot_predictions(y_test, predictions, ticker)
    #validate_regression_model(y_test, predictions, ticker)
