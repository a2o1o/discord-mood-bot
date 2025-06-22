import datetime
import os
import json
from typing import List, Dict, Optional, Tuple
import pandas as pd
import requests
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import pipeline
import matplotlib.pyplot as plt
import sys
import time
import argparse
from finnhub_news_fetcher import fetch_news_finnhub_monthly, normalize_dates, fetch_news_finnhub_for_range
from dateutil.relativedelta import relativedelta
import numpy as np
import re
import requests_cache
import logging
import feedparser
import urllib.parse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache Yahoo Finance requests
requests_cache.install_cache("yahoo_news_cache", expire_after=86400)

# List of tickers to analyze
TICKERS = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'TSLA', 'BRK-B', 'META', 'LLY', 'AVGO']

# Default Finnhub API key
DEFAULT_FINNHUB_API_KEY = "d1bvh0pr01qre5aip1ngd1bvh0pr01qre5aip1o0"

def fetch_price_data(ticker: str, years: int = 5) -> pd.DataFrame:
    """
    Fetch historical price data from yfinance.
    
    Args:
        ticker: Stock symbol
        years: Number of years of data to fetch
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
        end_date = datetime.date.today()
        start_date = end_date - relativedelta(years=years)
        if start_date.year < 1:
            start_date = datetime.date(1, 1, 1)
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            print(f"Warning: No price data available for {ticker}")
            return pd.DataFrame()
        
        # Handle multi-index columns
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs(ticker, axis=1, level=1)
        
        data.index = pd.to_datetime(data.index)
        print(f"✓ Fetched {len(data)} days of price data for {ticker}")
        
        return data
    except Exception as e:
        print(f"Error fetching price data for {ticker}: {e}")
        return pd.DataFrame()

def fetch_news_google_rss(ticker, start_date, end_date):
    """
    Returns list of headlines for `ticker` between start_date and end_date
    via Google News RSS with a 5-year filter.
    """
    # URL-encode ticker and time filter
    query = urllib.parse.quote_plus(f"{ticker} when:5y")
    url = f"https://news.google.com/rss/search?q={query}"
    feed = feedparser.parse(url)
    headlines = []
    for entry in feed.entries:
        try:
            pub_dt = datetime.datetime(*entry.published_parsed[:6]).date()
            if start_date <= pub_dt <= end_date:
                # Convert date to unix timestamp for compatibility with normalize_dates
                timestamp = int(time.mktime(pub_dt.timetuple()))
                headlines.append({'headline': entry.title, 'datetime': timestamp})
        except Exception:
            # Ignore entries with parsing errors
            continue
    return headlines

def fetch_news_data(ticker: str, start_date: datetime.date, end_date: datetime.date, api_key: str, years: int = 5) -> List[Dict]:
    """
    Fetches news data by iterating month by month.
    Uses Finnhub for 2025 and Google News RSS for all other years.
    """
    all_articles = []
    current_start = start_date

    while current_start <= end_date:
        # Define the end of the current monthly window
        current_end = current_start + relativedelta(months=1) - relativedelta(days=1)
        if current_end > end_date:
            current_end = end_date
        
        src = ""
        articles_for_window = []
        if current_start.year == 2025:
            # Finnhub for 2025
            articles_for_window = fetch_news_finnhub_for_range(ticker, api_key, current_start, current_end)
            src = "Finnhub"
            time.sleep(1) # Respect Finnhub rate limits
        else:
            # Google News RSS for other years
            articles_for_window = fetch_news_google_rss(ticker, current_start, current_end)
            src = "GoogleRSS"

        logger.info(f"{ticker} {current_start.strftime('%Y-%m-%d')}–{current_end.strftime('%Y-%m-%d')}: fetched {len(articles_for_window)} articles via {src}")
        if articles_for_window:
            all_articles.extend(articles_for_window)

        # Move to the start of the next month
        current_start += relativedelta(months=1)
        # Ensure we start exactly on the 1st of the next month to handle varying month lengths
        current_start = current_start.replace(day=1)

    return all_articles

def compute_sentiment(headlines: List[str], sentiment_model) -> List[float]:
    """
    Compute sentiment scores for a list of headlines.
    
    Args:
        headlines: List of news headlines
        sentiment_model: Pre-loaded sentiment analysis model
        
    Returns:
        List of sentiment scores (-1 to 1)
    """
    if not headlines:
        return []
    
    scores = []
    for headline in headlines:
        try:
            result = sentiment_model(headline)[0]
            label = result.get("label", "").lower()
            
            # Convert FinBERT labels to numeric scores
            if label == "positive":
                scores.append(1.0)
            elif label == "negative":
                scores.append(-1.0)
            else:
                scores.append(0.0)
        except Exception as e:
            print(f"Error computing sentiment for headline: {e}")
            scores.append(0.0)
    
    return scores

def merge_data(price_data: pd.DataFrame, news_data: List[dict], sentiment_model, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    if price_data.empty:
        return pd.DataFrame()
    merged_data = price_data.copy()
    # Normalize news dates and headlines
    news_df = normalize_dates(news_data, start_date, end_date)
    if news_df.empty:
        print(f"Warning: no valid news for this period")
    # Group headlines by date
    daily_sentiments = news_df.groupby('date')['headline'].apply(list).to_dict()
    sentiment_scores = []
    dates = []
    for date, headlines in daily_sentiments.items():
        if date in merged_data.index.date:
            scores = compute_sentiment(headlines, sentiment_model)
            if scores:
                avg_sentiment = sum(scores) / len(scores)
                sentiment_scores.append(avg_sentiment)
                dates.append(date)
    if sentiment_scores:
        sentiment_df = pd.DataFrame({'date': dates, 'sentiment': sentiment_scores})
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        sentiment_df.set_index('date', inplace=True)
        merged_data = merged_data.join(sentiment_df, how='left')
    else:
        merged_data['sentiment'] = 0.0
    merged_data['sentiment'].fillna(0.0, inplace=True)
    return merged_data

def extract_close_series(df: pd.DataFrame) -> pd.Series:
    """
    Extract the best available price series from a DataFrame.
    
    Args:
        df: DataFrame containing price data
        
    Returns:
        pd.Series: The selected price series
        
    Raises:
        ValueError: If DataFrame is empty
        KeyError: If no numeric price column is found
    """
    if df.empty:
        raise ValueError("Empty DataFrame—no price data")
    
    # Handle MultiIndex columns by flattening to single-level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    
    print(f"extract_close_series: flattened columns: {df.columns.tolist()}")
    
    # If all columns have the same name (like '^GSPC'), select the first one by position
    if len(set(df.columns)) == 1:
        print(f"extract_close_series: all columns have same name '{df.columns[0]}', selecting first column")
        series = df.iloc[:, 0]
        print(f"extract_close_series: selected first column (type: {type(series)})")
        return series
    
    # Helper to get first column as Series by position
    def get_first_col_series(col_name):
        matches = [i for i, c in enumerate(df.columns) if c == col_name]
        if matches:
            col_idx = matches[0]
            series = df.iloc[:, col_idx]
            print(f"extract_close_series: selected column '{df.columns[col_idx]}' at position {col_idx} (type: {type(series)})")
            return series
        return None
    
    # Try to get 'Close' column first
    close = get_first_col_series('Close')
    if close is not None:
        return close
    
    # Fall back to 'Adj Close' if 'Close' not available
    adj_close = get_first_col_series('Adj Close')
    if adj_close is not None:
        return adj_close
    
    # Last resort: find first numeric column by position
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise KeyError("No numeric price column found")
    
    # Get the first numeric column by position to avoid duplicate name issues
    first_numeric_idx = df.columns.get_loc(numeric_cols[0])
    series = df.iloc[:, first_numeric_idx]
    print(f"extract_close_series: selected numeric column '{df.columns[first_numeric_idx]}' at position {first_numeric_idx} (type: {type(series)})")
    return series

def add_sp500_return_feature(stock_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Add S&P 500 daily returns as a feature to the stock DataFrame.
    
    Args:
        stock_df: DataFrame containing stock data with datetime index
        start_date: Start date for S&P 500 data fetch (YYYY-MM-DD format)
        end_date: End date for S&P 500 data fetch (YYYY-MM-DD format)
        
    Returns:
        pd.DataFrame: Original stock DataFrame with 'SP_Return' column added
    """
    sp500 = yf.download("^GSPC", start=start_date, end=end_date, progress=False, auto_adjust=False)
    print("SP500 columns:", sp500.columns.tolist())
    print("SP500 sample:\n", sp500.head())
    try:
        close_series = extract_close_series(sp500)
        print("Extracted close_series type:", type(close_series))
        print("Extracted close_series head:\n", close_series.head())
        sp500_returns = close_series.pct_change()
        # Ensure sp500_returns is a Series with a DateTime index
        if not isinstance(sp500_returns, pd.Series):
            raise ValueError("sp500_returns is not a Series")
        if not isinstance(sp500_returns.index, pd.DatetimeIndex):
            raise ValueError("sp500_returns index is not DatetimeIndex")
        sp500_df = pd.DataFrame({'SP_Return': sp500_returns})
        print("sp500_df head:\n", sp500_df.head())
        merged_df = stock_df.merge(
            sp500_df, 
            left_index=True, 
            right_index=True, 
            how="left"
        )
        return merged_df
    except (ValueError, KeyError) as e:
        print(f"SP500 feature skipped: {e}")
        stock_df["SP_Return"] = np.nan
        return stock_df

def prepare_features(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and labels for training.
    
    Args:
        data: DataFrame with price and sentiment data
        
    Returns:
        Tuple of (features, labels)
    """
    # Create target variable: 1 if next day's close > current close, 0 otherwise
    data['target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    # Remove rows with missing target values
    data = data.dropna()
    
    # Prepare features (only sentiment for now)
    X = data[['sentiment']]
    y = data['target']
    
    return X, y

def train_and_evaluate(X: pd.DataFrame, y: pd.Series, ticker: str) -> Tuple[Optional[LogisticRegression], float]:
    """
    Train logistic regression model and evaluate accuracy.
    
    Args:
        X: Feature matrix
        y: Target labels
        ticker: Stock symbol for logging
        
    Returns:
        Tuple of (trained model, accuracy score)
    """
    if len(X) < 10:  # Need minimum data for training
        print(f"Warning: Insufficient data for {ticker} ({len(X)} samples)")
        return None, 0.0
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✓ {ticker} Model Accuracy: {accuracy:.3f} ({len(X)} samples)")
    
    return model, accuracy

def plot_sentiment_vs_price(data: pd.DataFrame, ticker: str, save_dir: str = "plots"):
    """
    Create and save sentiment vs price plot.
    
    Args:
        data: DataFrame with price and sentiment data
        ticker: Stock symbol
        save_dir: Directory to save plots
    """
    try:
        # Create plots directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot price on primary y-axis
        ax1.plot(data.index, data['Close'], color='blue', linewidth=1.5, label='Close Price')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Close Price ($)', color='blue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, alpha=0.3)
        
        # Plot sentiment on secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(data.index, data['sentiment'], color='red', linewidth=1, 
                linestyle='--', alpha=0.7, label='Sentiment')
        ax2.set_ylabel('Sentiment Score', color='red', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(-1.1, 1.1)
        
        # Plot S&P 500 return (secondary right y-axis, offset)
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(data.index, data['SP_Return'], color='green', linewidth=1, linestyle=':', alpha=0.7, label='S&P 500 Return')
        ax3.set_ylabel('S&P 500 Daily Return', color='green', fontsize=12)
        ax3.tick_params(axis='y', labelcolor='green')
        ax3.set_ylim(-0.05, 0.05)
        
        # Add title and legend
        plt.title(f'{ticker} - Daily Close Price, Sentiment, and S&P 500 Return', fontsize=14, fontweight='bold')
        
        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')
        
        plt.tight_layout()
        
        # Save plot
        filename = os.path.join(save_dir, f'sentiment_vs_price_{ticker}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved plot: {filename}")
        
    except Exception as e:
        print(f"Error creating plot for {ticker}: {e}")

def save_sentiment_data(data: pd.DataFrame, ticker: str, save_dir: str = "results"):
    """
    Save DataFrame with sentiment scores to a CSV file.
    
    Args:
        data: DataFrame with price and sentiment data
        ticker: Stock symbol
        save_dir: Directory to save the CSV file
    """
    try:
        # Create results directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        filename = os.path.join(save_dir, f'sentiment_data_{ticker}.csv')
        
        # Save the data to a CSV file
        data.to_csv(filename)
        
        print(f"✓ Saved sentiment data to: {filename}")
        
    except Exception as e:
        print(f"Error saving sentiment data for {ticker}: {e}")

def analyze_ticker(ticker: str, api_key: str, years: int = 5) -> Dict:
    """
    Perform full analysis for a single ticker.
    """
    print(f"\n{'='*60}\nAnalyzing {ticker}\n{'='*60}")
    
    # 1. Fetch data
    end_date = datetime.date.today()
    start_date = end_date - relativedelta(years=years)
    if start_date.year < 1:
        start_date = datetime.date(1, 1, 1)

    price_data = fetch_price_data(ticker, years=years)
    if price_data.empty:
        return {'ticker': ticker, 'success': False, 'error': 'No price data'}

    news_data = fetch_news_data(ticker, start_date, end_date, api_key, years=years)
    print(f"✓ Fetched {len(news_data)} news articles for {ticker}")
    
    # 2. Load sentiment model
    print("Loading sentiment model...")
    sentiment_model = pipeline("sentiment-analysis", model="ProsusAI/finbert", device="mps:0" if "mps" in sys.modules else "cpu")
    print(f"Device set to use {sentiment_model.device}")
    
    # 3. Merge data
    merged_data = merge_data(price_data, news_data, sentiment_model, start_date, end_date)
    
    # 4. Add S&P 500 feature
    merged_data = add_sp500_return_feature(merged_data, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    # Save sentiment data to CSV
    save_sentiment_data(merged_data, ticker)

    # 5. Prepare features
    X, y = prepare_features(merged_data)
    
    if X.empty or y.empty:
        return {'ticker': ticker, 'success': False, 'error': 'Not enough data for training'}
    
    # 6. Train and evaluate model
    model, accuracy = train_and_evaluate(X, y, ticker)
    if model:
        print(f"✓ {ticker} Model Accuracy: {accuracy:.3f} ({len(X)} samples)")
    
    # 7. Plot results
    plot_sentiment_vs_price(merged_data, ticker)
    
    analysis_result = {
        'ticker': ticker, 
        'success': True, 
        'accuracy': accuracy, 
        'samples': len(X)
    }
    return analysis_result

def main():
    """
    Main function to run the analysis for all tickers.
    """
    parser = argparse.ArgumentParser(description="Stock Price and Sentiment Analysis")
    parser.add_argument("ticker", nargs='?', default=None, help="Optional: Specific ticker to analyze")
    parser.add_argument("--years", type=int, default=5, help="Number of years for historical data")
    args = parser.parse_args()
    
    # Use environment variable for API key if available
    api_key = os.environ.get("FINNHUB_API_KEY", DEFAULT_FINNHUB_API_KEY)
    
    if not api_key:
        print("Error: Finnhub API key not found. Please set the FINNHUB_API_KEY environment variable.")
        sys.exit(1)
        
    all_results = {}
    
    tickers_to_process = [args.ticker] if args.ticker else TICKERS

    for ticker in tickers_to_process:
        print(f"\nAnalyzing {ticker}...")
        results = analyze_ticker(ticker, api_key, years=args.years)
        all_results[ticker] = results
        
    # Save results to a JSON file
    # Create the results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Save results to a JSON file
    with open("results/analysis_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
        
    print("\nAnalysis complete. Results saved to results/analysis_results.json")

if __name__ == "__main__":
    main()