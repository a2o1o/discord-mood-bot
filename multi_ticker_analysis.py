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
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import sys
import argparse
from dateutil.relativedelta import relativedelta
import numpy as np
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of tickers to analyze
TICKERS = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'TSLA', 'BRK-B', 'META', 'LLY', 'AVGO']

TICKER_KEYWORDS = {
    'AAPL': ['AAPL', 'Apple'],
    'MSFT': ['MSFT', 'Microsoft'],
    'AMZN': ['AMZN', 'Amazon'],
    'NVDA': ['NVDA', 'Nvidia', 'NVIDIA'],
    'GOOGL': ['GOOGL', 'Google', 'Alphabet'],
    'TSLA': ['TSLA', 'Tesla','TESLA'],
    'BRK-B': ['BRK-B', 'Berkshire Hathaway'],
    'META': ['META', 'Facebook','Instagram'],
    'LLY': ['LLY', 'Lilly', 'Eli Lilly'],
    'AVGO': ['AVGO', 'Broadcom']
}

# Load FinBERT
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model     = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def fetch_price_data(ticker: str, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    """
    Fetch historical price data from yfinance.
    
    Args:
        ticker: Stock symbol
        start_date: Start date for data fetch
        end_date: End date for data fetch
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
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

def fetch_external_headlines(ticker: str, start_date: datetime.date, end_date: datetime.date) -> list[dict]:
    """
    Fetch headlines for a ticker from All_external.csv for the given date range.
    Returns a list of dicts: {"headline": ..., "datetime": ..., "date": ...}
    Caches results to disk and loads from cache if available.
    """
    os.makedirs("external_headlines_cache", exist_ok=True)
    cache_file = f"external_headlines_cache/{ticker}_{start_date}_{end_date}.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)
    headlines = []
    try:
        # Read in chunks for large file
        chunk_iter = pd.read_csv("All_external.csv", chunksize=100_000, dtype=str)
        
        # Make start/end dates timezone-aware to match the data
        start_ts = pd.Timestamp(start_date, tz="UTC")
        end_ts = pd.Timestamp(end_date, tz="UTC")

        keywords = TICKER_KEYWORDS.get(ticker, [ticker])
        pattern = r'\b(' + '|'.join(map(re.escape, keywords)) + r')\b'

        for chunk in chunk_iter:
            # Convert 'Date' column to datetime
            chunk["Date"] = pd.to_datetime(chunk["Date"], errors="coerce", utc=True)
            
            # Drop rows where date conversion failed
            chunk.dropna(subset=['Date'], inplace=True)

            # Filter by stock symbol, title keywords, and date range
            stock_symbol_mask = chunk['Stock_symbol'] == ticker
            title_mask = chunk['Article_title'].str.contains(pattern, case=False, na=False, regex=True)
            date_mask = (chunk["Date"] >= start_ts) & (chunk["Date"] <= end_ts)
            
            chunk = chunk[(stock_symbol_mask | title_mask) & date_mask]
            
            for _, row in chunk.iterrows():
                title = row["Article_title"]
                date = row["Date"]
                if pd.isnull(title) or pd.isnull(date):
                    continue
                # Convert date to timestamp
                try:
                    dt = pd.to_datetime(date, utc=True)
                    timestamp = int(dt.timestamp())
                except Exception:
                    continue
                headlines.append({"headline": title, "datetime": timestamp, "date": dt.date()})
    except Exception as e:
        print(f"Error reading All_external.csv: {e}")
    print(f"✓ All_external: Found {len(headlines)} news items for {ticker}")
    with open(cache_file, "w") as f:
        json.dump(headlines, f, indent=2, default=str)
    return headlines

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

    # Ensure the price_data index is a timezone-naive DatetimeIndex at midnight
    merged_data = price_data.copy()
    merged_data.index = pd.to_datetime(merged_data.index).normalize()
    merged_data['sentiment'] = 0.0 # Default to 0

    if not news_data:
        print("Warning: No news data to merge.")
        return merged_data

    news_df = pd.DataFrame(news_data)
    if 'date' not in news_df.columns or 'headline' not in news_df.columns:
        print("Warning: News data is missing 'date' or 'headline' column.")
        return merged_data

    # Convert date column to datetime objects and normalize to midnight
    news_df['date'] = pd.to_datetime(news_df['date']).dt.normalize()

    # Group headlines by day
    daily_headlines = news_df.groupby('date')['headline'].apply(list)

    # Compute average sentiment for each day
    daily_sentiment_scores = {}
    for date, headlines in daily_headlines.items():
        if headlines:
            scores = compute_sentiment(headlines, sentiment_model)
            if scores:
                daily_sentiment_scores[date] = np.mean(scores)
    
    if not daily_sentiment_scores:
        print("Warning: Could not compute any daily sentiment scores.")
        return merged_data
    
    # Create sentiment Series with a DatetimeIndex
    sentiment_series = pd.Series(daily_sentiment_scores, name='sentiment')

    # Update the sentiment in the main DataFrame
    # This will align on the index and update values where dates match
    merged_data.update(sentiment_series)
    
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

def fetch_news_data(ticker: str, start_date: datetime.date, end_date: datetime.date, years: int = 5) -> list[dict]:
    return fetch_external_headlines(ticker, start_date, end_date)

def parse_date_arg(date_str: str) -> datetime.date:
    return pd.to_datetime(date_str).date()

def analyze_ticker(ticker: str, years: int = None, start_date: datetime.date = None, end_date: datetime.date = None) -> dict:
    print(f"\n{'='*60}\nAnalyzing {ticker}\n{'='*60}")
    if start_date is None or end_date is None:
        end_date = datetime.date.today()
        years = years if years else 5
        start_date = end_date - relativedelta(years=years)
        if start_date.year < 1:
            start_date = datetime.date(1, 1, 1)
    price_data = fetch_price_data(ticker, start_date=start_date, end_date=end_date)
    if price_data.empty:
        return {'ticker': ticker, 'success': False, 'error': 'No price data'}
    news_data = fetch_external_headlines(ticker, start_date, end_date)
    print(f"✓ Fetched {len(news_data)} news articles for {ticker}")
    print("Loading sentiment model...")
    sentiment_model = pipeline("sentiment-analysis", model="ProsusAI/finbert", device="mps:0" if "mps" in sys.modules else "cpu")
    print(f"Device set to use {sentiment_model.device}")
    merged_data = merge_data(price_data, news_data, sentiment_model, start_date, end_date)
    merged_data = add_sp500_return_feature(merged_data, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    save_sentiment_data(merged_data, ticker)
    X, y = prepare_features(merged_data)
    if X.empty or y.empty:
        return {'ticker': ticker, 'success': False, 'error': 'Not enough data for training'}
    model, accuracy = train_and_evaluate(X, y, ticker)
    if model:
        print(f"✓ {ticker} Model Accuracy: {accuracy:.3f} ({len(X)} samples)")
    plot_sentiment_vs_price(merged_data, ticker)
    analysis_result = {
        'ticker': ticker, 
        'success': True, 
        'accuracy': accuracy, 
        'samples': len(X)
    }
    return analysis_result

def main():
    parser = argparse.ArgumentParser(description="Stock Price and Sentiment Analysis")
    parser.add_argument("ticker", nargs='?', default=None, help="Optional: Specific ticker to analyze")
    parser.add_argument("--years", type=int, default=None, help="Number of years for historical data (ignored if --start-date is given)")
    parser.add_argument("--start-date", type=str, default=None, help="Start date for analysis (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date for analysis (YYYY-MM-DD, default: today)")
    args = parser.parse_args()
    all_results = {}
    tickers_to_process = [args.ticker] if args.ticker else TICKERS

    # Determine date range
    if args.start_date:
        start_date = parse_date_arg(args.start_date)
        end_date = parse_date_arg(args.end_date) if args.end_date else datetime.date.today()
        years = None
    else:
        end_date = datetime.date.today()
        years = args.years if args.years else 5
        start_date = end_date - relativedelta(years=years)
        if start_date.year < 1:
            start_date = datetime.date(1, 1, 1)
    print(f"Analysis window: {start_date} to {end_date}")

    for ticker in tickers_to_process:
        print(f"\nAnalyzing {ticker}...")
        results = analyze_ticker(ticker, start_date=start_date, end_date=end_date)
        all_results[ticker] = results
    os.makedirs("results", exist_ok=True)
    with open("results/analysis_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
    print("\nAnalysis complete. Results saved to results/analysis_results.json")

if __name__ == "__main__":
    main()