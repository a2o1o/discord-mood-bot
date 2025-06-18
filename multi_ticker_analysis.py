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
from finnhub_news_fetcher import fetch_news_finnhub_monthly, normalize_dates
from dateutil.relativedelta import relativedelta

# List of tickers to analyze
TICKERS = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'TSLA', 'BRK-B', 'META', 'LLY', 'AVGO']

# Default Finnhub API key
DEFAULT_FINNHUB_API_KEY = "d18ski9r01qkcat4kf10d18ski9r01qkcat4kf1g"

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

def fetch_news_data(ticker: str, start_date: datetime.date, end_date: datetime.date, api_key: str, years: int = 5) -> List[Dict]:
    # Just return all articles, no manual filtering
    return fetch_news_finnhub_monthly(ticker, api_key, years=years)

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
        
        # Add title and legend
        plt.title(f'{ticker} - Daily Close Price vs Sentiment Score', fontsize=14, fontweight='bold')
        
        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        # Save plot
        filename = os.path.join(save_dir, f'sentiment_vs_price_{ticker}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved plot: {filename}")
        
    except Exception as e:
        print(f"Error creating plot for {ticker}: {e}")

def analyze_ticker(ticker: str, api_key: str, years: int = 5) -> Dict:
    print(f"\n{'='*60}")
    print(f"Analyzing {ticker}")
    print(f"{'='*60}")
    price_data = fetch_price_data(ticker, years)
    if price_data.empty:
        return {'ticker': ticker, 'success': False, 'error': 'No price data'}
    end_date = datetime.date.today()
    start_date = end_date - relativedelta(years=years)
    if start_date.year < 1:
        start_date = datetime.date(1, 1, 1)
    raw_news = fetch_news_data(ticker, start_date, end_date, api_key, years=years)
    print("Loading sentiment model...")
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
    )
    merged_data = merge_data(price_data, raw_news, sentiment_model, start_date, end_date)
    if merged_data.empty:
        return {'ticker': ticker, 'success': False, 'error': 'No merged data'}
    X, y = prepare_features(merged_data)
    if len(X) == 0:
        return {'ticker': ticker, 'success': False, 'error': 'No features'}
    model, accuracy = train_and_evaluate(X, y, ticker)
    plot_sentiment_vs_price(merged_data, ticker)
    return {
        'ticker': ticker,
        'success': True,
        'accuracy': accuracy,
        'samples': len(X),
        'model': model,
        'data': merged_data
    }

def main():
    """Main function to run analysis for specified tickers."""
    print("Multi-Ticker Sentiment Analysis")
    print("=" * 40)
    
    parser = argparse.ArgumentParser(description="Multi-Ticker Sentiment Analysis")
    parser.add_argument('tickers', nargs='+', help="List of tickers or 'ALL' for default set")
    parser.add_argument('--years', '-y', type=int, default=5, help="Number of years to analyze (default: 5)")
    args = parser.parse_args()
    
    api_key = os.getenv('FINNHUB_API_KEY', DEFAULT_FINNHUB_API_KEY)
    
    if args.tickers[0].upper() == 'ALL':
        tickers_to_run = TICKERS
    else:
        tickers_to_run = [t.upper() for t in args.tickers]
    years = args.years
    
    os.makedirs('results', exist_ok=True)
    
    results = []
    for ticker in tickers_to_run:
        try:
            result = analyze_ticker(ticker, api_key, years=years)
            results.append(result)
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            results.append({'ticker': ticker, 'success': False, 'error': str(e)})
    
    # Print summary
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    successful_analyses = [r for r in results if r['success']]
    failed_analyses = [r for r in results if not r['success']]
    
    print(f"Successful analyses: {len(successful_analyses)}/{len(tickers_to_run)}")
    print(f"Failed analyses: {len(failed_analyses)}")
    
    if successful_analyses:
        print(f"\nModel Accuracies:")
        for result in successful_analyses:
            print(f"  {result['ticker']}: {result['accuracy']:.3f} ({result['samples']} samples)")
        
        avg_accuracy = sum(r['accuracy'] for r in successful_analyses) / len(successful_analyses)
        print(f"\nAverage Accuracy: {avg_accuracy:.3f}")
    
    if failed_analyses:
        print(f"\nFailed Analyses:")
        for result in failed_analyses:
            print(f"  {result['ticker']}: {result.get('error', 'Unknown error')}")
    
    # Save results
    results_file = os.path.join('results', 'analysis_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    print("Plots saved to: plots/")

if __name__ == "__main__":
    main() 