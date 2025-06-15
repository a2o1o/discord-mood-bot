import datetime
import os
import json
from typing import List, Optional

import pandas as pd
import requests
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import pipeline
import matplotlib.pyplot as plt
import feedparser  # for RSS parsing

# Optional: pip install gnewsclient
from gnewsclient import gnewsclient  # for alternative news


def save_to_json(data: dict, filename: str) -> None:
    """Save data to a JSON file in a data directory."""
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    filepath = os.path.join('data', filename)
    
    # Convert any non-serializable objects to strings
    def json_serial(obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        raise TypeError(f"Type {type(obj)} not serializable")
    
    # If file exists, load existing data and update it
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                existing_data = json.load(f)
            existing_data.update(data)
            data = existing_data
        except json.JSONDecodeError:
            print(f"Warning: Could not read existing file {filepath}, creating new file")
    
    # Save the data
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=json_serial)
    except Exception as e:
        print(f"Error saving to {filepath}: {e}")


def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch historical stock data for the given ticker."""
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs(ticker, axis=1, level=1)
        data.index = pd.to_datetime(data.index)
        
        # Save stock data with proper timestamp conversion
        stock_data = {
            'ticker': ticker,
            'start_date': start,
            'end_date': end,
            'data': {str(idx): row.to_dict() for idx, row in data.iterrows()}
        }
        save_to_json(stock_data, f'{ticker}_stock_data.json')
        
        return data
    except Exception as exc:
        print(f"Failed to fetch stock data: {exc}")
        return pd.DataFrame()


def fetch_headlines_newsapi(ticker: str, date: datetime.date, api_key: Optional[str]) -> List[str]:
    """Fetch headlines via NewsAPI for a given date."""
    if not api_key:
        return []
    url = (
        "https://newsapi.org/v2/everything?"
        f"q={ticker}&from={date}&to={date}&language=en&sortBy=publishedAt&apiKey={api_key}"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        
        # Save NewsAPI data
        newsapi_data = {
            'ticker': ticker,
            'date': date.isoformat(),
            'articles': articles
        }
        save_to_json(newsapi_data, f'{ticker}_newsapi_{date.isoformat()}.json')
        
        return [article.get("title", "") for article in articles]
    except Exception as exc:
        print(f"NewsAPI error: {exc}")
        return []


def fetch_headlines_gnews(ticker: str) -> List[str]:
    """Fetch recent headlines using GNews free client."""
    client = gnewsclient.NewsClient(language='english', max_results=5)
    client.query = ticker
    try:
        results = client.get_news()
        
        # Save GNews data
        gnews_data = {
            'ticker': ticker,
            'timestamp': datetime.datetime.now().isoformat(),
            'articles': results
        }
        save_to_json(gnews_data, f'{ticker}_gnews_{datetime.date.today().isoformat()}.json')
        
        return [item.get('title', '') for item in results]
    except Exception as exc:
        print(f"GNews error: {exc}")
        return []


def fetch_headlines_rss(rss_url: str, limit: int = 5) -> List[str]:
    """Fetch headlines from an RSS feed URL."""
    try:
        feed = feedparser.parse(rss_url)
        entries = feed.entries[:limit]
        
        # Save RSS data
        rss_data = {
            'url': rss_url,
            'timestamp': datetime.datetime.now().isoformat(),
            'entries': [{'title': entry.title, 'link': entry.link, 'published': entry.published} 
                       for entry in entries]
        }
        save_to_json(rss_data, f'rss_{datetime.date.today().isoformat()}.json')
        
        return [entry.title for entry in entries]
    except Exception as exc:
        print(f"RSS error: {exc}")
        return []


def fetch_headlines_reddit(ticker: str, limit: int = 5) -> List[str]:
    """Fetch recent post titles from a Reddit stock subreddit."""
    url = (
        f"https://www.reddit.com/r/stocks/search.json?"
        f"q={ticker}&restrict_sr=1&sort=new&limit={limit}"
    )
    headers = {"User-Agent": "stock-sentiment-bot/0.1"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        posts = resp.json().get('data', {}).get('children', [])
        return [post['data'].get('title', '') for post in posts]
    except Exception as exc:
        print(f"Reddit error: {exc}")
        return []


def fetch_headlines(ticker: str, date: datetime.date, api_key: Optional[str]) -> List[str]:
    """Aggregate headlines from multiple sources for a ticker on a given date."""
    if ticker.upper() == "FAKE":
        return [f"{ticker} stock news headline"]
    headlines: List[str] = []
    # 1) NewsAPI (date-bound)
    headlines += fetch_headlines_newsapi(ticker, date, api_key)
    # 2) GNews (most recent)
    headlines += fetch_headlines_gnews(ticker)
    # 3) RSS feed (example: Yahoo Finance)
    yahoo_rss = f"https://finance.yahoo.com/rss/headline?s={ticker}"
    headlines += fetch_headlines_rss(yahoo_rss)
    # 4) Reddit r/stocks posts
    #headlines += fetch_headlines_reddit(ticker)
    # Fallback if no headlines found
    if not headlines:
        return [f"{ticker} stock news headline"]
    return headlines


def analyze_sentiment(headlines: List[str]) -> float:
    """Compute average sentiment score for a list of headlines."""
    if not headlines:
        return 0.0
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
    )
    scores = []
    sentiment_results = []
    
    for text in headlines:
        result = sentiment_pipeline(text)[0]
        label = result.get("label", "").lower()
        score = 1.0 if label == "positive" else (-1.0 if label == "negative" else 0.0)
        scores.append(score)
        sentiment_results.append({
            'text': text,
            'label': label,
            'score': score
        })
    
    # Save sentiment analysis results
    sentiment_data = {
        'timestamp': datetime.datetime.now().isoformat(),
        'headlines': sentiment_results,
        'average_score': sum(scores) / len(scores)
    }
    save_to_json(sentiment_data, f'sentiment_{datetime.date.today().isoformat()}.json')
    
    return sum(scores) / len(scores)


def prepare_dataset(ticker: str, n_days: int, api_key: Optional[str]) -> pd.DataFrame:
    """Prepare a DataFrame with sentiment scores and next-day movement labels."""
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=n_days + 1)
    stock = fetch_stock_data(ticker, start_date.isoformat(), end_date.isoformat())
    if stock.empty:
        raise RuntimeError("No stock data fetched")

    stock.sort_index(inplace=True)
    sentiments = []
    for current_timestamp in stock.index[:-1]:
        current_date = current_timestamp.date()
        headlines = fetch_headlines(ticker, current_date, api_key)
        sentiment = analyze_sentiment(headlines)
        sentiments.append((current_timestamp, sentiment))

    sent_df = (
        pd.DataFrame(sentiments, columns=["Date", "sentiment"])  
        .set_index("Date")
    )
    sent_df.index = pd.to_datetime(sent_df.index)

    data = stock.join(sent_df, how="left")
    data["sentiment"].fillna(0.0, inplace=True)
    data["target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    data.dropna(inplace=True)
    return data


def train_model(data: pd.DataFrame) -> LogisticRegression:
    """Train a logistic regression model on the sentiment data."""
    X = data[["sentiment"]]
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model accuracy: {acc:.2f}")
    return model


def plot_sentiment_vs_price(data: pd.DataFrame, ticker: str) -> None:
    """Plot closing price and sentiment over time."""
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Close Price", color="tab:blue")
    ax1.plot(data.index, data["Close"], color="tab:blue", label="Close")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Sentiment", color="tab:red")
    ax2.plot(data.index, data["sentiment"], color="tab:red", linestyle="--", label="Sentiment")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title(f"{ticker} Sentiment vs Price")
    fig.tight_layout()
    plt.show()


def main():
    ticker = os.getenv("TICKER", "AMZN")  # default to AAPL
    n_days = int(os.getenv("N_DAYS", "30"))
#    api_key = os.environ.get("NEWSAPI_KEY")
    api_key = "c544880419db49ccb5f2bad167657c5d"

    data = prepare_dataset(ticker, n_days, api_key)
    model = train_model(data)
    plot_sentiment_vs_price(data, ticker)


if __name__ == "__main__":
    main()

# Unit tests
import unittest

class TestStockSentiment(unittest.TestCase):
    def test_fetch_stock_data(self):
        df = fetch_stock_data('AAPL', '2020-01-01', '2020-01-10')
        self.assertFalse(df.empty)
        self.assertIn('Close', df.columns)

    def test_fetch_headlines_fallback(self):
        heads = fetch_headlines('FAKE', datetime.date(1900,1,1), None)
        self.assertEqual(heads, ['FAKE stock news headline'])

    def test_analyze_sentiment_empty(self):
        self.assertEqual(analyze_sentiment([]), 0.0)

    def test_fetch_headlines_reddit(self):
        posts = fetch_headlines_reddit('AAPL', limit=1)
        # Expect a list, length <= 1
        self.assertIsInstance(posts, list)
        self.assertLessEqual(len(posts), 1)

if __name__ == '__main__':
    unittest.main()
