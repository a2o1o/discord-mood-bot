import datetime
import os
from typing import List

import pandas as pd
import requests
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import pipeline
import matplotlib.pyplot as plt


def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch historical stock data for the given ticker."""
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs(ticker, axis=1, level=1)
        data.index = pd.to_datetime(data.index)
        return data
    except Exception as exc:
        print(f"Failed to fetch stock data: {exc}")
        return pd.DataFrame()


def fetch_headlines(ticker: str, date: datetime.date, api_key: str | None) -> List[str]:
    """Return news headlines for a given date. Fallback to sample data if unavailable."""
    if not api_key:
        # Return sample headlines when API key is missing
        return [f"{ticker} stock news headline"]

    url = (
        "https://newsapi.org/v2/everything?"
        f"q={ticker}&from={date}&to={date}&language=en&sortBy=publishedAt&apiKey={api_key}"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        return [a.get("title", "") for a in articles]
    except Exception as exc:
        print(f"Failed to fetch headlines: {exc}")
        return [f"{ticker} stock news headline"]


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
    for text in headlines:
        result = sentiment_pipeline(text)[0]
        label = result["label"].lower()
        if label == "positive":
            scores.append(1.0)
        elif label == "negative":
            scores.append(-1.0)
        else:
            scores.append(0.0)
    return sum(scores) / len(scores)


def prepare_dataset(ticker: str, n_days: int, api_key: str | None) -> pd.DataFrame:
    """Prepare a DataFrame with sentiment and next-day movement labels."""
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=n_days + 1)
    stock = fetch_stock_data(ticker, start_date.isoformat(), end_date.isoformat())
    if stock.empty:
        raise RuntimeError("No stock data fetched")

    stock.sort_index(inplace=True)

    sentiments = []
    for current_date in stock.index[:-1]:
        headlines = fetch_headlines(ticker, current_date.date(), api_key)
        sentiment = analyze_sentiment(headlines)
        sentiments.append((current_date, sentiment))

    sent_df = pd.DataFrame(sentiments, columns=["Date", "sentiment"]).set_index("Date")
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.2f}")
    return model


def plot_sentiment_vs_price(data: pd.DataFrame, ticker: str) -> None:
    """Plot closing price and sentiment over time."""
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Close", color="tab:blue")
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
    ticker = "AAPL"
    n_days = 30
    api_key = os.environ.get("NEWSAPI_KEY")
    data = prepare_dataset(ticker, n_days, api_key)
    model = train_model(data)
    plot_sentiment_vs_price(data, ticker)


if __name__ == "__main__":
    main()
