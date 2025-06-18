import datetime
import time
import requests
from dateutil.relativedelta import relativedelta
from typing import List, Dict
import pandas as pd

def fetch_news_finnhub_monthly(ticker: str, api_key: str, years: int = 5) -> List[Dict]:
    """
    Fetch N years of news articles for a ticker from Finnhub, 1 month at a time.
    Args:
        ticker: Stock symbol (e.g., 'AAPL')
        api_key: Finnhub API key
        years: Number of years to go back (default 5)
    Returns:
        List of article dictionaries
    """
    articles = []
    end_date = datetime.date.today()
    start_date = end_date - relativedelta(years=years)
    current_start = start_date

    while current_start < end_date:
        current_end = min(current_start + relativedelta(months=1) - datetime.timedelta(days=1), end_date)
        from_str = current_start.strftime('%Y-%m-%d')
        to_str = current_end.strftime('%Y-%m-%d')
        print(f"Fetching news for {ticker}: {from_str} to {to_str}")
        url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={from_str}&to={to_str}&token={api_key}"
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if data:
                articles.extend(data)
        except Exception as e:
            print(f"Error fetching {from_str} to {to_str}: {e}")
        time.sleep(1)  # Respect Finnhub rate limits
        current_start = current_end + datetime.timedelta(days=1)
    print(f"✓ Fetched {len(articles)} news articles for {ticker}")
    return articles

def normalize_dates(raw_news: List[Dict], start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    """
    - Parses each article's 'datetime' (UNIX ts) into a date.
    - Drops zero/invalid timestamps and out-of-range dates.
    - Returns DataFrame with columns ['date', 'headline'].
    """
    df = pd.DataFrame(raw_news)
    # coerce invalid/zero into NaT
    df['date'] = pd.to_datetime(
        df.get('datetime', 0),
        unit='s',
        errors='coerce'
    ).dt.date
    # keep only good rows
    mask = (
        df['date'].notna() &
        (df['date'] >= start_date) &
        (df['date'] <= end_date) &
        df['headline'].notna()
    )
    dropped = len(df) - mask.sum()
    if dropped:
        print(f"normalize_dates: dropped {dropped} bad articles")
    cleaned = df.loc[mask, ['date', 'headline']].reset_index(drop=True)
    return cleaned

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python finnhub_news_fetcher.py TICKER API_KEY")
        sys.exit(1)
    ticker = sys.argv[1]
    api_key = sys.argv[2]
    articles = fetch_news_finnhub_monthly(ticker, api_key)
    print(f"Fetched {len(articles)} articles for {ticker} in the last 5 years.") 