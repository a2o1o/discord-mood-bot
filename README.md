# Stock Sentiment Analysis System

A comprehensive sentiment analysis system that analyzes news sentiment for major stocks using FinBERT (financial-specific BERT model) and correlates it with stock price movements.

## Features

- **Multi-Ticker Analysis**: Analyzes 10 major stocks (AAPL, MSFT, AMZN, NVDA, GOOGL, TSLA, BRK-B, META, LLY, AVGO)
- **FinBERT Sentiment Analysis**: Uses pre-trained financial BERT model for accurate sentiment scoring
- **Large-Scale News Data**: Processes 5.3GB of historical news data from All_external.csv
- **Date Range Flexibility**: Analyze any date range with custom start/end dates
- **Caching System**: Efficient caching to avoid repeated data processing
- **Visualization**: Generates sentiment vs price correlation plots
- **Results Export**: Saves sentiment data and analysis results to CSV/JSON

## Setup

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure you have the required data file:
   - `All_external.csv` (5.3GB) - Historical news database

## Usage

### Analyze All Stocks (Default 5 Years)
```bash
python multi_ticker_analysis.py
```

### Analyze Specific Stock
```bash
python multi_ticker_analysis.py AAPL
```

### Custom Date Range
```bash
# Specific date range
python multi_ticker_analysis.py AAPL --start-date 2020-01-01 --end-date 2023-12-31

# Custom number of years
python multi_ticker_analysis.py TSLA --years 3

# Analyze all stocks for specific period
python multi_ticker_analysis.py --start-date 2023-01-01 --end-date 2023-12-31
```

## Command Line Options

- **`ticker`** (optional): Specific stock symbol to analyze (default: all 10 stocks)
- **`--years`**: Number of years for historical data (default: 5)
- **`--start-date`**: Start date in YYYY-MM-DD format
- **`--end-date`**: End date in YYYY-MM-DD format (default: today)

## Output

The system generates:

- **Plots**: Sentiment vs price correlation charts saved to `plots/` directory
- **Results**: Sentiment data CSV files saved to `results/` directory
- **Analysis Summary**: JSON file with accuracy scores and sample counts
- **Cache**: Processed data cached in `external_headlines_cache/` for efficiency

## Data Sources

- **News Data**: All_external.csv (5.3GB historical news database)
- **Stock Prices**: Yahoo Finance API
- **Market Context**: S&P 500 returns as additional feature

## Sentiment Analysis

- Uses **FinBERT** (ProsusAI/finbert) for financial sentiment analysis
- Converts sentiment labels to numeric scores (-1 to 1)
- Aggregates multiple headlines per day into average daily sentiment
- Correlates sentiment with stock price movements

## Example Output

```
✓ Fetched 1257 days of price data for AAPL
✓ All_external: Found 2847 news items for AAPL
✓ AAPL Model Accuracy: 0.523 (1256 samples)
✓ Saved plot: plots/sentiment_vs_price_AAPL.png
✓ Saved sentiment data to: results/sentiment_data_AAPL.csv
```

## Requirements

- Python 3.7+
- pandas, numpy, matplotlib
- scikit-learn, transformers
- yfinance, requests
- All_external.csv data file

