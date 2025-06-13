# Simple Browser Automation with Playwright

This project uses [Playwright](https://playwright.dev/) to automate a browser.
It opens Google, searches for the term **"OpenAI Codex browser automation"**, and
saves a screenshot of the results page as `results.png`.

## Setup

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Install Playwright browsers:
   ```bash
   playwright install
   ```

## Usage

Run the automation script:
```bash
python automation.py
```
The screenshot will be saved as `results.png` in the project directory.

---

## Stock Sentiment Prediction

This repository also contains a simple NLP-based stock movement prediction prototype.
It fetches historical prices and recent news, computes daily sentiment using a pre-trained model, and trains a logistic regression classifier to predict whether the stock will go up the next day.

### Additional Setup

Install extra dependencies:
```bash
pip install -r requirements.txt
```
You may also need a NewsAPI key set as `NEWSAPI_KEY` environment variable. Without it, sample headlines will be used.

### Run

Execute the prediction script:
```bash
python stock_prediction.py
```
A plot showing sentiment vs closing price will display along with model accuracy in the console.

