import yfinance as yf
import pandas as pd
from ta import add_all_ta_features
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# 📥 Fetch stock data from yfinance
def fetch_stock_data(ticker):
    print(f"📥 Fetching data for {ticker}...")
    stock_data = yf.download(ticker, auto_adjust=True)

    # Flatten MultiIndex columns if necessary
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = ['_'.join(map(str, col)).strip() for col in stock_data.columns]

    # ✅ Fix: Rename columns without duplicating the ticker
    stock_data.columns = [f"{col}_{ticker}" if ticker not in col else col for col in stock_data.columns]

    print("✅ Renamed columns:\n", stock_data.columns)
    return stock_data


# 🛠️ Add technical indicators
def add_technical_indicators(df, ticker):
    high_col = f"High_{ticker}"
    low_col = f"Low_{ticker}"
    close_col = f"Close_{ticker}"
    open_col = f"Open_{ticker}"
    volume_col = f"Volume_{ticker}"

    # Add technical indicators
    df = add_all_ta_features(
        df,
        open=open_col,
        high=high_col,
        low=low_col,
        close=close_col,
        volume=volume_col,
        fillna=True
    )

    print("✅ Technical indicators added:\n", df.tail())
    return df

# 🧠 Add sentiment scores using VADER
def add_sentiment_scores(df, headlines):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = pd.Series(index=df.index, dtype=float)

    for date in df.index:
        daily_headlines = headlines.get(date.date(), [])
        if daily_headlines:
            scores = [analyzer.polarity_scores(text)['compound'] for text in daily_headlines]
            sentiment_scores[date] = sum(scores) / len(scores)
        else:
            sentiment_scores[date] = 0.0  # Neutral if no headlines

    df[f'sentiment'] = sentiment_scores
    print("✅ Sentiment scores added:\n", df[['sentiment']].tail())
    return df

# 📊 Prepare the full dataset
def prepare_dataset(ticker):
    stock_data = fetch_stock_data(ticker)

    # Add technical indicators
    stock_data = add_technical_indicators(stock_data, ticker)

    # Example: Mock headlines for sentiment analysis
    headlines = {
        pd.Timestamp('2020-02-24'): ["Apple stock falls amid market fears"],
        pd.Timestamp('2020-02-25'): ["Strong performance by Apple"],
        pd.Timestamp('2020-02-26'): ["Apple faces supply chain issues"],
        pd.Timestamp('2020-02-27'): ["Market recovers slightly, Apple up"],
        pd.Timestamp('2020-02-28'): ["New iPhone model rumors boost sentiment"],
    }

    # Add sentiment scores
    stock_data = add_sentiment_scores(stock_data, headlines)

    print("✅ Final dataset ready for model training:\n", stock_data.tail())
    return stock_data

# 🚀 Main execution
# 🚀 Main execution
if __name__ == "__main__":
    ticker = 'AAPL'
    try:
        final_data = prepare_dataset(ticker)
        print("✅ Final engineered dataset:\n", final_data.tail())

        # 💾 Save to CSV
        final_data.to_csv("engineered_stock_data.csv")
        print("✅ Data saved to engineered_stock_data.csv")

    except Exception as e:
        print(f"❌ Error: {e}")
