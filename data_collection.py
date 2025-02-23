import yfinance as yf
import pandas as pd
from datetime import datetime
import requests

def fetch_stock_data(ticker, start_date, end_date):
    """
    Enhanced stock data fetcher with multiple fallback methods
    """
    print(f"📊 Attempting to fetch {ticker} data ({start_date} to {end_date})")
    
    try:
        # Method 1: Direct API call with timeout
        stock = yf.Ticker(ticker, session=requests.Session())
        stock.history(period="max")  # Prime the session
        
        # Method 2: Try with different timezone handling
        data = stock.history(
            start=start_date,
            end=end_date,
            interval="1d",
            prepost=True,
            repair=True,
            timeout=15
        )
        
        if data.empty:
            # Method 3: Try alternative download approach
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                threads=True
            )
        
        # Validate data
        if not data.empty:
            print(f"✅ Success! Retrieved {len(data)} trading days")
            data = data.tz_localize(None)  # Remove timezone
            return data
            
        print("⚠️  Retrieved empty dataset")
        return None
        
    except Exception as e:
        print(f"❌ Critical error: {str(e)}")
        return None

def backup_data_fetch(ticker):
    """Fallback to Alpha Vantage API if needed"""
    print("🔄 Attempting backup data source...")
    API_KEY = "YOUR_API_KEY"  # Get free key from https://www.alphavantage.co
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={API_KEY}&datatype=csv&outputsize=full"
    
    try:
        df = pd.read_csv(url)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        print("✅ Backup data retrieval successful")
        return df
    except Exception as e:
        print(f"❌ Backup failed: {str(e)}")
        return None

if __name__ == "__main__":
    # Test with multiple date formats
    ticker = "AAPL"
    date_formats = [
        ("2020-01-01", "2023-12-31"),
        ("2020-1-1", "2023-12-31"),
        (datetime(2020,1,1), datetime(2023,12,31))
    ]
    
    for start, end in date_formats:
        data = fetch_stock_data(ticker, start, end)
        if data is not None:
            break
    
    # Final fallback to Alpha Vantage
    if data is None:
        data = backup_data_fetch(ticker)
    
    if data is not None:
        filename = f"raw_stock_data.csv"
        data.to_csv(filename)
        print(f"💾 Data saved to {filename}")
        print("Sample data:")
        print(data.head())
    else:
        print("🔴 All data retrieval methods failed")