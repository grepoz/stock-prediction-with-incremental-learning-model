import yfinance as yf


if __name__ == '__main__':
    ticker = yf.Ticker("AAPL")
    data = ticker.history(interval="1m")  # 1 minute interval
    print(data)
