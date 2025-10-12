import yfinance as yf

ticker = '^NDX'
data = yf.download(ticker, start='2025-9-1', end='2025-10-11', interval='1d')

data.to_csv('data/test/9,1 10,11 1day.csv')