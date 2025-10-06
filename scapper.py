import yfinance as yf

ticker = '^NDX'
data = yf.download(ticker, start='2025-7-2', end='2025-9-30', interval='1d')

data.to_csv('7,2 9,29 1day.csv')