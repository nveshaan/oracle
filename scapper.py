import yfinance as yf
import pandas as pd

ticker = '^NDX'
data = yf.download(ticker, start='2025-8-15', end='2025-9-15', interval='2m')

data.to_csv('one_month_1m.csv', index=False)