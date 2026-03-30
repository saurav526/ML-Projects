import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

ticker = 'GME'
df = yf.download(ticker, start='2020-03-01', end='2025-03-01', interval='1d')
df.head()

df = df[['Close']]
df = df.rename(columns={'Close': 'value'})
df.index.name = 'Date'
df.head()

# plot the stock prices
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['value'], label='GME Price')
plt.title('GME Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

df['Z-Score'] = zscore(df['value'])

# define threshold for anomaly
threshold = 2.5
df['Anomaly Z-Score'] = df['Z-Score'].abs() > threshold

X = df[['value']].values

# initialize and fit isolation forest
isolation_forest = IsolationForest(contamination=0.01, random_state=42)
df['Anomaly Isolation Forest'] = isolation_forest.fit_predict(X) == -1

# decompose the time series
decomposition = seasonal_decompose(df['value'], model='additive', period=30)
df['Residuals'] = decomposition.resid

# residual threshold
residual_threshold = 15
df['Anomaly Residual'] = df['Residuals'].abs() > residual_threshold

df.head()

plt.figure(figsize=(15, 12))

# plotting z-score anomalies
plt.subplot(3, 1, 1)
plt.plot(df.index, df['value'], label='Stock Price')
plt.scatter(df.index[df['Anomaly Z-Score']], df['value'][df['Anomaly Z-Score']], color='red', label='Anomalies')
plt.title('Z-Score Anomaly Detection')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# plotting isolation forest anomalies
plt.subplot(3, 1, 2)
plt.plot(df.index, df['value'], label='Stock Price')
plt.scatter(df.index[df['Anomaly Isolation Forest']], df['value'][df['Anomaly Isolation Forest']], color='red', label='Anomalies')
plt.title('Isolation Forest Anomaly Detection')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# plotting residual anomalies
plt.subplot(3, 1, 3)
plt.plot(df.index, df['value'], label='Stock Price')
plt.scatter(df.index[df['Anomaly Residual']], df['value'][df['Anomaly Residual']], color='red', label='Anomalies')
plt.title('Residual Anomaly Detection')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

plt.tight_layout()
plt.show()

