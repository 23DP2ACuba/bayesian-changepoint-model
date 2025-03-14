'''
Bayesian Change Point Detector
'''
import yfinance
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import pandas_ta as ta
import pandas as pd

# -------- config -------
SYMBOL = "ETH-USD"
PERIOD = "1d"
START = "2023-01-01"
END = "2025-01-01"
WINDOW = 20
plt.style.use("ggplot")
LAMBDA_RATE = 5.0
NUM_SAMPLES = 1000
NUM_BURNIN = 500
NUM_CHAINS = 2
MAX_NUM_CHANGEPOINTS = 1000

# -----------------------

# gather data
data = yfinance.Ticker(SYMBOL).history(period=PERIOD, start=START, end=END)

# feature creation
def compute_std_dev(prices: pd.Series, log_returns: bool = True, window: int = 20, min_periods: int = 1) -> pd.Series:
    returns = np.log(prices/prices.shift(1)) if log_returns else prices/prices.shift(1)
    return returns.rolling(window=window, min_periods=min_periods).std()

def OCV_pct(open: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    _oc = (close - open) / open * 100
    _avg_vol = volume.rolling(window=window, min_periods=1).mean()
    ocv_pct = _oc * (volume / _avg_vol)
    return ocv_pct

data['SMA_50'] = np.log(ta.sma(data['Close'], length=50))
data['SMA_200'] = np.log(ta.sma(data['Close'], length=200))
data['EMA_50'] = np.log(ta.ema(data['Close'], length=50))
data['EMA_200'] = np.log(ta.ema(data['Close'], length=200))
data["StdDev"] = compute_std_dev(data.Close, window=WINDOW)
data["HL_pct"] = (data.High - data.Low) / data.Open
data["OCV_pct"] = OCV_pct(close=data.Close, open=data.Open, volume=data.Volume, window=WINDOW)

features = ["Close", "StdDev", "HL_pct", "OCV_pct"]
data = data[features]
data.fillna(0, inplace=True)

def log_likelihood(data, changepoints, LAMBDA_RATE):
  poisson_dist = tfp.distributions.Poisson(rate=LAMBDA_RATE)
  log_prob = tf.reduce_sum(poisson_dist.log_prob(data))
  return log_prob

n = 100
log_likelihoods = []
for cp in range(1, n):
    log_likelihoods.append(log_likelihood(data, cp, LAMBDA_RATE))

plt.plot(range(1, n), log_likelihoods)
plt.axvline(x=120, color='r', linestyle='--', label="True Changepoint")
plt.title("Log Likelihood for Different Changepoint Locations")
plt.xlabel("Changepoint Location")
plt.ylabel("Log Likelihood")
plt.legend()
plt.show()


most_probable_changepoint = np.argmax(log_likelihoods) + 1
print(f"Most probable changepoint location: {most_probable_changepoint}")