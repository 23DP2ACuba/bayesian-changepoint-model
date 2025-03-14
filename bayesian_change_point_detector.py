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
tfdist = tfp.distributions
tfbij = tfp.bijectors
SYMBOL = "TSLA"
PERIOD = "1d"
INTERVAL = "1h"
START = "2024-01-01"
END = "2025-01-01"
WINDOW = 50
plt.style.use("ggplot")
LAMBDA_RATE = 10.0
NUM_SAMPLES = 2000
NUM_BURNIN_STEPS = 1000
NUM_CHAINS = 2
MAX_NUM_CHANGEPOINTS = 100

# -----------------------

# gather data
data = yfinance.Ticker(SYMBOL).history(period=PERIOD, interval = INTERVAL, start=START, end=END)

# feature creation
def compute_std_dev(prices: pd.Series, log_returns: bool = True, window: int = 20, min_periods: int = 1) -> pd.Series:
    returns = np.log(prices/prices.shift(1)) if log_returns else prices/prices.shift(1)
    return returns.rolling(window=window, min_periods=min_periods).std()

def OCV_pct(open: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    _oc = (close - open) / open * 100
    _avg_vol = volume.rolling(window=window, min_periods=1).mean()
    ocv_pct = _oc * (volume / _avg_vol)
    return ocv_pct

data['SMA_10'] = np.log(ta.sma(data['Close'], length=50))
data['SMA_20'] = np.log(ta.sma(data['Close'], length=200))
data['EMA_10'] = np.log(ta.ema(data['Close'], length=50))
data['EMA_20'] = np.log(ta.ema(data['Close'], length=200))
data["StdDev"] = compute_std_dev(data.Close, window=WINDOW)
data["HL_pct"] = (data.High - data.Low) / data.Open
data["OCV_pct"] = OCV_pct(close=data.Close, open=data.Open, volume=data.Volume, window=WINDOW)

features = ["Close", "StdDev", "HL_pct", "OCV_pct"]
data = data[features]
data.fillna(0, inplace=True)

# Model definition
class BayesianChangePointModel:
  def __init__(self, data, lambda_rate, max_num_changepoints):
    self.data = tf.convert_to_tensor(data.Close.values, dtype = tf.float32)
    self.num_points = len(data)
    self.poisson_prior = tfdist.Poisson(lambda_rate)
    self.max_num_changepoints = max_num_changepoints
  
# log and segment probagility computation
  def joint_log_prob(self, changepoint_locs, means, stddevs):
    num_changepoints = tf.cast(tf.shape(changepoint_locs)[0], tf.float32)
    prior_cp = self.poisson_prior.log_prob(num_changepoints)

    rv_cp_locs = tfdist.Uniform(0, tf.cast(self.num_points, tf.float32))
    rv_means = tfdist.Normal(loc = 0., scale = tf.cast(10, tf.float32))
    rv_stddevs = tfdist.HalfNormal(scale = tf.cast(2, tf.float32))

    cp_locs_int = tf.cast(changepoint_locs, tf.int32)
    all_cps = tf.concat([tf.constant([0], dtype=tf.int32), 
                         cp_locs_int, 
                         tf.constant([self.num_points], dtype=tf.int32)], axis=0)
    
    # Likelihood computation
    def compute_log_likelihood(all_cps):
      
      def compute_segment_likelihood(index):
        start = all_cps[i]
        end = all_cps[i + 1]
        segment = self.data[start:end]
        return tf.reduce_sum(
            tfdist.Normal(loc = means[i], scale = stddevs[i]).log_prob(segment)
        )

      segment_i = tf.range(tf.size(all_cps) - 1)
      return tf.reduce_sum(
          tf.map_fn(compute_segment_likelihood, segment_i, dtype = tf.float32)
      )

    return (
            prior_cp +
            tf.reduce_sum(rv_cp_locs.log_prob(changepoint_locs)) +
            tf.reduce_sum(rv_means.log_prob(means)) +
            tf.reduce_sum(rv_stddevs.log_prob(stddevs)) +
            compute_log_likelihood(all_cps)
        )
# model
model = BayesianChangePointModel(data, LAMBDA_RATE, MAX_NUM_CHANGEPOINTS)
# initial state preparation
@tf.function
def get_initial_state():
    num_cps = tf.minimum(
        tf.cast(model.poisson_prior.sample(), tf.int32),
        model.max_num_changepoints
    )
    return [
        tf.sort(tfdist.Uniform(0, model.num_points).sample([num_cps])),
        tf.random.normal([num_cps + 1]),
        tf.abs(tf.random.normal([num_cps + 1]))
    ]

# Markov chain monte carlo sampling
hmc = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=lambda cp_locs, means, stddevs: model.joint_log_prob(cp_locs, means, stddevs),
    num_leapfrog_steps=3,
    step_size=0.1
)

@tf.function
def run_chain():
    return tfp.mcmc.sample_chain(
        num_results=NUM_SAMPLES,
        num_burnin_steps=NUM_BURNIN,
        current_state=get_initial_state(),
        kernel=hmc,
        trace_fn=None
    )

samples = run_chain()
changepoint_samples = samples[0].numpy()
print(changepoint_samples) 