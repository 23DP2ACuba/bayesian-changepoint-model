import yfinance as yf
import numpy as np
import pandas as pd
import autobnn as abnn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import jax

def label_data(df, lookahead, threshold = 0.002):
    df["future_return"] = df.Close.pct_change(lookahead).shift(-lookahead)
    df["label"] = 0
    df.loc[df["future_return"] > threshold, "label"] = 2
    df.loc[df["future_return"] < threshold, "label"] = 1

    df.drop(columns = ["future_return"], inplace = True)

def sliding_window(df, window_size = 50, lookahead = 5, threshold = 0.002):
    label_data(df = df, lookahead = lookahead, threshold = threshold)

    close = df.Close.values
    labels = df.label.values
    n_samples = len(df) - window_size
    x_list, y_list = [], []

    for i in range(n_samples):
        window_data = close[i : i + window_size]
        last_label = labels[i + window_size - 1]

    x = np.array(x_list)
    y = np.array(y_list)

    return x, y

data = yf.Ticker("TSLA").history(start="2022-01-01", end="2025-04-01")  # End date adjusted to current date
data.drop(columns=["Stock Splits", "Dividends"], inplace=True, errors='ignore')
data = data.dropna()

print(data.head(20))

def walk_forward(
        df, 
        window_size = 100, 
        train_size = 50, 
        step_size = 5, 
        threshold = 0.002, 
        n_classes = 3
):
    X, y = sliding_window(df=df, window_size=window_size, threshold=threshold)
    n_total = len(X)
    if n_total < train_size + 1:
        raise ValueError("Not enough samples for walk-forward analysis.")

    accuracies = []

    for i in range(0, n_total - train_size, step_size):
        X_train = X[i : i + train_size]
        y_train = y[i : i + train_size]

        test_index = i + train_size
        if test_index >= n_total:
            break

        X_test = X[test_index : test_index + 1]  
        y_test = y[test_index : test_index + 1]  

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        estimators = []
        for c in range(n_classes):
            y_train_c = (y_train == c).astype(int)

            model_c = abnn.operators.Add(
                bnns=(
                    abnn.kernels.PeriodicBNN(width=20, period=12.0),
                    abnn.kernels.LinearBNN(width=20),
                    abnn.kernels.MaternBNN(width=20),
                )
            )
            estimator_c = abnn.estimators.AutoBnnMapEstimator(
                model_c,
                likelihood_model="normal_likelihood_logistic_noise",  
                seed=jax.random.PRNGKey(42),
                periods=[12],

            )

            estimator_c.fit(X_train_scaled, y_train_c)
            estimators.append(estimator_c)

        class_probs = []
        for c in range(n_classes):
            y_pred_c = estimators[c].predict(X_test_scaled)
            
   
            if y_pred_c.ndim == 2:
                logit = y_pred_c[0, 0] 
            else:
                logit = y_pred_c[0]
            
            prob_c = 1.0 / (1.0 + np.exp(-logit))
            class_probs.append(prob_c)

        y_pred_class = np.argmax(class_probs)  

        acc = accuracy_score(y_test, [y_pred_class])
        accuracies.append(acc)

    return accuracies

dfsample = data[:1000].copy()

results = walk_forward(
    df=dfsample,
    window_size=40,
    train_size=20,
    step_size=5,
    threshold=0.002,
    n_classes=3
)

print("Accuracies:", results)
if results:
    print("Mean Accuracy:", np.mean(results))