import numpy as np


def rolling_mean_anomaly_detection(series, window_size=20, threshold=3.0):
    """
    Detect anomalies using Rolling Mean and Rolling Standard Deviation.
    """
    # Initialize output arrays with NaN or False values
    rolling_mean = np.full_like(series, fill_value=np.nan, dtype=float)
    rolling_std = np.full_like(series, fill_value=np.nan, dtype=float)
    anomalies = np.zeros(len(series), dtype=bool)

    # Loop through the series using a sliding window
    for i in range(window_size, len(series)):
        window = series[i - window_size:i]
        mean = np.mean(window)
        std = np.std(window)

        rolling_mean[i] = mean
        rolling_std[i] = std

        # Avoid issues with zero standard deviation
        if std == 0:
            std = 1e-8

        # Mark as anomaly if the deviation exceeds the threshold
        if abs(series[i] - mean) > threshold * std:
            anomalies[i] = True

    return anomalies, rolling_mean, rolling_std