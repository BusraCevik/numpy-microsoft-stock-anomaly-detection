import numpy as np

"""
This module implements anomaly detection using Rolling Mean
and Rolling Standard Deviation. Each data point is compared
against a local mean and standard deviation calculated over
a sliding window.
"""


def rolling_mean_anomaly_detection(series, window_size=20, threshold=3.0):
    """
    Detect anomalies using Rolling Mean and Rolling Standard Deviation.

    Args:
        series (np.ndarray): Input time series data.
        window_size (int): Size of the rolling window.
        threshold (float): Multiplier for standard deviation.

    Returns:
        anomalies (np.ndarray): Boolean array indicating anomalies.
        rolling_mean (np.ndarray): Rolling mean values.
        rolling_std (np.ndarray): Rolling standard deviation values.
    """
    rolling_mean = np.full_like(series, fill_value=np.nan, dtype=float)
    rolling_std = np.full_like(series, fill_value=np.nan, dtype=float)
    anomalies = np.zeros(len(series), dtype=bool)

    for i in range(window_size, len(series)):
        window = series[i - window_size:i]
        mean = np.mean(window)
        std = np.std(window)

        rolling_mean[i] = mean
        rolling_std[i] = std

        if abs(series[i] - mean) > threshold * std:
            anomalies[i] = True

    return anomalies, rolling_mean, rolling_std
