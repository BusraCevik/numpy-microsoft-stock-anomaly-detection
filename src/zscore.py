import numpy as np


def zscore_anomaly_detection(series, threshold):
    """
    Detect anomalies in a time series using Z-Score.
    """
    # Calculate global mean and standard deviation
    mean = np.mean(series)
    std = np.std(series)

    # Avoid division by zero if variance is zero
    if std == 0:
        std = 1e-8

    # Compute Z-Scores for all data points
    z_scores = (series - mean) / std

    # Mark values exceeding the threshold as anomalies
    anomalies = np.abs(z_scores) > threshold

    return anomalies, z_scores