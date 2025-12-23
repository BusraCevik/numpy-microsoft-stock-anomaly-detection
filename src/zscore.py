import numpy as np

"""
This module implements the Z-Score anomaly detection method.
Z-Score measures how many standard deviations a data point 
is from the mean. Points with absolute Z-Score above a threshold 
are considered anomalies.
"""


def zscore_anomaly_detection(series, threshold):
    """
    Detect anomalies in a time series using Z-Score.

    Args:
        series (np.ndarray): Input time series.
        threshold (float): Z-Score threshold to identify anomalies.

    Returns:
        anomalies (np.ndarray): Boolean array where True indicates an anomaly.
        z_scores (np.ndarray): Z-Score values of the series.
    """
    mean = np.mean(series)
    std = np.std(series)

    z_scores = (series - mean) / std
    anomalies = np.abs(z_scores) > threshold

    return anomalies, z_scores
