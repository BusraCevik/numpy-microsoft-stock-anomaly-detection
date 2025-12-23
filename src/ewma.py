import numpy as np

"""
This module implements anomaly detection using
Exponentially Weighted Moving Average (EWMA).
EWMA gives more importance to recent observations,
making it suitable for trend-aware anomaly detection.
"""


def ewma_anomaly_detection(series, alpha=0.3, threshold=3.0):
    """
    Detect anomalies using EWMA.

    Args:
        series (np.ndarray): Input time series data.
        alpha (float): Smoothing factor (0 < alpha <= 1).
        threshold (float): Multiplier for standard deviation.

    Returns:
        anomalies (np.ndarray): Boolean array indicating anomalies.
        ewma (np.ndarray): EWMA values.
        ewma_std (np.ndarray): Rolling standard deviation of EWMA residuals.
    """
    ewma = np.zeros_like(series, dtype=float)
    ewma[0] = series[0]

    # Compute EWMA values
    for i in range(1, len(series)):
        ewma[i] = alpha * series[i] + (1 - alpha) * ewma[i - 1]

    # Residuals between actual values and EWMA
    residuals = series - ewma

    # Rolling standard deviation of residuals
    ewma_std = np.zeros_like(series, dtype=float)
    for i in range(1, len(series)):
        ewma_std[i] = np.std(residuals[:i + 1])

    # Anomaly condition
    anomalies = np.abs(residuals) > threshold * ewma_std

    return anomalies, ewma, ewma_std
