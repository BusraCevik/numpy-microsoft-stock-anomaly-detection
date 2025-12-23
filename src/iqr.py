import numpy as np

"""
This module implements anomaly detection using the
Interquartile Range (IQR) method. Data points that fall
outside the lower and upper bounds are considered anomalies.
"""


def iqr_anomaly_detection(series, multiplier=1.5):
    """
    Detect anomalies using the IQR method.

    Args:
        series (np.ndarray): Input time series data.
        multiplier (float): IQR multiplier (default: 1.5).

    Returns:
        anomalies (np.ndarray): Boolean array indicating anomalies.
        lower_bound (float): Lower threshold.
        upper_bound (float): Upper threshold.
    """
    q1 = np.percentile(series, 25)
    q3 = np.percentile(series, 75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    anomalies = (series < lower_bound) | (series > upper_bound)

    return anomalies, lower_bound, upper_bound
