import numpy as np


def iqr_anomaly_detection(series, multiplier=1.5):
    """
    Detect anomalies using the Interquartile Range (IQR) method.
    """
    # Calculate the first and third quartiles
    q1 = np.percentile(series, 25)
    q3 = np.percentile(series, 75)

    # Compute the interquartile range
    iqr = q3 - q1

    # Define lower and upper anomaly bounds
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    # Identify data points falling outside the limits
    anomalies = (series < lower_bound) | (series > upper_bound)

    return anomalies, lower_bound, upper_bound