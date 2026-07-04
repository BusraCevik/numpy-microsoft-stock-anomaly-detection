import numpy as np


def ewma_anomaly_detection(series, alpha=0.3, threshold=3.0):
    """
    Detect anomalies using Exponentially Weighted Moving Average (EWMA).
    """
    ewma = np.zeros_like(series, dtype=float)
    ewma[0] = series[0]

    # Calculate EWMA values sequentially
    for i in range(1, len(series)):
        ewma[i] = alpha * series[i] + (1 - alpha) * ewma[i - 1]

    # Calculate residuals between actual values and EWMA
    residuals = series - ewma

    # Calculate the global standard deviation of residuals for stability
    residual_std = np.std(residuals)
    if residual_std == 0:
        residual_std = 1e-8

    # Create a broadcasted array for compatibility with main.py output
    ewma_std = np.full_like(series, fill_value=residual_std, dtype=float)

    # Mark anomalies where absolute residuals exceed the threshold
    anomalies = np.abs(residuals) > threshold * ewma_std

    return anomalies, ewma, ewma_std