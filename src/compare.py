import numpy as np

"""
This module compares anomaly detection algorithms by
counting detected anomalies and providing summary statistics.
"""


def compare_anomaly_methods(results_dict):
    """
    Compare multiple anomaly detection results.

    Args:
        results_dict (dict):
            {
                "method_name": boolean_anomaly_array
            }

    Returns:
        summary (dict): Method name mapped to number of anomalies.
    """
    summary = {}

    for method, anomalies in results_dict.items():
        summary[method] = int(np.sum(anomalies))

    return summary
