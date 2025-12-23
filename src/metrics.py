import numpy as np

"""
This module computes evaluation metrics such as
precision, recall, F1-score, and confusion matrix
for anomaly detection using pseudo ground truth.
"""


def create_pseudo_ground_truth(anomaly_results, min_votes=2):
    """
    Create pseudo ground truth based on consensus.

    Args:
        anomaly_results (list of np.ndarray): List of anomaly boolean arrays.
        min_votes (int): Minimum number of votes to mark a point as anomaly.

    Returns:
        ground_truth (np.ndarray): Boolean ground truth array.
    """
    votes = np.sum(anomaly_results, axis=0)
    return votes >= min_votes


def compute_metrics(predictions, ground_truth):
    """
    Compute precision, recall, and F1-score.

    Args:
        predictions (np.ndarray): Predicted anomaly array.
        ground_truth (np.ndarray): Ground truth anomaly array.

    Returns:
        metrics (dict): Precision, Recall, F1-score, TP, FP, FN, TN
    """
    tp = np.sum((predictions == 1) & (ground_truth == 1))
    fp = np.sum((predictions == 1) & (ground_truth == 0))
    fn = np.sum((predictions == 0) & (ground_truth == 1))
    tn = np.sum((predictions == 0) & (ground_truth == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }
