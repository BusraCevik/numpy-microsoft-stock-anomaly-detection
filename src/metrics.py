import numpy as np


def create_pseudo_ground_truth(anomaly_results, min_votes=2):
    """
    Create pseudo ground truth based on consensus voting.
    """
    # Sum the binary/boolean votes across all algorithms
    votes = np.sum(anomaly_results, axis=0)

    # Mark as true ground truth if votes meet the minimum threshold
    return votes >= min_votes


def compute_metrics(predictions, ground_truth):
    """
    Compute evaluation metrics: precision, recall, and F1-score.
    """
    # Ensure inputs are evaluated as boolean masks
    pred_mask = predictions.astype(bool)
    gt_mask = ground_truth.astype(bool)

    # Compute confusion matrix components
    tp = int(np.sum(pred_mask & gt_mask))
    fp = int(np.sum(pred_mask & ~gt_mask))
    fn = int(np.sum(~pred_mask & gt_mask))
    tn = int(np.sum(~pred_mask & ~gt_mask))

    # Calculate precision, recall, and F1-score with division safety
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