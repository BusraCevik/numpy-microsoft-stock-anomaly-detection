import os
import matplotlib.pyplot as plt
import numpy as np

def plot_metrics_comparison(metrics_dict, save_path):
    """
    Plot precision, recall, and F1-score comparison as a bar chart
    with pink/purple color palette.

    Args:
        metrics_dict (dict):
            {
              "Z-Score": {"precision": ..., "recall": ..., "f1_score": ...},
              "Rolling Mean": {...},
              "IQR": {...},
              "EWMA": {...}
            }
        save_path (str): Path to save the PNG plot.
    """
    methods = list(metrics_dict.keys())

    precision = [metrics_dict[m]["precision"] for m in methods]
    recall = [metrics_dict[m]["recall"] for m in methods]
    f1 = [metrics_dict[m]["f1_score"] for m in methods]

    x = np.arange(len(methods))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, precision, width, label="Precision", color="#FFC0CB")
    plt.bar(x, recall, width, label="Recall", color="#FF69B4")
    plt.bar(x + width, f1, width, label="F1-score", color="#800080")

    plt.xticks(x, methods, rotation=15)
    plt.ylabel("Score")
    plt.title("Anomaly Detection Metrics Comparison")
    plt.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
