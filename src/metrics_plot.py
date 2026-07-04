import os
import matplotlib.pyplot as plt
import numpy as np


def plot_metrics_comparison(metrics_dict, save_path):
    """
    Plot precision, recall, and F1-score comparison as a grouped bar chart.
    """
    methods = list(metrics_dict.keys())

    # Extract metrics for each method
    precision = [metrics_dict[m]["precision"] for m in methods]
    recall = [metrics_dict[m]["recall"] for m in methods]
    f1 = [metrics_dict[m]["f1_score"] for m in methods]

    # Set label positions and bar width
    x = np.arange(len(methods))
    width = 0.25

    plt.figure(figsize=(12, 6))

    # Plot grouped bars with the custom pink/purple palette
    plt.bar(x - width, precision, width, label="Precision", color="#FFC0CB")
    plt.bar(x, recall, width, label="Recall", color="#FF69B4")
    plt.bar(x + width, f1, width, label="F1-score", color="#800080")

    # Configure axes and layout
    plt.xticks(x, methods, rotation=15)
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.title("Anomaly Detection Metrics Comparison", fontsize=14, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")

    # Save the generated figure
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()