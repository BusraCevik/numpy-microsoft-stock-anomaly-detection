import os
import numpy as np
import csv

from src.data_preparation import prepare_stock_data
from src.utils import plot_anomalies
from src.zscore import zscore_anomaly_detection
from src.rolling_mean import rolling_mean_anomaly_detection
from src.iqr import iqr_anomaly_detection
from src.ewma import ewma_anomaly_detection
from src.compare import compare_anomaly_methods
from src.metrics import create_pseudo_ground_truth, compute_metrics
from src.metrics_plot import plot_metrics_comparison

import plotly.graph_objects as go

"""
Main pipeline for anomaly detection on Microsoft stock data.
Runs multiple algorithms, compares results, evaluates metrics,
and generates both static and interactive visualizations.
"""

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA = os.path.join(BASE_DIR, "data", "raw", "Microsoft_Stock.csv")
PROCESSED_DATA = os.path.join(BASE_DIR, "data", "processed", "msft_close.npy")

PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
INTERACTIVE_DIR = os.path.join(BASE_DIR, "docs")

METRICS_PLOT_PATH = os.path.join(PLOTS_DIR, "metrics_comparison.png")
INTERACTIVE_HTML_PATH = os.path.join(INTERACTIVE_DIR, "index.html")

# -------------------------------------------------
# Parameters
# -------------------------------------------------
Z_THRESHOLD = 1.2
ROLLING_WINDOW = 20
ROLLING_THRESHOLD = 3.0
IQR_MULTIPLIER = 1.0
EWMA_ALPHA = 0.3
EWMA_THRESHOLD = 3.0


def generate_interactive_dashboard(dates, prices, anomaly_results):
    """
    Generate an interactive Plotly dashboard comparing
    anomalies detected by different methods.
    """
    fig = go.Figure()

    # Base price line (blue)
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=prices,
            mode="lines",
            name="Stock Price",
            line=dict(color="blue", width=2)
        )
    )

    # Define colors for each algorithm
    colors = {
        "Z-Score": "pink",
        "Rolling Mean": "purple",
        "IQR": "orange",
        "EWMA": "green"
    }

    # Add anomaly traces
    for method, anomalies in anomaly_results.items():
        fig.add_trace(
            go.Scatter(
                x=dates[anomalies],
                y=prices[anomalies],
                mode="markers",
                name=f"{method} Anomalies",
                marker=dict(color=colors.get(method, "red"), size=8, symbol="circle")
            )
        )

    # Layout adjustments for better spacing
    fig.update_layout(
        title="Microsoft Stock – Anomaly Detection Comparison",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Methods",
        template="plotly_white",
        xaxis=dict(
            tickangle=-45,
            tickmode="auto",
            nticks=20,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            fixedrange=False
        ),
        width=1200,
        height=600
    )

    os.makedirs(INTERACTIVE_DIR, exist_ok=True)
    fig.write_html(INTERACTIVE_HTML_PATH)



def main():
    # -------------------------------------------------
    #  Data preparation
    # -------------------------------------------------
    dates, prices = prepare_stock_data(
        input_path=RAW_DATA,
        output_path=PROCESSED_DATA,
        column="Close",
        log_transform=False
    )

    # -------------------------------------------------
    #  Run anomaly detection algorithms
    # -------------------------------------------------
    z_anom, _ = zscore_anomaly_detection(prices, Z_THRESHOLD)
    r_anom, _, _ = rolling_mean_anomaly_detection(prices, ROLLING_WINDOW, ROLLING_THRESHOLD)
    iqr_anom, _, _ = iqr_anomaly_detection(prices, IQR_MULTIPLIER)
    ewma_anom, _, _ = ewma_anomaly_detection(prices, EWMA_ALPHA, EWMA_THRESHOLD)

    anomaly_results: dict[str, np.ndarray] = {
        "Z-Score": z_anom,
        "Rolling Mean": r_anom,
        "IQR": iqr_anom,
        "EWMA": ewma_anom
    }

    # -------------------------------------------------
    #  Static PNG plots
    # -------------------------------------------------
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for method, anomalies in anomaly_results.items():
        plot_anomalies(
            dates,
            prices,
            anomalies,
            title=f"{method} Anomaly Detection",
            save_path=os.path.join(PLOTS_DIR, f"{method.lower().replace(' ', '_')}.png")
        )

    # -------------------------------------------------
    #  Compare anomaly counts
    # -------------------------------------------------
    summary = compare_anomaly_methods(anomaly_results)
    print("Anomaly count comparison:", summary)

    # -------------------------------------------------
    #  Metrics evaluation (pseudo ground truth)
    # -------------------------------------------------
    ground_truth = create_pseudo_ground_truth(
        list(anomaly_results.values()),
        min_votes=2
    )

    metrics_results: dict[str, dict] = {}
    for method, anomalies in anomaly_results.items():
        metrics_results[method] = compute_metrics(anomalies, ground_truth)

    # -------------------------------------------------
    #  Metrics comparison PNG
    # -------------------------------------------------
    plot_metrics_comparison(metrics_results, METRICS_PLOT_PATH)

    # -------------------------------------------------
    #  Interactive dashboard
    # -------------------------------------------------
    generate_interactive_dashboard(dates, prices, anomaly_results)

    print("Pipeline completed successfully.")

    # -------------------------------------------------
    #  Save reports (CSV)
    # -------------------------------------------------
    REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    #  Anomaly counts
    counts_path = os.path.join(REPORTS_DIR, "anomaly_counts.csv")
    with open(counts_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Total_Anomalies"])
        for method, anomalies in anomaly_results.items():
            writer.writerow([method, int(np.sum(anomalies))])

    #  Metrics
    metrics_path = os.path.join(REPORTS_DIR, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Method", "Precision", "Recall", "F1", "TP", "FP", "FN", "TN"]
        writer.writerow(header)
        for method, metric in metrics_results.items():
            writer.writerow([
                method,
                metric["precision"],
                metric["recall"],
                metric["f1_score"],
                metric["tp"],
                metric["fp"],
                metric["fn"],
                metric["tn"]
            ])


if __name__ == "__main__":
    main()
