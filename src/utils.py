import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def load_stock_data(csv_path, column="Close"):
    """
    Load stock data from a CSV file.
    """
    # Read CSV file
    df = pd.read_csv(csv_path)

    # Convert date column to datetime format
    df["Date"] = pd.to_datetime(df["Date"])

    # Extract values as a float array
    values = df[column].values.astype(float)
    return df["Date"], values


def save_numpy_array(array, path):
    """
    Save a NumPy array to a specified file path.
    """
    # Create parent directories if they do not exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)


def plot_anomalies(dates, values, anomalies, title, save_path):
    """
    Plot a time series and highlight anomalies in red.
    """
    plt.figure(figsize=(12, 6))

    # Plot baseline stock price
    plt.plot(dates, values, label="Stock Price", color="blue", alpha=0.7)

    # Highlight detected anomalies as red scatter points
    plt.scatter(dates[anomalies], values[anomalies], color="red", label="Anomaly", zorder=3)

    # Configure plot labels and style
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")

    # Auto-rotate dates to avoid overlapping
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    # Save image and close figure object
    plt.savefig(save_path)
    plt.close()