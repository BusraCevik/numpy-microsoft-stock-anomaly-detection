import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

"""
This module contains utility functions for loading stock data, 
saving NumPy arrays, and plotting anomalies detected in time series data.

Functions:
- load_stock_data: Loads stock CSV data and returns dates and a selected column as a NumPy array.
- save_numpy_array: Saves a NumPy array to a specified path.
- plot_anomalies: Plots the stock time series with anomalies highlighted.
"""


def load_stock_data(csv_path, column="Close"):
    """
    Load stock data from a CSV file.

    Args:
        csv_path (str): Path to the CSV file containing stock data.
        column (str): Name of the column to use (default is "Close").

    Returns:
        dates (pd.Series): Series of datetime objects.
        values (np.ndarray): Array of float values for the selected column.
    """
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    values = df[column].values.astype(float)
    return df["Date"], values


def save_numpy_array(array, path):
    """
    Save a NumPy array to a specified file path.
    Creates directories if they do not exist.

    Args:
        array (np.ndarray): The NumPy array to save.
        path (str): File path to save the array.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)


def plot_anomalies(dates, values, anomalies, title, save_path):
    """
    Plot a time series and highlight anomalies in red.

    Args:
        dates (array-like): Array of datetime values.
        values (array-like): Stock price values.
        anomalies (array-like or boolean mask): Boolean mask indicating anomalies.
        title (str): Plot title.
        save_path (str): Path to save the plot image.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, values, label="Stock Price")
    plt.scatter(dates[anomalies], values[anomalies], color="red", label="Anomaly")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
