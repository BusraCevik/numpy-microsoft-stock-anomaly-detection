import pandas as pd
import numpy as np
import os

"""
This module handles preprocessing of raw stock data.
It sorts the data by date, removes missing values, 
and optionally applies a log transformation. The processed
data is saved as a NumPy array for easy use in anomaly detection algorithms.
"""


def prepare_stock_data(input_path, output_path, column="Close", log_transform=False):
    """
    Prepares stock data for anomaly detection.

    Steps:
    1. Load CSV data.
    2. Convert 'Date' column to datetime.
    3. Sort by date.
    4. Remove missing values in the selected column.
    5. Convert the selected column to NumPy array of floats.
    6. Optionally apply log transformation.
    7. Save the processed array to disk.

    Args:
        input_path (str): Path to raw CSV file.
        output_path (str): Path to save processed NumPy array.
        column (str): Column to use (default: 'Close').
        log_transform (bool): Apply log transform if True.

    Returns:
        dates (np.ndarray): Array of datetime objects.
        values (np.ndarray): Array of processed stock values.
    """
    df = pd.read_csv(input_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df = df.dropna(subset=[column])

    values = df[column].values.astype(float)

    if log_transform:
        values = np.log(values)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, values)

    return df["Date"].values, values
