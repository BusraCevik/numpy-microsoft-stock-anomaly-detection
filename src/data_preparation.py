import pandas as pd
import numpy as np
import os


def prepare_stock_data(input_path, output_path, column="Close", log_transform=False):
    """
    Load raw CSV data, sort by date, clean missing values, and save as NumPy array.
    """
    # Load dataset
    df = pd.read_csv(input_path)

    # Convert date column to datetime format
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort data chronologically
    df = df.sort_values("Date")

    # Remove rows with missing target column values
    df = df.dropna(subset=[column])

    # Extract target values as float array
    values = df[column].values.astype(float)

    # Apply log transformation if requested
    if log_transform:
        values = np.log(values)

    # Save processed numpy array
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, values)

    return df["Date"].values, values