"""
Utilities for handling threshold values in cisTEM outputs.

This module provides functions for extracting threshold values from cisTEM output files.
"""

import os
import re


def extract_threshold_value(hist_file_path):
    """
    Extract the threshold value from the histogram text file.

    The threshold value is in the first line of the file, which starts with
    "# Expected threshold = ". This function extracts the numerical value following this marker.

    Args:
        hist_file_path (str): Path to the histogram text file

    Returns:
        float: The extracted threshold value

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the threshold value cannot be found or parsed
    """
    if not os.path.exists(hist_file_path):
        raise FileNotFoundError(f"Histogram file not found at {hist_file_path}")

    with open(hist_file_path, 'r') as f:
        first_line = f.readline().strip()

    # Use regex to extract the threshold value from the line
    # Looking for a pattern like "# Expected threshold = 6.90"
    match = re.search(r'# Expected threshold = ([\d.e+-]+)', first_line)
    if not match:
        raise ValueError(f"Could not find threshold value in {hist_file_path}")

    threshold_value = float(match.group(1))
    return threshold_value