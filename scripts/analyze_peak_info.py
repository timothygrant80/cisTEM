#!/usr/bin/env python3
"""
Analyze peak_info output files from cisTEM match_template.

This script parses peak_info files (8 columns: x_pos, y_pos, defocus,
corrected_peak_height, original_score, above_threshold, sub_pixel_x, sub_pixel_y)
and creates a scatter plot of percent peak height change vs subpixel radius.

Usage:
    python analyze_peak_info.py <peak_info_file1.txt> [peak_info_file2.txt ...]
    python analyze_peak_info.py --output plot.png <peak_info_file1.txt>
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuration constants
ENABLE_QUADRATIC_FIT = False  # Set to True to show quadratic fit on plot


def parse_peak_info_file(filepath: Path) -> pd.DataFrame:
    """Parse a peak_info file into a DataFrame."""
    # Skip comment lines (lines starting with #)
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            values = line.split()
            if len(values) == 8:
                data.append([float(v) for v in values])

    if not data:
        raise ValueError(f"No valid data found in {filepath}")

    columns = ['x_pos', 'y_pos', 'defocus', 'corrected_peak_height',
               'original_score', 'above_threshold', 'sub_pixel_x', 'sub_pixel_y']
    df = pd.DataFrame(data, columns=columns)
    df['source_file'] = str(filepath)
    return df


def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate derived metrics for analysis."""
    # Percent change: (corrected - original) / original * 100
    # Avoid division by zero
    df['percent_change'] = np.where(
        df['original_score'] != 0,
        (df['corrected_peak_height'] - df['original_score']) / df['original_score'] * 100,
        0
    )

    # Subpixel radius: sqrt(sub_pixel_x^2 + sub_pixel_y^2)
    df['subpixel_radius'] = np.sqrt(df['sub_pixel_x']**2 + df['sub_pixel_y']**2)

    return df


def create_scatter_plot(df: pd.DataFrame, output_path: Path = None, base_fontsize: int = 16):
    """Create scatter plot of percent_change vs subpixel_radius."""
    # Use a larger figure and scale fonts for high-res displays
    fig, ax = plt.subplots(figsize=(14, 11))

    # Set global font sizes relative to base_fontsize
    plt.rcParams.update({
        'font.size': base_fontsize,
        'axes.titlesize': base_fontsize * 1.4,
        'axes.labelsize': base_fontsize * 1.2,
        'xtick.labelsize': base_fontsize,
        'ytick.labelsize': base_fontsize,
        'legend.fontsize': base_fontsize,
    })

    # Filter out points with subpixel_radius > 0.5*sqrt(2) (~0.707, diagonal of half-pixel)
    max_radius = 0.5 * np.sqrt(2)
    total_before = len(df)
    df = df[df['subpixel_radius'] <= max_radius].copy()
    n_removed = total_before - len(df)

    # Separate pre-existing peaks (above_threshold=1) from recovered peaks (above_threshold=0)
    preexisting = df[df['above_threshold'] == 1]
    recovered = df[df['above_threshold'] == 0]

    # Plot points with larger markers
    marker_size = 60
    if len(preexisting) > 0:
        ax.scatter(preexisting['subpixel_radius'], preexisting['percent_change'],
                   c='blue', alpha=0.6, label=f'Pre-existing peaks (n={len(preexisting)})', s=marker_size)
    if len(recovered) > 0:
        ax.scatter(recovered['subpixel_radius'], recovered['percent_change'],
                   c='black', marker='x', alpha=0.8, label=f'Recovered peaks (n={len(recovered)})', s=marker_size, linewidths=2)

    # Add histogram of recovered peaks distribution on secondary y-axis
    if len(recovered) > 0:
        ax2 = ax.twinx()

        # Get unique radius values (discrete due to grid) and count recovered peaks at each
        # Round to avoid floating point issues when grouping
        recovered_radii = recovered['subpixel_radius'].round(6)
        unique_radii = np.sort(recovered_radii.unique())

        # Count peaks at each radius and convert to percentage of total recovered
        counts = recovered_radii.value_counts().sort_index()
        percentages = (counts / len(recovered)) * 100

        # Plot as step histogram (bar chart at discrete positions)
        bar_width = 0.012  # Narrow bars for discrete values
        ax2.bar(percentages.index, percentages.values, width=bar_width,
                alpha=0.3, color='orange', edgecolor='darkorange', linewidth=1,
                label='Recovered distribution')

        ax2.set_ylabel('% of recovered peaks', color='darkorange')
        ax2.tick_params(axis='y', labelcolor='darkorange')
        ax2.set_ylim(0, None)  # Start from 0

        # Add to legend (need to combine legends from both axes later)
        ax2.legend(loc='center right', fontsize=base_fontsize * 0.9)

    # Calculate and plot lines of best fit forced through origin (0,0)
    x = df['subpixel_radius'].values
    y = df['percent_change'].values
    r_squared = np.nan
    r_squared_quad = np.nan
    slope = np.nan

    # Linear regression through origin: y = mx
    # Least squares solution: m = sum(x*y) / sum(x^2)
    if len(x) > 1:
        sum_xy = np.sum(x * y)
        sum_xx = np.sum(x * x)
        slope = sum_xy / sum_xx if sum_xx != 0 else 0

        # Calculate R² for regression through origin
        y_pred = slope * x
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum(y ** 2)  # For origin-constrained regression, use sum(y^2) not sum((y-mean)^2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Plot linear fit line from origin
        x_fit = np.linspace(0, x.max(), 100)
        y_fit = slope * x_fit
        ax.plot(x_fit, y_fit, 'k-', linewidth=2.5,
                label=f'Linear fit (R²={r_squared:.3f})')

        # Quadratic regression through origin: y = ax² + bx
        if ENABLE_QUADRATIC_FIT:
            X_quad = np.column_stack([x**2, x])
            coeffs_quad, residuals, rank, s = np.linalg.lstsq(X_quad, y, rcond=None)
            a_quad, b_quad = coeffs_quad

            # Calculate R² for quadratic fit
            y_pred_quad = a_quad * x**2 + b_quad * x
            ss_res_quad = np.sum((y - y_pred_quad) ** 2)
            r_squared_quad = 1 - (ss_res_quad / ss_tot) if ss_tot != 0 else 0

            # Plot quadratic fit from origin
            y_fit_quad = a_quad * x_fit**2 + b_quad * x_fit
            ax.plot(x_fit, y_fit_quad, 'g--', linewidth=2.5,
                    label=f'Quadratic fit (R²={r_squared_quad:.3f})')

    # Calculate statistics
    overall_avg = df['percent_change'].mean()
    preexisting_avg = preexisting['percent_change'].mean() if len(preexisting) > 0 else np.nan
    recovered_avg = recovered['percent_change'].mean() if len(recovered) > 0 else np.nan

    # Calculate average shift radius
    overall_radius_avg = df['subpixel_radius'].mean()
    recovered_radius_avg = recovered['subpixel_radius'].mean() if len(recovered) > 0 else np.nan

    # Count peaks at zero shift (they overlap) and get their average percent change
    n_preexisting_at_zero = (preexisting['subpixel_radius'] == 0).sum() if len(preexisting) > 0 else 0
    n_recovered_at_zero = (recovered['subpixel_radius'] == 0).sum() if len(recovered) > 0 else 0
    avg_pct_at_zero = df[df['subpixel_radius'] == 0]['percent_change'].mean() if (df['subpixel_radius'] == 0).any() else np.nan

    # Add statistics as text
    stats_text = ""
    if n_removed > 0:
        stats_text += f"Removed {n_removed} pts (radius > {max_radius:.3f})\n"
    if not np.isnan(r_squared):
        stats_text += f"Linear R² = {r_squared:.3f}\n"
    if not np.isnan(r_squared_quad):
        stats_text += f"Quadratic R² = {r_squared_quad:.3f}\n"
    stats_text += f"Overall avg change: {overall_avg:.2f}%\n"
    if not np.isnan(preexisting_avg):
        stats_text += f"Pre-existing avg change: {preexisting_avg:.2f}%\n"
    if not np.isnan(recovered_avg):
        stats_text += f"Recovered avg change: {recovered_avg:.2f}%\n"
    stats_text += f"Avg shift radius (all): {overall_radius_avg:.3f}\n"
    if not np.isnan(recovered_radius_avg):
        stats_text += f"Avg shift radius (recovered): {recovered_radius_avg:.3f}"

    # Place text in upper left with larger font
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=base_fontsize * 1.1,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Annotate number of peaks at zero shift (since markers overlap)
    if n_preexisting_at_zero > 0 or n_recovered_at_zero > 0:
        zero_label = f"n @ 0: {n_preexisting_at_zero} pre, {n_recovered_at_zero} rec\navg: {avg_pct_at_zero:.2f}%"
        # Position annotation slightly offset from origin
        ax.annotate(zero_label, xy=(0, 0), xytext=(0.05, ax.get_ylim()[0] + 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0])),
                    fontsize=base_fontsize * 0.9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='gray'))

    ax.set_xlabel('Sub-pixel radius (pixels)')
    ax.set_ylabel('Percent peak height change (%)')
    ax.set_title('Peak Height Change vs Sub-pixel Offset')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    return fig, ax


def print_summary(df: pd.DataFrame):
    """Print summary statistics to stdout."""
    preexisting = df[df['above_threshold'] == 1]
    recovered = df[df['above_threshold'] == 0]

    print("\n" + "="*60)
    print("PEAK ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total peaks analyzed: {len(df)}")
    print(f"  Pre-existing peaks (above threshold): {len(preexisting)}")
    print(f"  Recovered peaks (below threshold): {len(recovered)}")
    print()

    print("Percent Peak Height Change:")
    print(f"  Overall average: {df['percent_change'].mean():.2f}%")
    print(f"  Overall std dev: {df['percent_change'].std():.2f}%")
    if len(preexisting) > 0:
        print(f"  Pre-existing average: {preexisting['percent_change'].mean():.2f}%")
    if len(recovered) > 0:
        print(f"  Recovered average: {recovered['percent_change'].mean():.2f}%")
    print()

    print("Sub-pixel Radius:")
    print(f"  Overall average: {df['subpixel_radius'].mean():.3f} pixels")
    print(f"  Overall max: {df['subpixel_radius'].max():.3f} pixels")
    if len(preexisting) > 0:
        print(f"  Pre-existing average: {preexisting['subpixel_radius'].mean():.3f} pixels")
    if len(recovered) > 0:
        print(f"  Recovered average: {recovered['subpixel_radius'].mean():.3f} pixels")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze peak_info output files from cisTEM match_template')
    parser.add_argument('files', nargs='+', type=Path,
                        help='One or more peak_info files to analyze')
    parser.add_argument('--output', '-o', type=Path, default=None,
                        help='Output path for plot (if not specified, displays interactively)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plot generation, only print summary')
    parser.add_argument('--fontsize', '-f', type=int, default=16,
                        help='Base font size for plot (default: 16, increase for high-res displays)')

    args = parser.parse_args()

    # Parse all input files
    all_data = []
    for filepath in args.files:
        if not filepath.exists():
            print(f"Warning: File not found: {filepath}", file=sys.stderr)
            continue
        try:
            df = parse_peak_info_file(filepath)
            all_data.append(df)
            print(f"Loaded {len(df)} peaks from {filepath}")
        except Exception as e:
            print(f"Error reading {filepath}: {e}", file=sys.stderr)

    if not all_data:
        print("Error: No valid data files found", file=sys.stderr)
        sys.exit(1)

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = calculate_metrics(combined_df)

    # Print summary
    print_summary(combined_df)

    # Create plot
    if not args.no_plot:
        create_scatter_plot(combined_df, args.output, args.fontsize)


if __name__ == '__main__':
    main()
