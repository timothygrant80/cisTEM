import argparse
import pandas as pd
import tm_post.starfile as starfile
from tm_post.compare_starfiles import compare_starfiles_for_matched_peaks

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare two STAR files and extract matched peaks based on spatial and angular thresholds."
    )
    parser.add_argument('--starfile_a', type=str, required=True, help="Path to first STAR file (e.g. bin2x).")
    parser.add_argument('--starfile_b', type=str, required=True, help="Path to second STAR file (e.g. bin1x).")
    parser.add_argument('--d_xy_cutoff', type=float, default=10.0, help="Maximum XY distance in Å for matching.")
    parser.add_argument('--euler_err_cutoff', type=float, default=5.0, help="Maximum Euler angle error in degrees.")
    parser.add_argument('--pattern', type=str, default=r"mc2_[12]x_(.*?frames)", help="Regex pattern for extracting match key.")
    parser.add_argument('--output', type=str, required=True, help="Path to output STAR file with matched peaks.")

    return parser.parse_args()

def main():
    args = parse_arguments()

    print("[INFO] Comparing starfiles...")
    matched_df = compare_starfiles_for_matched_peaks(
        starfile_a=args.starfile_a,
        starfile_b=args.starfile_b,
        d_xy_cutoff=args.d_xy_cutoff,
        euler_err_cutoff=args.euler_err_cutoff,
        pattern=args.pattern
    )

    if matched_df.empty:
        print("[INFO] No matching peaks found.")
        return

    # Convert matched_df to STAR format (with dummy column) and write with standard header
    matched_df_star = starfile.add_star_dummy_column(matched_df)
    header_lines = starfile.read_tm_package_starfile_header()
    starfile.write_starfile_with_headers(args.output, header_lines, matched_df_star)
    print(f"[INFO] Matched particles saved to: {args.output}")

if __name__ == "__main__":
    main()