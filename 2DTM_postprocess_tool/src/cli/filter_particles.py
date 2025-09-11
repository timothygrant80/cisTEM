import argparse
import tm_post.starfile as starfile
import tm_post.database as db
from tm_post.filters import apply_filter

def parse_arguments():
    parser = argparse.ArgumentParser(description="Filter peaks using TM results and image quality.")
    # read job information from database file
    parser.add_argument('--star_file', type=str, required=True, help="Path to the particle starfile.")
    parser.add_argument('--db_file', type=str, required=True, help="Path to the database file.")
    parser.add_argument('--tm_job_id', type=int, required=True, help="Template match job ID.")
    parser.add_argument('--ctf_job_id', type=int, required=True, help="CTF job ID.")
    parser.add_argument('--pixel_size', type=float, required=True, help="Pixel size in Angstroms.")

    # read particle information from .star file (extract_peaks.py output)
    parser.add_argument('--avg_cutoff_lb', type=float, default=None, help="Lower bound for average cutoff.")
    parser.add_argument('--sd_cutoff_ub', type=float, default=None, help="Upper bound for SD cutoff.")
    parser.add_argument('--pval_cutoff_lb', type=float, default=None, help="Lower bound for p-value cutoff.")
    parser.add_argument('--snr_cutoff_ub', type=float, default=None, help="Upper bound for SNR (optional).")
    parser.add_argument('--snr_cutoff_lb', type=float, default=None, help="Lower bound for SNR (optional).")
    parser.add_argument('--filter_by_image_thickness', action="store_true", help="Use thickness to filter good micrographs? (default: False)")
    parser.add_argument('--thickness_cutoff_lb', type=float, default=None, help="Lower bound for thickness cutoff (A).")
    parser.add_argument('--thickness_cutoff_ub', type=float, default=None, help="Upper bound for thickness cutoff (A).")
    parser.add_argument('--filter_by_angular_invariance', action="store_true", help="Use angular invariance to filter good particles (default: False)?")
    parser.add_argument('--geodesic_r', type=int, default=None, help="Radius in pixels for local patch.")
    parser.add_argument('--geodesic_threads', type=int, default=None, help="Number of threads for geodesic computation.")
    parser.add_argument('--geodesic_method', type=str, default=None, help="Method for geodesic filtering ('quantile' or 'cutoff').")
    parser.add_argument('--geodesic_threshold', type=float, default=None, help="Threshold value for geodesic filtering.")

    parser.add_argument('--ctf_fitting_score_lb', type=float, default=None, help="Lower bound for CTF fitting score.")
    parser.add_argument('--ctf_fitting_score_ub', type=float, default=None, help="Upper bound for CTF fitting score.")

    
    parser.add_argument('--output', type=str, required=True, help="Path to the output star file.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Load particle information from star file
    print("[INFO] Loading peak file...")
    df_peaks = starfile.load_particle_starfile(args.star_file)

    # Extract header lines
    #header_lines, _ = starfile.extract_header_lines(args.star_file)

    # Load database information
    print("[INFO] Loading database...")
    result = db.get_info_from_cistem_database(
        args.db_file, args.tm_job_id, args.ctf_job_id
    )

    # Extract relevant data
    image_list = result["image_list"]
    psi_list = result["PSI_OUTPUT_FILE"]
    theta_list = result["THETA_OUTPUT_FILE"]
    phi_list = result["PHI_OUTPUT_FILE"]
    df_ctf = result["df_ctf"]
    df_info = result["df_info"]

    # Apply filters
    if args.filter_by_image_thickness:
        if args.thickness_cutoff_lb is None:
            args.thickness_cutoff_lb = 0.0
        if args.thickness_cutoff_ub is None:
            args.thickness_cutoff_ub = 500.0

    if args.filter_by_angular_invariance:
        if args.geodesic_r is None:
            args.geodesic_r = 4
        if args.geodesic_threads is None:
            args.geodesic_threads = 8
        if args.geodesic_method is None:
            args.geodesic_method = "quantile"
        if args.geodesic_threshold is None:
            args.geodesic_threshold = 0.8

    filtered_df, all_df = apply_filter(
    df=df_peaks,
    image_list=image_list,
    psi_list=psi_list,
    theta_list=theta_list,
    phi_list=phi_list,
    pixel_size=args.pixel_size,
    df_ctf=df_ctf,
    df_info=df_info,
    avg_cutoff_lb=args.avg_cutoff_lb,
    sd_cutoff_ub=args.sd_cutoff_ub,
    pval_cutoff_lb=args.pval_cutoff_lb,
    snr_cutoff_ub=args.snr_cutoff_ub,
    snr_cutoff_lb=args.snr_cutoff_lb,
    filter_by_image_thickness=args.filter_by_image_thickness,
    thickness_lb=args.thickness_cutoff_lb,
    thickness_ub=args.thickness_cutoff_ub,
    ctf_fitting_score_lb=args.ctf_fitting_score_lb,
    ctf_fitting_score_ub=args.ctf_fitting_score_ub,
    filter_by_angular_invariance=args.filter_by_angular_invariance,
    geodesic_r=args.geodesic_r,
    geodesic_threads=args.geodesic_threads,
    geodesic_method=args.geodesic_method,
    geodesic_threshold=args.geodesic_threshold
    )
    
    # Save filtered STAR file with updated SCORE (no extra metadata columns)
    columns_to_keep = ["ORIGINAL_IMAGE_FILENAME","ORIGX","ORIGY","AVG","SD","PVALUE","ZSCORE","SNR"]
    meta_df = all_df[columns_to_keep].copy()

    # Convert filtered DataFrame to lines with empty column
    starfile_df_star = starfile.add_star_dummy_column(filtered_df)

    # Write output with headers
    header_lines = starfile.read_tm_package_starfile_header()  # provide default STAR header
    starfile.write_starfile_with_headers(args.output, header_lines, starfile_df_star)
    print(f"[INFO] Filtered data saved to {args.output}")

    metadata_file = args.output.replace(".star", "_metadata.txt")
    meta_df.to_csv(metadata_file, sep="\t", index=False, float_format="%.2f")

if __name__ == "__main__":
    main()