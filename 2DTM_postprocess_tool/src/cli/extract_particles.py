import argparse
import pandas as pd
import tm_post.database as db
from tm_post.database import load_tm_images_from_db
from tm_post.extract import extract_particles_from_2dtm_search
from tm_post.starfile import write_starfile_with_headers, read_tm_package_starfile_header

def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract peaks from 2DTM searches.")

    parser.add_argument('--db_file', type=str, required=True, help="Path to the database file.")
    parser.add_argument('--tm_job_id', type=int, required=True, help="Template match job ID.")
    parser.add_argument('--ctf_job_id', type=int, required=True, help="CTF job ID.")
    
    parser.add_argument('--min_peak_radius', type=int, default=10, help="Cutoff for XY distance.")
    parser.add_argument('--exclude_borders', type=int, default=35, help="Exclude borders in the image.")
    
    parser.add_argument('--local_max_filter', type=str, default="zscore", choices=["zscore", "snr"], help="Local max filter to use.")
    parser.add_argument('--metric', type=str, default="pval", choices=["pval", "zscore", "snr"], help="Metric to use for filtering.")
    parser.add_argument('--metric_cutoff', type=float, default=8.0, help="Selected metric cutoff.")
    parser.add_argument('--pixel_size', type=float, required=True, default=1.0, help="Wanted pixel size in final stack.")
    parser.add_argument('--threads', type=int, default=4, help="Number of threads for parallel processing.")

    parser.add_argument('--quadrants', type=int, default=1, help="Number of quadrants to use for filtering.")
    
    parser.add_argument('--output', type=str, required=True, help="Path to the output star file.")

    return parser.parse_args()

def main():
    args = parse_arguments()

    # Load database information
    print("[INFO] Loading TM image data from database...")
    tm_images, df_ctf, df_info = load_tm_images_from_db(
        db_file=args.db_file,
        tm_job_id=args.tm_job_id,
        ctf_job_id=args.ctf_job_id
    )
    
    print(f"[INFO] Running extraction on {len(tm_images)} images...")

    df_star = extract_particles_from_2dtm_search(
        tm_images=tm_images,
        local_max_filter=args.local_max_filter,
        metric=args.metric,
        metric_cutoff=args.metric_cutoff,
        pixel_size=args.pixel_size,
        max_threads=args.threads,
        df_ctf=df_ctf,
        df_info=df_info,
        ctf_job_id=args.ctf_job_id,
        min_radius=args.min_peak_radius,
        exclude_borders=args.exclude_borders,
        q=args.quadrants
    )

    print("[INFO] Writing STAR file...")
    header_lines = read_tm_package_starfile_header()  # provide default STAR header
    write_starfile_with_headers(args.output, header_lines, df_star)
    print(f"[INFO] Done. Extracted particles saved to {args.output}")
    

if __name__ == "__main__":
    main()