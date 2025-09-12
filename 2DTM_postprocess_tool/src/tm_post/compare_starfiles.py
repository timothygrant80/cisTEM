import pandas as pd
from scipy.spatial import cKDTree
import re
from tm_post.starfile import load_particle_starfile
from tm_post.geometry import return_euler_err

def extract_match_key(filename, pattern):
    match = re.search(pattern, filename)
    return match.group(1) if match else None

def compare_starfiles_for_matched_peaks(
    starfile_a: str,
    starfile_b: str,
    d_xy_cutoff: float = 10.0,         # in Å
    euler_err_cutoff: float = 5.0,     # in degrees
    pattern=r"mc2_[12]x_(.*?frames)"
) -> pd.DataFrame:
    """
    Compare two STAR DataFrames (df_a and df_b) and return matched peaks.
    Accounts for pixel size differences between the two searches.
    """
    # Load the starfile
    df_a = load_particle_starfile(starfile_a)
    df_b = load_particle_starfile(starfile_b)
    df_a["match_key"] = df_a["ORIGINAL_IMAGE_FILENAME"].apply(lambda x: extract_match_key(x, pattern))
    df_b["match_key"] = df_b["ORIGINAL_IMAGE_FILENAME"].apply(lambda x: extract_match_key(x, pattern))

    # Find overlapped images
    common_keys = set(df_a["match_key"]) & set(df_b["match_key"])
    print(f"[INFO] Found {len(common_keys)} overlapping images between the two searches.")

    matched_idx_b = []
    # Iterate through each common key
    for key in common_keys:
        peaks_a = df_a[df_a["match_key"] == key]
        peaks_b = df_b[df_b["match_key"] == key]

        if peaks_a.empty or peaks_b.empty:
            continue

        coords_a = peaks_a[["ORIGX", "ORIGY"]].values
        coords_b = peaks_b[["ORIGX", "ORIGY"]].values

        tree_b = cKDTree(coords_b)
        dists, idxs_b = tree_b.query(coords_a)

        for i, (dist, j) in enumerate(zip(dists, idxs_b)):
            if dist >= d_xy_cutoff:
                continue

            row_a = peaks_a.iloc[i]
            row_b = peaks_b.iloc[j]

            err = return_euler_err(
                gt_psi=row_a["PSI"], gt_theta=row_a["THETA"], gt_phi=row_a["PHI"],
                tm_psi=row_b["PSI"], tm_theta=row_b["THETA"], tm_phi=row_b["PHI"]
            )

            if err < euler_err_cutoff:
                idx_b = peaks_b.index[j]
                matched_idx_b.append(idx_b)

    # Annotate matched rows in df_b
    df_b_matched = df_b.loc[matched_idx_b].copy()
    df_b_matched.drop(columns=["match_key"], inplace=True)
    print(f"[INFO] Found {len(df_b_matched)} matched peaks.")

    return df_b_matched

    