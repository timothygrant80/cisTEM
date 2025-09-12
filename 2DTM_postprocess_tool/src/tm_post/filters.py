import numpy as np 
import pandas as pd
import operator

from tm_post.geodesic import calculate_all_geodesic_means


def apply_image_thickness_filter(peaks, df_ctf, df_info, cutoff_lb, cutoff_ub):
    # Get filenames of images within thickness cutoff
    df_ctf_filtered = df_ctf[
        (df_ctf['SAMPLE_THICKNESS'] > cutoff_lb) & 
        (df_ctf['SAMPLE_THICKNESS'] < cutoff_ub)
    ]
    
    filenames = df_info[df_info['IMAGE_ASSET_ID'].isin(df_ctf_filtered['IMAGE_ASSET_ID'])]['FILENAME']
    filenames = [f"'{filename}'" for filename in filenames]  # keep quotes if needed
    
    # Create a boolean mask
    mask = peaks['ORIGINAL_IMAGE_FILENAME'].isin(filenames)
    
    return mask

def apply_ctf_score_filter(peaks, df_ctf, df_info, cutoff_lb, cutoff_ub):
    df_ctf_filtered = df_ctf[
        (df_ctf['SCORE'] > cutoff_lb) &
        (df_ctf['SCORE'] < cutoff_ub)
    ]
    filenames = df_info[df_info['IMAGE_ASSET_ID'].isin(df_ctf_filtered['IMAGE_ASSET_ID'])]['FILENAME']
    filenames = [f"'{filename}'" for filename in filenames]  # keep quotes if needed
    # Create a boolean mask
    mask = peaks['ORIGINAL_IMAGE_FILENAME'].isin(filenames)
    return mask

def apply_angular_invariance_filter(df, mean_geodesic_array, method='quantile', threshold=0.95):
    """
    Create a lookup function for image thickness based on ORIGINAL_IMAGE_FILENAME.
    Returns a function that takes a filename string and returns the sample thickness.
    """
    if method == 'quantile':
        cutoff = np.nanquantile(mean_geodesic_array, threshold)
    elif method == 'cutoff':
        cutoff = threshold
    else:
        raise ValueError("method must be 'quantile' or 'cutoff'")

    #df_filtered = df[keep_mask].reset_index(drop=True)
    return mean_geodesic_array < cutoff, cutoff


def apply_filter(
    df,
    image_list,
    psi_list,
    theta_list,
    phi_list,
    pixel_size,
    df_ctf,
    df_info,
    avg_cutoff_lb=None,
    sd_cutoff_ub=None,
    pval_cutoff_lb=None,
    snr_cutoff_lb=None,
    snr_cutoff_ub=None,
    ctf_fitting_score_lb=None,
    ctf_fitting_score_ub=None,
    filter_by_image_thickness=True,
    thickness_lb=None,
    thickness_ub=None,
    filter_by_angular_invariance=False,
    geodesic_r=4,
    geodesic_threads=8,
    geodesic_method='quantile',  # or 'cutoff'
    geodesic_threshold=0.8       # quantile (0.8) or distance cutoff (e.g., 0.3)
):
    """
    Apply multi-criteria filtering to template matching results.
    """
    df_filtered = df.copy()
    df_record = df_filtered[['ORIGINAL_IMAGE_FILENAME','ORIGX','ORIGY','AVG','SD','PVALUE','ZSCORE','SNR']].copy()

    # Initialize to all True
    current_mask = pd.Series(True, index=df_filtered.index)
    geodesic_means = None
    
    # Define standard numeric filters (col, value, operator)
    value_filters = [
        ('AVG', avg_cutoff_lb, operator.gt),
        ('SD', sd_cutoff_ub, operator.lt),
        ('SNR', snr_cutoff_lb, operator.gt),
        ('SNR', snr_cutoff_ub, operator.lt),
        ('PVALUE', pval_cutoff_lb, operator.gt),
    ]

    # Apply scalar filters
    for name, cutoff, op in value_filters:
        if cutoff is not None:
            mask = op(df_filtered[name], cutoff)
            current_mask &= mask
            print(f"[INFO] Filter `{name} {op.__name__} {cutoff}`: {mask.sum()} particles retained")        


    if filter_by_image_thickness and thickness_lb is not None and thickness_ub is not None:
        thickness_mask = apply_image_thickness_filter(df_filtered, df_ctf, df_info, thickness_lb, thickness_ub)
        current_mask &= thickness_mask
        print(f"[INFO] Thickness filter [{thickness_lb}, {thickness_ub}]: {current_mask.sum()} particles retained")

    if ctf_fitting_score_lb is not None and ctf_fitting_score_ub is not None:
        ctf_mask = apply_ctf_score_filter(df_filtered, df_ctf, df_info, ctf_fitting_score_lb, ctf_fitting_score_ub)
        current_mask &= ctf_mask
        print(f"[INFO] CTF SCORE filter [{ctf_fitting_score_lb}, {ctf_fitting_score_ub}]: {current_mask.sum()} particles retained")

    # Apply angular invariance filtering
    if filter_by_angular_invariance:
        print(f"[INFO] Calculating angular variance...")
        geodesic_means = calculate_all_geodesic_means(
            df_filtered, image_list, psi_list, theta_list, phi_list,
            pixel_size, r=geodesic_r, threads=geodesic_threads
        )
        # Keep only thickness-filtered geodesic values
        df_record['mean_geodesic_distance'] = geodesic_means

        angular_mask, cutoff_val = apply_angular_invariance_filter(
            df_filtered, geodesic_means, method=geodesic_method, threshold=geodesic_threshold
        )
        current_mask &= angular_mask
        print(f"[INFO] Geodesic filter ({geodesic_method} ≤ {cutoff_val:.3f}): {current_mask.sum()} particles retained")

    # Final mask application
    df_filtered = df_filtered[current_mask].reset_index(drop=True)
    print(f"[INFO] Final filtered particles: {len(df_filtered)}")
    return df_filtered, df_record
