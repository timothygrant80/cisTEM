
from tm_post.peak import Peak
from tm_post.image_data import TMImage
from skimage.feature import peak_local_max
import pandas as pd
from tm_post.statistics import calculate_2dtm_pval
from tm_post.mrcfile import read_mrc_file
from tm_post.starfile import convert_peaks_to_star_df
import concurrent.futures
from tqdm import tqdm

def return_peaks_for_image(image: TMImage, metric_cutoff, local_max_filter="zscore", metric="pval", min_radius=10, exclude_borders=35, q=3):# -> list[Peak]:
    """Generate peak information for a given image in a database."""
    # Read all maps
    snr_image = read_mrc_file(image.snr_file)
    zscore_image = read_mrc_file(image.zscore_file)
    psi_image = read_mrc_file(image.psi_file)
    theta_image = read_mrc_file(image.theta_file)
    phi_image = read_mrc_file(image.phi_file)
    defocus_image = read_mrc_file(image.defocus_file)
    avg_image = read_mrc_file(image.avg_file)
    sd_image = read_mrc_file(image.sd_file)
    
    if local_max_filter == "zscore":
        peaks_coordinates = peak_local_max(zscore_image, min_distance=min_radius, exclude_border=exclude_borders, threshold_abs=0.0)
    elif local_max_filter == "snr":
        peaks_coordinates = peak_local_max(snr_image, min_distance=min_radius, exclude_border=exclude_borders, threshold_abs=0.0)
        
    # Collect raw values from detected peaks
    peak_data = []
    for (y,x) in peaks_coordinates:
        peak_data.append({
            "x_pixel": x,
            "y_pixel": y,
            "snr": snr_image[y, x],
            "zscore": zscore_image[y, x],
            "psi": psi_image[y, x],
            "theta": theta_image[y, x],
            "phi": phi_image[y, x],
            "delta_defocus": defocus_image[y, x],
            "avg": avg_image[y, x],
            "sd": sd_image[y, x],
        })
        
    # Compute p-values
    df_peaks = pd.DataFrame(peak_data)
    df_peaks["pval"] = calculate_2dtm_pval(df_peaks["zscore"].values, df_peaks["snr"].values, q=q)
    
    # Filter and create Peak objects
    filtered_peaks = []
    for _, row in df_peaks.iterrows():
        #if row["avg"] > avg_cutoff and row["snr"] < snr_cutoff:
        if (metric == "pval" and row["pval"] >= metric_cutoff) or \
            (metric == "zscore" and row["zscore"] >= metric_cutoff) or \
            (metric == "snr" and row["snr"] >= metric_cutoff):
            filtered_peaks.append(
                Peak(
                    image_id=image.image_id,
                    image_name=image.filename,
                    x=row["x_pixel"],
                    y=row["y_pixel"],
                    delta_defocus=row["delta_defocus"],
                    psi=row["psi"],
                    theta=row["theta"],
                    phi=row["phi"],
                    snr=row["snr"],
                    zscore=row["zscore"],
                    pval=row["pval"],
                    avg=row["avg"],
                    sd=row["sd"],
                )
            )

    # Sort peaks by selected metric in descending order
    filtered_peaks = sorted(filtered_peaks, key=lambda p: getattr(p, metric), reverse=True)

    print(f"[INFO] Extracted {len(filtered_peaks)} peaks from image: {image.filename}")
    return filtered_peaks

def extract_particles_from_2dtm_search(
    tm_images, 
    local_max_filter,
    df_ctf,
    df_info,
    ctf_job_id,
    metric = "pval",
    metric_cutoff = 8.0,
    #avg_cutoff = 0.0,
    #snr_cutoff = 9.0,
    pixel_size = 1.0,
    min_radius = 10,
    exclude_borders = 35,
    max_threads = 4,
    q = 3,
    ):
    """
    Extract particles (peaks) from a list of TMImage objects in parallel.
    
    Returns a full STAR-format DataFrame of all particles across all images.
    """
    all_particles = []
    def process_image(image: TMImage):
        # extract Peak objects for this image
        peaks = return_peaks_for_image(
            image=image,
            #avg_cutoff=avg_cutoff,
            #snr_cutoff=snr_cutoff,
            local_max_filter=local_max_filter,
            metric_cutoff=metric_cutoff,
            metric=metric,
            min_radius=min_radius,
            exclude_borders=exclude_borders,
            q=q
        )
        print(f"[INFO] Found {len(peaks)} peaks in image {image.filename}")

        # convert to STAR-format DataFrame
        return convert_peaks_to_star_df(
            peaks=peaks,
            image_id=image.image_id,
            df_ctf=df_ctf,
            df_info=df_info,
            ctf_job_id=ctf_job_id,
            pixel_size=pixel_size,
            multiply_pixel_size=True,
            metric=metric
        )

    all_dataframes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(process_image, img) for img in tm_images]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Extracting particles"):
            df = future.result()
            all_dataframes.append(df)
    
    return pd.concat(all_dataframes, ignore_index=True)