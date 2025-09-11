import numpy as np
import mrcfile
from tqdm import tqdm
import concurrent.futures
from tm_post.geometry import geodesic_distance, euler_to_rotation

def get_local_patch(file_name, x, y, r):
    """Extract a square patch around (x, y) from an MRC file."""
    with mrcfile.open(file_name) as mrc:
        data = mrc.data[0]  # Assuming it's 3D with first slice of interest
    return data[y - r:y + r, x - r:x + r]

def rotation_matrix_patch(psi_patch, theta_patch, phi_patch):
    """Convert Euler angle patches to rotation matrix patches."""
    r = psi_patch.shape[0] // 2
    rot_patch = np.empty((2*r, 2*r), dtype=object)  # Store Rotation objects
    for i in range(2*r):
        for j in range(2*r):
            rot_patch[i, j] = euler_to_rotation(
                psi_patch[i, j], theta_patch[i, j], phi_patch[i, j]
            )
    return rot_patch

def compute_geodesic_distances(rot_patch, center_coord):
    """Compute geodesic distances from center rotation to all others."""
    r = rot_patch.shape[0] // 2
    ref_rot = rot_patch[r, r]
    distances = []
    for i in range(2*r):
        for j in range(2*r):
            if i != r or j != r:  # exclude center
                dist = geodesic_distance(ref_rot, rot_patch[i, j])
                distances.append(dist)
    return distances

def calculate_particle_geodesic_distance(df, image_list, psi_list, theta_list, phi_list, peak_number, pixel_size, r=10):
    """Top-level function to calculate geodesic distances near a particle."""
    # Get particle info
    row = df.iloc[peak_number-1]
    x, y = row['ORIGX'] / pixel_size, row['ORIGY'] / pixel_size
    x, y = int(round(x,1)), int(round(y,1))
    image_name = row['ORIGINAL_IMAGE_FILENAME'].strip("'")

    image_idx = image_list.index(image_name)
    psi_file, theta_file, phi_file = psi_list[image_idx], theta_list[image_idx], phi_list[image_idx]

    # Load patches
    psi_patch = get_local_patch(psi_file, x, y, r)
    theta_patch = get_local_patch(theta_file, x, y, r)
    phi_patch = get_local_patch(phi_file, x, y, r)
   
    # Convert to rotation matrices and compute distances
    rot_patch = rotation_matrix_patch(psi_patch, theta_patch, phi_patch)
    distances = compute_geodesic_distances(rot_patch, (r, r))

    return distances


def calculate_mean_geodesic_for_row(args):
    """Wrapper to calculate mean geodesic distance for one row."""
    df, image_list, psi_list, theta_list, phi_list, pixel_size, r, row_idx = args
    try:
        distances = calculate_particle_geodesic_distance(
            df, image_list, psi_list, theta_list, phi_list,
            peak_number=row_idx + 1,  # your function is 1-based
            pixel_size=pixel_size,
            r=r
        )
        return np.mean(distances)
    except Exception as e:
        print(f"[Warning] Skipped row {row_idx} due to error: {e}")
        return np.nan

def calculate_all_geodesic_means(df, image_list, psi_list, theta_list, phi_list, pixel_size, r=10, threads=4):
    """Compute mean geodesic distance for each row in df in parallel, with progress bar."""
    args_list = [
        (df, image_list, psi_list, theta_list, phi_list, pixel_size, r, i)
        for i in range(len(df))
    ]
    means = [None] * len(args_list)

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {executor.submit(calculate_mean_geodesic_for_row, args): idx for idx, args in enumerate(args_list)}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Calculating geodesic means"):
            idx = futures[future]
            try:
                means[idx] = future.result()
            except Exception as e:
                print(f"[Warning] Error in row {idx}: {e}")
                means[idx] = np.nan
        
    return np.array(means)