import pandas as pd
from tm_post.peak import Peak
from tm_post.image_data import TMImage
from pathlib import Path

def read_tm_package_starfile_header(): #-> list[str]:
    """
    Reads the default STAR file header from sample_data/header.star.

    Returns:
        A list of header lines (each ending in '\n').
    """
    header_path = Path(__file__).resolve().parents[2] / "sample_data" / "header.star"
    with open(header_path, "r") as f:
        lines = f.readlines()
    return lines

def add_star_dummy_column(df: pd.DataFrame) -> pd.DataFrame:
    """Insert a dummy column named '#' as the first column, filled with empty strings."""
    df_out = df.copy()
    df_out.insert(0, "#", "")
    return df_out

def extract_header_lines(file_path):
    """Extract real header lines (e.g., lines starting with '#' and containing PSI, etc.)."""
    header_keywords = ["PSI", "THETA", "PHI", "DF1", "SCORE"]
    header_lines = []
    data_lines = []

    with open(file_path, 'r') as f:
        for line in f:
            if line.lstrip().startswith('#') and all(key in line.upper() for key in header_keywords):
                header_lines.append(line)
            else:
                data_lines.append(line)

    return header_lines, data_lines

def write_starfile_with_headers(filepath, header_lines, df):
    """
    Write a STAR file with preformatted header lines and a data DataFrame that already includes the "#" column.

    Parameters:
    - filepath: path to the output file
    - header_lines: list of strings starting with "#", already including newline
    - df: DataFrame with the "#" column already included as the first column
    """    
    with open(filepath, 'w') as f:
        f.writelines(header_lines)
        f.write("\n")  # Add a blank line between header and data
        df.to_csv(f, sep="\t", index=False, header=False)

def find_data_header_lines(file_path):
    """Find line numbers of header lines that contain particle column names."""
    header_keywords = ["PSI", "THETA", "PHI", "DF1", "SCORE"]  # feel free to expand
    header_lines = []
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f, start=1):
            if line.lstrip().startswith('#') and all(key in line.upper() for key in header_keywords):
                header_lines.append(idx)
    return header_lines

def load_particle_starfile(file_path):
    """Load particle data from a starfile."""
    # Read in starfile and find header lines
    header_lines = find_data_header_lines(file_path)
    if not header_lines:
        raise ValueError("No header lines found in the starfile.")
    # Read the starfile, skipping the header lines
    df = pd.read_csv(file_path, delim_whitespace=True, skiprows=header_lines[0], header=None)
    # Set column names based on column indices
    if df.shape[1]==23: 
        df.columns = ["PSI", "THETA", "PHI", "DF1", "DF2", "ANGAST", 
                      "SCORE", "PSIZE", "VOLT", "Cs", "AmpC", 
                      "BTILTX", "BTILTY", "ISHFTX", "ISHFTY", 
                      "ORIGINAL_IMAGE_FILENAME", "ORIGX", "ORIGY", 
                      "PVALUE", "ZSCORE", "SNR", "AVG", "SD"
        ]
    elif df.shape[1]==18:
        df.columns = ["PSI", "THETA", "PHI", "DF1", "DF2", "ANGAST", 
                      "SCORE", "PSIZE", "VOLT", "Cs", "AmpC", 
                      "BTILTX", "BTILTY", "ISHFTX", "ISHFTY", 
                      "ORIGINAL_IMAGE_FILENAME", "ORIGX", "ORIGY"
        ]
    elif df.shape[1]==24: # from binary 
        df.columns = ["POS", "PSI", "THETA", "PHI", "SHX", "SHY", "DF1", 
                      "DF2", "ANGAST", "PSHIFT", "STAT", "OCC", 
                      "LogP", "SIGMA", "SCORE", "PSIZE", 
                      "VOLT", "Cs", "AmpC", "BTILTX", "BTILTY",
                        "ISHFTX", "ISHFTY", "SUBSET",
        ]
    elif df.shape[1]==29: # from simulator
        df.columns = ["POS", "PSI", "THETA", "PHI", "SHX", "SHY", 
                      "DF1", "DF2", "ANGAST", "PSHIFT", "OCC",
                      "LogP", "SIGMA", "SCORE", "CHANGE", "PSIZE",
                      "VOLT", "Cs", "AmpC", "BTILTX", "BTILTY",
                      "ISHFTX", "ISHFTY", "2DCLS", "TGRP", "PaGRP",
                      "SUBSET", "PREEXP", "TOTEXP"]
    return df



def convert_peaks_to_star_df(peaks,image_id,df_ctf,df_info,ctf_job_id,pixel_size,metric="pval",multiply_pixel_size= False
):
    """
    Convert a list of Peak objects to a STAR-format DataFrame.
    Parameters:
    - peaks: list of Peak objects
    - image_id: ID of the image
    - df_ctf: DataFrame containing CTF information
    - df_info: DataFrame containing image information
    - ctf_job_id: CTF job ID
    - pixel_size: pixel size in Angstroms
    - multiply_pixel_size: whether to multiply x and y coordinates by pixel size
    Returns:
    - df_out: DataFrame in STAR format
    """
    if not peaks:
        return pd.DataFrame()
    
    # Get image metadata
    row_ctf = df_ctf[(df_ctf.CTF_ESTIMATION_JOB_ID == ctf_job_id) & (df_ctf.IMAGE_ASSET_ID == image_id)]
    row_info = df_info[df_info.IMAGE_ASSET_ID == image_id]
    
    defocus1 = row_ctf['DEFOCUS1'].values[0]
    defocus2 = row_ctf['DEFOCUS2'].values[0]
    defocus_angle = row_ctf['DEFOCUS_ANGLE'].values[0]
    filename = row_info['FILENAME'].values[0]
    cs = row_info['SPHERICAL_ABERRATION'].values[0]
    voltage = row_info['VOLTAGE'].values[0]
    amp_contrast = row_ctf['AMPLITUDE_CONTRAST'].values[0]
    # Build STAR-format DataFrame
    df_out = pd.DataFrame({
        "#": ["" for _ in peaks],
        "PSI": [round(p.psi, 1) for p in peaks],
        "THETA": [round(p.theta, 1) for p in peaks],
        "PHI": [round(p.phi, 1) for p in peaks],
        "DF1": [round(p.delta_defocus + defocus1,1) for p in peaks],
        "DF2": [round(p.delta_defocus + defocus2,1) for p in peaks],
        "ANGAST": [round(defocus_angle,1) for _ in peaks],
        "SCORE": [
        round(
            p.pval if metric == "pval"
            else p.zscore if metric == "zscore"
            else p.snr if metric == "snr"
            else float("nan"), 2
        ) for p in peaks
    ],
        "PSIZE": [pixel_size for _ in peaks],
        "VOLT": [round(voltage,1) for _ in peaks],
        "Cs": [round(cs,1) for _ in peaks],
        "AmpC": [round(amp_contrast,3) for _ in peaks],  # adjust as needed
        "BTILTX": [0.0 for _ in peaks],
        "BTILTY": [0.0 for _ in peaks],
        "ISHFTX": [0.0 for _ in peaks],
        "ISHFTY": [0.0 for _ in peaks],
        "ORIGINAL_IMAGE_FILENAME": [f"'{filename}'" for _ in peaks],
        "ORIGX": [round(p.x * pixel_size if multiply_pixel_size else p.x,2) for p in peaks],
        "ORIGY": [round(p.y * pixel_size if multiply_pixel_size else p.y,2) for p in peaks],
        "PVALUE": [round(p.pval, 2) for p in peaks],
        "ZSCORE": [round(p.zscore, 2) for p in peaks],
        "SNR": [round(p.snr, 2) for p in peaks],
        "AVG": [round(p.avg, 2) for p in peaks],
        "SD": [round(p.sd, 2) for p in peaks]
    })
    return df_out
