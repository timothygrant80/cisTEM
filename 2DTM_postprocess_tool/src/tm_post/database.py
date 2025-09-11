import sqlite3
import pandas as pd
from tm_post.image_data import TMImage

def get_info_from_cistem_database(db_file, tm_job_id, ctf_job_id, requested_output_names=None):
    # Table names: CTF, image info, TM job
    table_ctf = 'ESTIMATED_CTF_PARAMETERS'
    table_info = "IMAGE_ASSETS"
    table_tm = 'TEMPLATE_MATCH_LIST'

    # Connect to the database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Load all three tables into dataframes
    def load_table(query):
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame(rows, columns=columns)

    df_ctf = load_table(f"SELECT * FROM {table_ctf}")
    df_info = load_table(f"SELECT * FROM {table_info}")
    df_tm = load_table(f"SELECT * FROM {table_tm}")

    # Filter CTF data
    df_ctf = df_ctf[df_ctf['CTF_ESTIMATION_JOB_ID'] == ctf_job_id]
    
    # Filter TM data
    df_tm = df_tm[df_tm['TEMPLATE_MATCH_JOB_ID'] == tm_job_id]

    # Close the connection
    conn.close()

    # Get relevant image IDs
    image_ids = df_tm.IMAGE_ASSET_ID.values

    # Always needed
    image_list = [df_info[df_info.IMAGE_ASSET_ID == image_id].FILENAME.values[0] for image_id in image_ids]

    # Define mapping of column name to list
    all_output_columns = {
        "MIP_OUTPUT_FILE": [],
        "SCALED_MIP_OUTPUT_FILE": [],
        "PSI_OUTPUT_FILE": [],
        "THETA_OUTPUT_FILE": [],
        "PHI_OUTPUT_FILE": [],
        "DEFOCUS_OUTPUT_FILE": [],
        "AVG_OUTPUT_FILE": [],
        "STD_OUTPUT_FILE": []
    }

    # Default is return all output columns
    if requested_output_names is None:
        requested_output_names = all_output_columns
    elif isinstance(requested_output_names, str):
        requested_output_names = [requested_output_names]

    # Create output lists
    output_lists = {col: [] for col in requested_output_names}

    for image_id in image_ids:
        row_tm = df_tm[df_tm.IMAGE_ASSET_ID == image_id]

        # Output files
        for col in requested_output_names:
            if col not in output_lists:
                raise ValueError(f"Column '{col}' is not recognized.")
            if col not in row_tm.columns:
                raise ValueError(f"Column '{col}' not found in TM database table.")
            output_lists[col].append(row_tm[col].values[0])
            
    # Assemble result
    result = {
        "image_list": image_list,
        "image_ids": image_ids,
        "df_ctf": df_ctf,
        "df_info" : df_info
    }

    for col in requested_output_names:
        result[col] = output_lists[col]
    
    return result

def load_tm_images_from_db(db_file, tm_job_id, ctf_job_id):# -> list[TMImage]:
    # Table names: CTF, image info, TM job
    table_ctf = 'ESTIMATED_CTF_PARAMETERS'
    table_info = "IMAGE_ASSETS"
    table_tm = 'TEMPLATE_MATCH_LIST'

    # Connect to the database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Load all three tables into dataframes
    def load_table(query):
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame(rows, columns=columns)

    df_ctf = load_table(f"SELECT * FROM {table_ctf}")
    df_info = load_table(f"SELECT * FROM {table_info}")
    df_tm = load_table(f"SELECT * FROM {table_tm}")

    # Filter CTF data
    df_ctf = df_ctf[df_ctf['CTF_ESTIMATION_JOB_ID'] == ctf_job_id]
    
    # Filter TM data
    df_tm = df_tm[df_tm['TEMPLATE_MATCH_JOB_ID'] == tm_job_id]

    # Close the connection
    conn.close()

    # Get relevant image IDs
    image_ids = df_tm.IMAGE_ASSET_ID.values

    # Instead of returning lists, build TMImage objects
    images = []
    
    for image_id in image_ids:
        filename = df_info[df_info.IMAGE_ASSET_ID == image_id].FILENAME.values[0]
        row_tm = df_tm[df_tm.IMAGE_ASSET_ID == image_id]
        row_ctf = df_ctf[df_ctf.IMAGE_ASSET_ID == image_id]

        image = TMImage(
            image_id=image_id,
            filename=filename,
            pixel_size=row_tm["USED_PIXEL_SIZE"].values[0],
            psi_file=row_tm["PSI_OUTPUT_FILE"].values[0],
            theta_file=row_tm["THETA_OUTPUT_FILE"].values[0],
            phi_file=row_tm["PHI_OUTPUT_FILE"].values[0],
            snr_file=row_tm["MIP_OUTPUT_FILE"].values[0],
            zscore_file=row_tm["SCALED_MIP_OUTPUT_FILE"].values[0],
            defocus_file=row_tm["DEFOCUS_OUTPUT_FILE"].values[0],
            avg_file=row_tm["AVG_OUTPUT_FILE"].values[0],
            sd_file=row_tm["STD_OUTPUT_FILE"].values[0],
            defocus1=row_ctf["DEFOCUS1"].values[0],
            defocus2=row_ctf["DEFOCUS2"].values[0],
            defocus_angle=row_ctf["DEFOCUS_ANGLE"].values[0],
            amp_contrast=row_ctf["AMPLITUDE_CONTRAST"].values[0],
            voltage=row_ctf["VOLTAGE"].values[0],
            cs=row_ctf["SPHERICAL_ABERRATION"].values[0],
        )
        images.append(image)

    return images, df_ctf, df_info