"""3D refinements in the project database: cisTEM's Refinement /
Database::AddRefinement() family, for Refine 3D (server/refine3d.py).

A refinement is a REFINEMENT_LIST row plus, per class k, REFINEMENT_DETAILS_<id>
(one row per class: the settings, the average occupancy, the estimated
resolution and the reconstructed volume), REFINEMENT_RESULT_<id>_<k> (one
row per particle: Euler angles, shifts, occupancy, logP, sigma, score and the
imaging parameters -- the same 24 columns refinement_packages.py writes for a
package's "Random Parameters" start), REFINEMENT_RESOLUTION_STATISTICS_<id>_<k>
(the FSC / part FSC / SSNR curves) and REFINEMENT_ANGULAR_DISTRIBUTION_<id>_<k>
(an 18 x 72 theta/phi histogram of the assigned views). The refinement's id
is appended to REFINEMENT_PACKAGE_REFINEMENTS_LIST_<package>, the package's
LAST_REFINEMENT_ID is moved on, and each class's reconstruction is recorded
in RECONSTRUCTION_LIST (AddReconstructionJob) and made the package's
current reference (REFINEMENT_PACKAGE_CURRENT_REFERENCES_<package>).

Rows travel as the dicts starfile.py reads and writes (keys "psi", "theta",
..., "assigned_subset"); statistics as [{shell, resolution, fsc, part_fsc,
part_ssnr, rec_ssnr}] like abinitio.py's.

One deliberate simplification: cisTEM's angular distribution counts every
symmetry-related view of each particle (SymmetryMatrix); the histogram here
counts each assigned view once, which is the same picture for C1 and a
coarser one otherwise.
"""
import math
import os

import db
import symmetry as symmetry_module

RESULT_COLUMNS = ("POSITION_IN_STACK", "PSI", "THETA", "PHI", "XSHIFT", "YSHIFT", "DEFOCUS1", "DEFOCUS2", "DEFOCUS_ANGLE", "PHASE_SHIFT",
                  "OCCUPANCY", "LOGP", "SIGMA", "SCORE", "IMAGE_IS_ACTIVE", "PIXEL_SIZE", "MICROSCOPE_VOLTAGE", "MICROSCOPE_CS",
                  "AMPLITUDE_CONTRAST", "BEAM_TILT_X", "BEAM_TILT_Y", "IMAGE_SHIFT_X", "IMAGE_SHIFT_Y", "ASSIGNED_SUBSET")
RESULT_KEYS = ("position_in_stack", "psi", "theta", "phi", "x_shift", "y_shift", "defocus_1", "defocus_2", "defocus_angle", "phase_shift",
               "occupancy", "logp", "sigma", "score", "image_is_active", "pixel_size", "voltage", "cs", "amplitude_contrast",
               "beam_tilt_x", "beam_tilt_y", "image_shift_x", "image_shift_y", "assigned_subset")

DETAIL_COLUMNS = ("CLASS_NUMBER", "REFERENCE_VOLUME_ASSET_ID", "LOW_RESOLUTION_LIMIT", "HIGH_RESOLUTION_LIMIT", "MASK_RADIUS",
                  "SIGNED_CC_RESOLUTION_LIMIT", "GLOBAL_RESOLUTION_LIMIT", "GLOBAL_MASK_RADIUS", "NUMBER_RESULTS_TO_REFINE", "ANGULAR_SEARCH_STEP",
                  "SEARCH_RANGE_X", "SEARCH_RANGE_Y", "CLASSIFICATION_RESOLUTION_LIMIT", "SHOULD_FOCUS_CLASSIFY", "SPHERE_X_COORD", "SPHERE_Y_COORD",
                  "SPHERE_Z_COORD", "SPHERE_RADIUS", "SHOULD_REFINE_CTF", "DEFOCUS_SEARCH_RANGE", "DEFOCUS_SEARCH_STEP", "AVERAGE_OCCUPANCY",
                  "ESTIMATED_RESOLUTION", "RECONSTRUCTED_VOLUME_ASSET_ID", "RECONSTRUCTION_ID", "SHOULD_AUTOMASK", "SHOULD_REFINE_INPUT_PARAMS",
                  "SHOULD_USE_SUPPLIED_MASK", "MASK_ASSET_ID", "MASK_EDGE_WIDTH", "OUTSIDE_MASK_WEIGHT", "SHOULD_LOWPASS_OUTSIDE_MASK",
                  "MASK_FILTER_RESOLUTION")
DETAIL_DDL = ("CLASS_NUMBER INTEGER PRIMARY KEY, REFERENCE_VOLUME_ASSET_ID INTEGER, LOW_RESOLUTION_LIMIT REAL, HIGH_RESOLUTION_LIMIT REAL, MASK_RADIUS REAL, "
              "SIGNED_CC_RESOLUTION_LIMIT REAL, GLOBAL_RESOLUTION_LIMIT REAL, GLOBAL_MASK_RADIUS REAL, NUMBER_RESULTS_TO_REFINE INTEGER, ANGULAR_SEARCH_STEP REAL, "
              "SEARCH_RANGE_X REAL, SEARCH_RANGE_Y REAL, CLASSIFICATION_RESOLUTION_LIMIT REAL, SHOULD_FOCUS_CLASSIFY INTEGER, SPHERE_X_COORD REAL, SPHERE_Y_COORD REAL, "
              "SPHERE_Z_COORD REAL, SPHERE_RADIUS REAL, SHOULD_REFINE_CTF INTEGER, DEFOCUS_SEARCH_RANGE REAL, DEFOCUS_SEARCH_STEP REAL, AVERAGE_OCCUPANCY REAL, "
              "ESTIMATED_RESOLUTION REAL, RECONSTRUCTED_VOLUME_ASSET_ID INTEGER, RECONSTRUCTION_ID INTEGER, SHOULD_AUTOMASK INTEGER, SHOULD_REFINE_INPUT_PARAMS INTEGER, "
              "SHOULD_USE_SUPPLIED_MASK INTEGER, MASK_ASSET_ID INTEGER, MASK_EDGE_WIDTH REAL, OUTSIDE_MASK_WEIGHT REAL, SHOULD_LOWPASS_OUTSIDE_MASK INTEGER, "
              "MASK_FILTER_RESOLUTION REAL")
RESULT_DDL = ("POSITION_IN_STACK INTEGER PRIMARY KEY, PSI REAL, THETA REAL, PHI REAL, XSHIFT REAL, YSHIFT REAL, DEFOCUS1 REAL, DEFOCUS2 REAL, DEFOCUS_ANGLE REAL, "
              "PHASE_SHIFT REAL, OCCUPANCY REAL, LOGP REAL, SIGMA REAL, SCORE REAL, IMAGE_IS_ACTIVE INTEGER, PIXEL_SIZE REAL, MICROSCOPE_VOLTAGE REAL, "
              "MICROSCOPE_CS REAL, AMPLITUDE_CONTRAST REAL, BEAM_TILT_X REAL, BEAM_TILT_Y REAL, IMAGE_SHIFT_X REAL, IMAGE_SHIFT_Y REAL, ASSIGNED_SUBSET INTEGER")

THETA_BINS, PHI_BINS = 18, 72


def _table_exists(conn, name):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def next_refinement_id(conn):
    return conn.execute("SELECT COALESCE(MAX(REFINEMENT_ID), 0) + 1 FROM REFINEMENT_LIST").fetchone()[0]


def next_reconstruction_id(conn):
    return conn.execute("SELECT COALESCE(MAX(RECONSTRUCTION_ID), 0) + 1 FROM RECONSTRUCTION_LIST").fetchone()[0]


def refinement_row(conn, refinement_id):
    return conn.execute("SELECT * FROM REFINEMENT_LIST WHERE REFINEMENT_ID=?", (int(refinement_id),)).fetchone()


def load_rows(conn, refinement_id, class_number):
    """REFINEMENT_RESULT_<id>_<k> as star rows."""
    table = "REFINEMENT_RESULT_{}_{}".format(int(refinement_id), int(class_number))
    if not _table_exists(conn, table):
        raise ValueError("refinement {} has no results for class {}".format(refinement_id, class_number))
    rows = []
    for r in conn.execute("SELECT * FROM {} ORDER BY POSITION_IN_STACK".format(table)).fetchall():
        rows.append({k: (r[c] if r[c] is not None else 0) for k, c in zip(RESULT_KEYS, RESULT_COLUMNS)})
    return rows


def load_statistics(conn, refinement_id, class_number):
    table = "REFINEMENT_RESOLUTION_STATISTICS_{}_{}".format(int(refinement_id), int(class_number))
    if not _table_exists(conn, table):
        return []
    return [{"shell": r["SHELL"], "resolution": r["RESOLUTION"], "fsc": r["FSC"], "part_fsc": r["PART_FSC"], "part_ssnr": r["PART_SSNR"], "rec_ssnr": r["REC_SSNR"]}
            for r in conn.execute("SELECT * FROM {} ORDER BY SHELL".format(table)).fetchall()]


def load_angular_distribution(conn, refinement_id, class_number):
    """One class's 18 x 72 angular histogram (REFINEMENT_ANGULAR_DISTRIBUTION_<id>_<k>), bin order; [] when none."""
    table = "REFINEMENT_ANGULAR_DISTRIBUTION_{}_{}".format(int(refinement_id), int(class_number))
    if not _table_exists(conn, table):
        return []
    return [r[0] for r in conn.execute("SELECT NUMBER_IN_BIN FROM {} ORDER BY BIN_NUMBER".format(table)).fetchall()]


def load_details(conn, refinement_id):
    table = "REFINEMENT_DETAILS_{}".format(int(refinement_id))
    if not _table_exists(conn, table):
        return []
    return [dict(r) for r in conn.execute("SELECT * FROM {} ORDER BY CLASS_NUMBER".format(table)).fetchall()]


def estimated_resolution(stats, pixel_size, use_part_fsc=True):
    """ResolutionStatistics::ReturnEstimatedResolution(): the resolution
    where the FSC first drops below 0.143, midway between shells, never
    better than Nyquist. Reported from the **particle** FSC here (cisTEM's
    GUI reports the plain FSC; the part FSC, corrected for the mask and
    the particle count, is the one this app records and shows, at the
    author's request)."""
    key = "part_fsc" if use_part_fsc else "fsc"
    est = 0.0
    for i in range(1, len(stats)):
        if stats[i][key] < 0.143:
            est = (stats[i - 1]["resolution"] + stats[i]["resolution"]) / 2.0
            break
    return max(est, 2.0 * float(pixel_size))


def _theta_phi_bin(theta, phi):
    """AngularDistributionHistogram::ReturnThetaBin / ReturnPhiBin."""
    phi %= 360.0
    tb = next((i for i, b in enumerate(_THETA_BOUNDS) if theta < b), len(_THETA_BOUNDS))
    pb = next((i for i, b in enumerate(_PHI_BOUNDS) if phi < b), len(_PHI_BOUNDS))
    return THETA_BINS * pb + tb


_THETA_BOUNDS = [math.degrees(math.acos(t / 90.0)) for t in (90.0 - 90.0 / THETA_BINS * i for i in range(1, THETA_BINS)) if t > 0]
_PHI_BOUNDS = [360.0 / PHI_BINS * i for i in range(1, PHI_BINS)]


def best_class_per_particle(class_rows):
    """Refinement::ReturnClassWithHighestOccupanyForGivenParticle() for
    every particle: the 1-based class with the highest occupancy, matched
    across classes by position in the stack."""
    best = {}
    for k, rows in enumerate(class_rows, start=1):
        for r in rows:
            pos = int(r.get("position_in_stack", 0))
            occ = float(r.get("occupancy", 0.0))
            if pos not in best or occ > best[pos][0]:
                best[pos] = (occ, k)
    return {pos: k for pos, (occ, k) in best.items()}


def angular_histogram(class_rows, wanted_class=1, symmetry="C1"):
    """Refinement::FillAngularDistributionHistogram(): 18 theta bins
    (equal-area in cos theta) x 72 phi bins over the northern hemisphere,
    counting every symmetry-related view of each active particle whose
    highest-occupancy class is `wanted_class` (1-based). `class_rows` is
    the per-class list of particle rows; a flat list of rows is taken as a
    single class."""
    if class_rows and isinstance(class_rows[0], dict):
        class_rows = [class_rows]
    hist = [0] * (THETA_BINS * PHI_BINS)
    if not class_rows or wanted_class < 1 or wanted_class > len(class_rows):
        return hist
    try:
        mats = symmetry_module.matrices(symmetry)
    except ValueError:
        mats = symmetry_module.matrices("C1")
    best = best_class_per_particle(class_rows) if len(class_rows) > 1 else None
    for r in class_rows[wanted_class - 1]:
        if r.get("image_is_active", 1) < 0:
            continue
        if best is not None and best.get(int(r.get("position_in_stack", 0))) != wanted_class:
            continue
        em = symmetry_module.euler_matrix(float(r.get("phi", 0.0)), float(r.get("theta", 0.0)), float(r.get("psi", 0.0)))
        for rm in mats:
            x, y, z = symmetry_module.rotate(symmetry_module.matmul(rm, em), (0.0, 0.0, 1.0))
            if z < 0.0:
                x, y = -x, -y
            hist[_theta_phi_bin(symmetry_module.projection_theta_deg(x, y), symmetry_module.projection_phi_deg(x, y))] += 1
    return hist


def write_angular_distribution(conn, rid, k, hist):
    conn.execute("DROP TABLE IF EXISTS REFINEMENT_ANGULAR_DISTRIBUTION_{}_{}".format(rid, k))
    conn.execute("CREATE TABLE REFINEMENT_ANGULAR_DISTRIBUTION_{}_{}(BIN_NUMBER INTEGER PRIMARY KEY, NUMBER_IN_BIN INTEGER)".format(rid, k))
    conn.executemany("INSERT INTO REFINEMENT_ANGULAR_DISTRIBUTION_{}_{} VALUES (?, ?)".format(rid, k), list(enumerate(hist)))


def rebuild_angular_distributions(conn, rid):
    """Recompute a stored refinement's angular distributions from its result
    tables with its package's symmetry (for refinements written before the
    symmetry expansion, or after a package's symmetry changes)."""
    row = conn.execute(_LIST_SELECT + "WHERE r.REFINEMENT_ID=?", (int(rid),)).fetchone()
    if row is None:
        raise KeyError("no refinement {}".format(rid))
    classes = int(row["NUMBER_OF_CLASSES"] or 1)
    class_rows = [load_rows(conn, rid, k) for k in range(1, classes + 1)]
    with conn:
        for k in range(1, classes + 1):
            write_angular_distribution(conn, rid, k, angular_histogram(class_rows, k, row["SYMMETRY"] or "C1"))


def add_refinement(conn, ref, class_rows, class_stats, class_details, angular=True, symmetry="C1"):
    """Database::AddRefinement() + the package bookkeeping around it.
    `ref` carries refinement_id, refinement_package_asset_id, name,
    starting_refinement_id, number_of_particles, number_of_classes,
    resolution_statistics_box_size / _pixel_size, percent_used, job_id;
    class_details[k] the REFINEMENT_DETAILS values for class k+1 (a dict
    keyed by column name; missing ones default); `symmetry` the package's
    point group, which the angular distributions are expanded by."""
    rid = int(ref["refinement_id"])
    package_id = int(ref["refinement_package_asset_id"])
    with conn:
        conn.execute("INSERT OR REPLACE INTO REFINEMENT_LIST(REFINEMENT_ID, REFINEMENT_PACKAGE_ASSET_ID, NAME, RESOLUTION_STATISTICS_ARE_GENERATED, "
                     "DATETIME_OF_RUN, STARTING_REFINEMENT_ID, NUMBER_OF_PARTICLES, NUMBER_OF_CLASSES, RESOLUTION_STATISTICS_BOX_SIZE, "
                     "RESOLUTION_STATISTICS_PIXEL_SIZE, PERCENT_USED, JOB_ID) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                     (rid, package_id, ref["name"], 1 if ref.get("resolution_statistics_are_generated") else 0, int(ref.get("datetime_of_run") or db.now_epoch()),
                      int(ref.get("starting_refinement_id") if ref.get("starting_refinement_id") is not None else -1), int(ref["number_of_particles"]),
                      int(ref["number_of_classes"]), int(ref["resolution_statistics_box_size"]), float(ref["resolution_statistics_pixel_size"]),
                      float(ref.get("percent_used", 100.0)), ref.get("job_id")))
        conn.execute("DROP TABLE IF EXISTS REFINEMENT_DETAILS_{}".format(rid))
        conn.execute("CREATE TABLE REFINEMENT_DETAILS_{}({})".format(rid, DETAIL_DDL))
        for k in range(1, int(ref["number_of_classes"]) + 1):
            d = dict(class_details[k - 1]) if k - 1 < len(class_details) else {}
            d["CLASS_NUMBER"] = k
            values = [d.get(c, _DETAIL_DEFAULTS.get(c, 0)) for c in DETAIL_COLUMNS]
            conn.execute("INSERT INTO REFINEMENT_DETAILS_{} VALUES ({})".format(rid, ",".join("?" * len(DETAIL_COLUMNS))), values)
            conn.execute("DROP TABLE IF EXISTS REFINEMENT_RESULT_{}_{}".format(rid, k))
            conn.execute("CREATE TABLE REFINEMENT_RESULT_{}_{}({})".format(rid, k, RESULT_DDL))
            conn.executemany("INSERT OR REPLACE INTO REFINEMENT_RESULT_{}_{} VALUES ({})".format(rid, k, ",".join("?" * len(RESULT_KEYS))),
                             [tuple(r.get(key, 0) for key in RESULT_KEYS) for r in class_rows[k - 1]])
            write_statistics(conn, rid, k, class_stats[k - 1] if k - 1 < len(class_stats) else [])
            if angular:
                write_angular_distribution(conn, rid, k, angular_histogram(class_rows, k, symmetry))
        list_table = "REFINEMENT_PACKAGE_REFINEMENTS_LIST_{}".format(package_id)
        conn.execute("CREATE TABLE IF NOT EXISTS {}(REFINEMENT_NUMBER INTEGER PRIMARY KEY, REFINEMENT_ID INTEGER)".format(list_table))
        if conn.execute("SELECT 1 FROM {} WHERE REFINEMENT_ID=?".format(list_table), (rid,)).fetchone() is None:
            n = conn.execute("SELECT COALESCE(MAX(REFINEMENT_NUMBER), 0) + 1 FROM {}".format(list_table)).fetchone()[0]
            conn.execute("INSERT INTO {} VALUES (?, ?)".format(list_table), (n, rid))
        conn.execute("UPDATE REFINEMENT_PACKAGE_ASSETS SET LAST_REFINEMENT_ID=?, NUMBER_OF_REFINEMENTS=COALESCE(NUMBER_OF_REFINEMENTS, 0) + 1 "
                     "WHERE REFINEMENT_PACKAGE_ASSET_ID=?", (rid, package_id))


_DETAIL_DEFAULTS = {"REFERENCE_VOLUME_ASSET_ID": -1, "RECONSTRUCTED_VOLUME_ASSET_ID": -1, "RECONSTRUCTION_ID": -1, "MASK_ASSET_ID": -1,
                    "AVERAGE_OCCUPANCY": 100.0, "MASK_EDGE_WIDTH": 10.0, "MASK_FILTER_RESOLUTION": 20.0, "SHOULD_REFINE_INPUT_PARAMS": 1}


def write_statistics(conn, refinement_id, class_number, stats):
    """UpdateRefinementResolutionStatistics() for one class: replace the curve."""
    table = "REFINEMENT_RESOLUTION_STATISTICS_{}_{}".format(int(refinement_id), int(class_number))
    conn.execute("DROP TABLE IF EXISTS {}".format(table))
    conn.execute("CREATE TABLE {}(SHELL INTEGER PRIMARY KEY, RESOLUTION REAL, FSC REAL, PART_FSC REAL, PART_SSNR REAL, REC_SSNR REAL)".format(table))
    conn.executemany("INSERT OR REPLACE INTO {} VALUES (?,?,?,?,?,?)".format(table),
                     [(s["shell"], s["resolution"], s["fsc"], s["part_fsc"], s["part_ssnr"], s["rec_ssnr"]) for s in stats])


def update_details(conn, refinement_id, class_number, **fields):
    if not fields:
        return
    cols = ", ".join("{}=?".format(k) for k in fields)
    with conn:
        conn.execute("UPDATE REFINEMENT_DETAILS_{} SET {} WHERE CLASS_NUMBER=?".format(int(refinement_id), cols), list(fields.values()) + [int(class_number)])


def add_reconstruction_job(conn, reconstruction_id, package_id, refinement_id, name, inner_mask, outer_mask, resolution_limit,
                           score_weight_conversion, adjust_scores, crop_images, save_half_maps, likelihood_blur, smoothing_factor,
                           class_number, volume_asset_id):
    with conn:
        conn.execute("INSERT OR REPLACE INTO RECONSTRUCTION_LIST VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (int(reconstruction_id), int(package_id), int(refinement_id), name, float(inner_mask), float(outer_mask), float(resolution_limit),
                      float(score_weight_conversion), 1 if adjust_scores else 0, 1 if crop_images else 0, 1 if save_half_maps else 0,
                      1 if likelihood_blur else 0, float(smoothing_factor), int(class_number), int(volume_asset_id)))


def set_current_reference(conn, package_id, class_number, volume_asset_id):
    table = "REFINEMENT_PACKAGE_CURRENT_REFERENCES_{}".format(int(package_id))
    with conn:
        conn.execute("CREATE TABLE IF NOT EXISTS {}(CLASS_NUMBER INTEGER PRIMARY KEY, VOLUME_ASSET_ID INTEGER)".format(table))
        conn.execute("INSERT OR REPLACE INTO {} VALUES (?, ?)".format(table), (int(class_number), int(volume_asset_id)))


def current_references(conn, package_id):
    table = "REFINEMENT_PACKAGE_CURRENT_REFERENCES_{}".format(int(package_id))
    if not _table_exists(conn, table):
        return {}
    return {r["CLASS_NUMBER"]: r["VOLUME_ASSET_ID"] for r in conn.execute("SELECT * FROM {}".format(table)).fetchall()}


_LIST_SELECT = ("SELECT r.*, rp.NAME AS PACKAGE_NAME, rp.SYMMETRY, j.JOB_NUMBER FROM REFINEMENT_LIST r "
                "LEFT JOIN REFINEMENT_PACKAGE_ASSETS rp ON rp.REFINEMENT_PACKAGE_ASSET_ID = r.REFINEMENT_PACKAGE_ASSET_ID "
                "LEFT JOIN JOBS j ON j.JOB_ID = r.JOB_ID ")


def _refinement_json(conn, row, with_classes=False):
    d = {k.lower(): row[k] for k in row.keys()}
    d["resolution_statistics_are_generated"] = bool(row["RESOLUTION_STATISTICS_ARE_GENERATED"])
    details = load_details(conn, row["REFINEMENT_ID"])
    d["classes"] = []
    for k in range(1, int(row["NUMBER_OF_CLASSES"] or 1) + 1):
        det = next((x for x in details if x["CLASS_NUMBER"] == k), {})
        c = {kk.lower(): v for kk, v in det.items()}
        vid = det.get("RECONSTRUCTED_VOLUME_ASSET_ID", -1)
        vol = conn.execute("SELECT NAME, FILENAME FROM VOLUME_ASSETS WHERE VOLUME_ASSET_ID=?", (vid,)).fetchone() if vid is not None and vid >= 0 else None
        c["class_number"] = k
        # The reported estimate follows the stored statistics, so refinements
        # written before the estimate moved to the part FSC read the same way.
        if vid is not None and vid >= 0:
            stats_k = load_statistics(conn, row["REFINEMENT_ID"], k)
            if stats_k:
                c["estimated_resolution"] = estimated_resolution(stats_k, row["RESOLUTION_STATISTICS_PIXEL_SIZE"] or 1.0)
        c["volume_name"] = vol["NAME"] if vol else None
        c["volume_filename"] = vol["FILENAME"] if vol else None
        c["volume_file_exists"] = bool(vol and vol["FILENAME"] and os.path.isfile(vol["FILENAME"]))
        if with_classes:
            c["statistics"] = load_statistics(conn, row["REFINEMENT_ID"], k)
            table = "REFINEMENT_ANGULAR_DISTRIBUTION_{}_{}".format(row["REFINEMENT_ID"], k)
            c["angular_distribution"] = [r[0] for r in conn.execute("SELECT NUMBER_IN_BIN FROM {} ORDER BY BIN_NUMBER".format(table)).fetchall()] if _table_exists(conn, table) else None
        d["classes"].append(c)
    return d


def list_refinements(conn, package_id=None):
    sql, args = _LIST_SELECT, ()
    if package_id is not None:
        sql += "WHERE r.REFINEMENT_PACKAGE_ASSET_ID = ? "
        args = (int(package_id),)
    return [_refinement_json(conn, r) for r in conn.execute(sql + "ORDER BY r.REFINEMENT_ID", args).fetchall()]


def get_refinement(conn, refinement_id):
    row = conn.execute(_LIST_SELECT + "WHERE r.REFINEMENT_ID = ?", (int(refinement_id),)).fetchone()
    return _refinement_json(conn, row, with_classes=True) if row is not None else None
