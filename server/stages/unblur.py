"""Align Movies -> cisTEM's `unblur`, argument for argument.

The argument list is MyAlignMoviesPanel::StartAlignmentClick()'s AddJob call
(format string "ssfffbbfifbiifffbsbsfbfffbtbtiiiibttii", 38 arguments) in
the order UnBlurApp::DoCalculation() reads them back out of
my_current_job.arguments[0..37]. The order is the contract; change it here
only if unblur.cpp changes.

Output naming follows cisTEM: <project>/Assets/Images/<movie>_<asset id>_<n>.mrc
where n is how many alignments that movie already has, with the amplitude
spectrum under Images/Spectra/ and the optional scaled sum under
Images/Scaled/. Re-aligning a movie therefore never overwrites an earlier
result, and -- as in cisTEM's WriteResultToDataBase() -- *updates* the
movie's existing image asset rather than adding a second one.
"""

import json
import os
from pathlib import Path

import db
import job_protocol as jp
from imageheaders import HeaderError, read_image_header

PROGRAM = {"name": "unblur", "executable": "unblur"}

# Placeholder cisTEM passes for a file it doesn't want written or read.
DEV_NULL = "/dev/null"


def _num(params, key, default, cast):
    """Form values arrive as strings or numbers; missing or blank means the
    cisTEM default the front end also prefilled."""
    value = params.get(key)
    if value is None or value == "":
        return default
    return cast(value)


def _flag(params, key, default):
    value = params.get(key)
    if value is None or value == "":
        return default
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "on")
    return bool(value)


def _ensure_dirs(project_id):
    images = db.project_dir(project_id) / "Assets" / "Images"
    for d in (images, images / "Spectra", images / "Scaled"):
        d.mkdir(parents=True, exist_ok=True)
    return images


def build_tasks(conn, project_id, params):
    """One task per movie in the chosen group, `ref` = MOVIE_ASSET_ID.
    Raises ValueError for anything the API should turn into a 400."""
    group_id = params.get("movie_group_id")
    if group_id is None:
        raise ValueError("movie_group_id is required")
    movies = conn.execute(
        "SELECT ma.* FROM MOVIE_ASSETS ma "
        "JOIN MOVIE_GROUP_MEMBERS gm ON gm.MOVIE_ASSET_ID = ma.MOVIE_ASSET_ID "
        "WHERE gm.GROUP_ID = ? ORDER BY ma.MOVIE_ASSET_ID",
        (int(group_id),),
    ).fetchall()
    if not movies:
        raise ValueError("movie group has no movies")

    previous = {
        r["MOVIE_ASSET_ID"]: r["N"]
        for r in conn.execute(
            "SELECT MOVIE_ASSET_ID, COUNT(*) AS N FROM MOVIE_ALIGNMENT_LIST GROUP BY MOVIE_ASSET_ID"
        ).fetchall()
    }
    images_dir = _ensure_dirs(project_id)

    # Expert options, with MyAlignMoviesPanel::ResetDefaults() as the fallbacks.
    min_shift = _num(params, "min_shift_a", 2.0, float)
    max_shift = _num(params, "max_shift_a", 40.0, float)
    dose_filter = _flag(params, "should_dose_filter", True)
    restore_power = _flag(params, "should_restore_power", True)
    termination = _num(params, "termination_threshold_a", 1.0, float)
    max_iterations = _num(params, "max_iterations", 10, int)
    bfactor = _num(params, "bfactor_a2", 1500.0, float)
    mask_cross = _flag(params, "mask_central_cross", True)
    h_mask = _num(params, "horizontal_mask_px", 1, int)
    v_mask = _num(params, "vertical_mask_px", 1, int)
    save_scaled = _flag(params, "save_scaled_sum", True)
    if _flag(params, "include_all_frames", True):
        first_frame, last_frame = 1, 0
    else:
        first_frame = _num(params, "first_frame", 1, int)
        last_frame = _num(params, "last_frame", 0, int)

    tasks = []
    for index, movie in enumerate(movies):
        asset_id = movie["MOVIE_ASSET_ID"]
        base = Path(movie["FILENAME"]).stem
        n = previous.get(asset_id, 0)
        stem = "{}_{}_{}.mrc".format(base, asset_id, n)
        output = str(images_dir / stem)
        spectrum = str(images_dir / "Spectra" / stem)
        scaled = str(images_dir / "Scaled" / stem) if save_scaled else DEV_NULL

        gain = movie["GAIN_FILENAME"] or ""
        dark = movie["DARK_FILENAME"] or ""
        binning = movie["OUTPUT_BINNING_FACTOR"] or 1.0

        A = jp.arg
        args = [
            A("text", movie["FILENAME"]),                      # 0  input movie
            A("text", output),                                 # 1  output aligned sum
            A("float", movie["PIXEL_SIZE"] or 1.0),            # 2  pixel size (A)
            A("float", min_shift),                             # 3
            A("float", max_shift),                             # 4
            A("bool", dose_filter),                            # 5
            A("bool", restore_power),                          # 6
            A("float", termination),                           # 7
            A("int", max_iterations),                          # 8
            A("float", bfactor),                               # 9
            A("bool", mask_cross),                             # 10
            A("int", h_mask),                                  # 11
            A("int", v_mask),                                  # 12
            A("float", movie["VOLTAGE"] or 300.0),             # 13 kV
            A("float", movie["DOSE_PER_FRAME"] or 1.0),        # 14 exposure per frame (e/A^2)
            A("float", 0.0),                                   # 15 pre-exposure -- not tracked per asset here
            A("bool", gain == ""),                             # 16 movie is gain corrected
            A("text", gain or DEV_NULL),                       # 17
            A("bool", dark == ""),                             # 18 movie is dark corrected
            A("text", dark or DEV_NULL),                       # 19
            A("float", binning),                               # 20 output binning factor
            A("bool", bool(movie["CORRECT_MAG_DISTORTION"])),  # 21
            A("float", movie["MAG_DISTORTION_ANGLE"] or 0.0),  # 22
            A("float", movie["MAG_DISTORTION_MAJOR_SCALE"] or 1.0),  # 23
            A("float", movie["MAG_DISTORTION_MINOR_SCALE"] or 1.0),  # 24
            A("bool", True),                                   # 25 write amplitude spectrum
            A("text", spectrum),                               # 26
            A("bool", save_scaled),                            # 27 write small sum image
            A("text", scaled),                                 # 28
            A("int", first_frame),                             # 29
            A("int", last_frame),                              # 30
            A("int", 1),                                       # 31 frames for running average
            A("int", 1),                                       # 32 max threads (worker overrides from its command line)
            A("bool", False),                                  # 33 save aligned frames
            A("text", DEV_NULL),                               # 34
            A("text", DEV_NULL),                               # 35 shift text file -- results come back over the socket
            A("int", movie["EER_FRAMES_PER_IMAGE"] or 25),     # 36
            A("int", movie["EER_SUPER_RES_FACTOR"] or 1),      # 37
        ]
        assert len(args) == 38
        tasks.append({"index": index, "ref": asset_id, "args": args})
    return tasks


def _arg_values(task):
    return [a["value"] for a in task["args"]]


def _mag_corrected_pixel_size(pixel_size, major, minor):
    # functions.cpp: ReturnMagDistortionCorrectedPixelSize
    return pixel_size / ((major + minor) / 2.0)


def finalize(conn, project_id, job, sent_tasks, task_rows, log):
    """Turn recorded task results into MOVIE_ALIGNMENT_LIST rows, per-frame
    MOVIE_ALIGNMENT_PARAMETERS_<id> tables, updated frame counts, and image
    assets -- MyAlignMoviesPanel::WriteResultToDataBase(), in SQL.

    `sent_tasks` is the task list that went in the package (for the argument
    values); `task_rows` are JOB_TASKS rows. Only `ok` tasks with a `floats`
    result are written; anything else is logged and skipped, so a job that
    lost one movie still records the rest.
    """
    by_index = {t["index"]: t for t in sent_tasks}
    written = skipped = 0
    now = db.now_epoch()

    with conn:
        for row in task_rows:
            index = row["TASK_INDEX"]
            task = by_index.get(index)
            if row["STATUS"] != "ok" or task is None:
                skipped += 1
                continue
            result = json.loads(row["RESULT_JSON"]) if row["RESULT_JSON"] else None
            data = (result or {}).get("data") if (result or {}).get("kind") == "floats" else None
            if not data or len(data) % 2:
                log("task {}: no usable shift array in result; skipped".format(index), level="error")
                skipped += 1
                continue

            v = _arg_values(task)
            movie_id = int(row["REF"]) if row["REF"] is not None else int(task["ref"])
            movie = conn.execute("SELECT * FROM MOVIE_ASSETS WHERE MOVIE_ASSET_ID=?", (movie_id,)).fetchone()
            if movie is None:
                log("task {}: movie asset {} no longer exists; skipped".format(index, movie_id), level="error")
                skipped += 1
                continue

            n_frames = len(data) // 2
            x_shifts, y_shifts = data[:n_frames], data[n_frames:]
            output_file = v[1]
            pixel_size = float(v[2])
            binning = float(v[20]) or 1.0

            # Actual output dimensions from the file unblur wrote; the binned
            # size is an integer resize so the real factor can differ from
            # the requested one (cisTEM re-derives the pixel size from it).
            try:
                hdr = read_image_header(output_file)
                x_size, y_size = hdr["x_size"], hdr["y_size"]
                bin_factor = ((movie["X_SIZE"] / x_size) + (movie["Y_SIZE"] / y_size)) / 2.0 \
                    if movie["X_SIZE"] and movie["Y_SIZE"] else binning
            except (HeaderError, OSError):
                log("task {}: could not read {}; image size estimated from the movie".format(
                    index, os.path.basename(output_file)), level="error")
                x_size = int(round((movie["X_SIZE"] or 0) / binning))
                y_size = int(round((movie["Y_SIZE"] or 0) / binning))
                bin_factor = binning
            final_pixel_size = pixel_size * bin_factor
            if v[21]:
                final_pixel_size = _mag_corrected_pixel_size(final_pixel_size, float(v[23]), float(v[24]))

            cur = conn.execute(
                "INSERT INTO MOVIE_ALIGNMENT_LIST("
                "DATETIME_OF_RUN, ALIGNMENT_JOB_ID, MOVIE_ASSET_ID, OUTPUT_FILE, VOLTAGE, PIXEL_SIZE, "
                "EXPOSURE_PER_FRAME, PRE_EXPOSURE_AMOUNT, MIN_SHIFT, MAX_SHIFT, SHOULD_DOSE_FILTER, "
                "SHOULD_RESTORE_POWER, TERMINATION_THRESHOLD, MAX_ITERATIONS, BFACTOR, "
                "SHOULD_MASK_CENTRAL_CROSS, HORIZONTAL_MASK, VERTICAL_MASK, SHOULD_INCLUDE_ALL_FRAMES_IN_SUM, "
                "FIRST_FRAME_TO_SUM, LAST_FRAME_TO_SUM, FINAL_PIXEL_SIZE) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    now, job["id"], movie_id, output_file, v[13], pixel_size, v[14], v[15], v[3], v[4],
                    1 if v[5] else 0, 1 if v[6] else 0, v[7], v[8], int(v[9]), 1 if v[10] else 0, v[11], v[12],
                    1 if (v[29] == 1 and v[30] == 0) else 0, v[29], v[30], final_pixel_size,
                ),
            )
            alignment_id = cur.lastrowid

            # cisTEM's per-alignment shift table, verbatim.
            table = "MOVIE_ALIGNMENT_PARAMETERS_{}".format(alignment_id)
            conn.execute("CREATE TABLE IF NOT EXISTS {}(FRAME_NUMBER INTEGER PRIMARY KEY, X_SHIFT REAL, Y_SHIFT REAL)".format(table))
            conn.executemany(
                "INSERT INTO {}(FRAME_NUMBER, X_SHIFT, Y_SHIFT) VALUES (?, ?, ?)".format(table),
                [(i + 1, float(x), float(y)) for i, (x, y) in enumerate(zip(x_shifts, y_shifts))],
            )

            # unblur has now actually opened the movie, so this count is authoritative.
            conn.execute("UPDATE MOVIE_ASSETS SET NUMBER_OF_FRAMES=? WHERE MOVIE_ASSET_ID=?", (n_frames, movie_id))

            existing = conn.execute(
                "SELECT IMAGE_ASSET_ID FROM IMAGE_ASSETS WHERE PARENT_MOVIE_ID=? ORDER BY IMAGE_ASSET_ID LIMIT 1",
                (movie_id,),
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE IMAGE_ASSETS SET FILENAME=?, POSITION_IN_STACK=1, ALIGNMENT_ID=?, X_SIZE=?, Y_SIZE=?, "
                    "PIXEL_SIZE=?, VOLTAGE=? WHERE IMAGE_ASSET_ID=?",
                    (output_file, alignment_id, x_size, y_size, final_pixel_size, v[13], existing["IMAGE_ASSET_ID"]),
                )
            else:
                cur = conn.execute(
                    "INSERT INTO IMAGE_ASSETS(NAME, FILENAME, POSITION_IN_STACK, PARENT_MOVIE_ID, ALIGNMENT_ID, "
                    "CTF_ESTIMATION_ID, X_SIZE, Y_SIZE, PIXEL_SIZE, VOLTAGE, SPHERICAL_ABERRATION, PROTEIN_IS_WHITE) "
                    "VALUES (?,?,1,?,?,-1,?,?,?,?,?,?)",
                    (
                        movie["NAME"] + "_aligned", output_file, movie_id, alignment_id, x_size, y_size,
                        final_pixel_size, v[13], movie["SPHERICAL_ABERRATION"], movie["PROTEIN_IS_WHITE"],
                    ),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO IMAGE_GROUP_MEMBERS(GROUP_ID, IMAGE_ASSET_ID) VALUES (0, ?)",
                    (cur.lastrowid,),
                )
            written += 1

    return {"alignments_written": written, "tasks_skipped": skipped}
