"""Refinement package export and import -- cisTEM's ExportRefinementPackageWizard
and ImportRefinementPackageWizard (src/gui/*RefinementPackageWizard.cpp).

Export writes one refinement's parameters for one class beside a copy of the
package's particle stack, in the formats the wizard offers:

* **frealign** -- `Refinement::WriteSingleClassFrealignParameterFile()`: the
  17-column par file (PSI THETA PHI SHX SHY MAG INCLUDE DF1 DF2 ANGAST PSHIFT
  OCC LogP SIGMA SCORE CHANGE) with the averages line and the total line
  `FrealignParameterFile::WriteLine()` prints, and the stack copied as is.
* **relion** / **relion3** -- the particles star (`data_` for Relion 2,
  `data_optics` + `data_particles` for Relion 3.1), with cisTEM's Euler
  angles written as they are (`_rlnAngleRot` = phi, `_rlnAngleTilt` = theta,
  `_rlnAnglePsi` = psi), the shifts negated and in pixels (Relion 2) or
  angstroms (Relion 3.1), coordinates in pixels, magnification 10000 with
  the pixel size as the detector pixel size, and the stack rewritten as
  `.mrcs`, each particle inverted unless the package holds white protein
  (Relion wants white) and normalised to sigma 1 from the pixels outside
  the particle radius (`ZeroFloatAndNormalize(1.0, radius, true)`). Relion
  3.1 also gets `<name>_corrected_micrographs.star` and one
  `<name>_motioncorr_<image id>.star` of whole-frame shifts per micrograph
  whose particles have an alignment in the project.
* **cistem** -- `Refinement::WriteSingleClasscisTEMStarFile()`'s 24-column
  star file, the format cisTEM's own import reads back, plus the copied
  stack. The desktop wizard doesn't offer this (its programs write these
  files as job inputs); it is here so a package can round-trip.

Import builds a package and its "Imported Parameters" refinement from a
stack plus a parameter file, as `ImportRefinementPackageWizard::OnFinished()`
does: box size from the stack header, one class, particles with
parent image -1 (no image in this project cut them), half sets from the
file where it has them, `GenerateDefaultStatistics()` for the FSC. A cisTEM
star supplies the pixel size, voltage and amplitude contrast per particle;
Frealign and Relion files carry none of those, so they come from the form.
Relion shifts are negated back and phase shifts converted from degrees.
"""
import math
import os
import re
import shutil

import numpy as np

import db
import refinement_packages as rp
import starfile
import refinements

EXPORT_FORMATS = ("frealign", "relion", "relion3", "cistem")
IMPORT_FORMATS = ("cistem", "frealign", "relion")

# Frealign par columns after the position: the 17-record line.
_PAR_HEADER = "C           PSI   THETA     PHI       SHX       SHY     MAG  INCLUDE   DF1      DF2  ANGAST  PSHIFT     OCC      LogP      SIGMA   SCORE  CHANGE"
_PAR_FORMAT = "%7i %7.2f %7.2f %7.2f %9.2f %9.2f %7.0f %5i %8.1f %8.1f %7.2f %7.2f %7.2f %9i %10.4f %7.2f %7.2f"
_PAR_AVERAGE_FORMAT = "C       %7.2f %7.2f %7.2f %9.2f %9.2f %7.0f %5i %8.1f %8.1f %7.2f %7.2f %7.2f %9i %10.4f %7.2f %7.2f"


def _package(conn, package_id):
    row = conn.execute("SELECT * FROM REFINEMENT_PACKAGE_ASSETS WHERE REFINEMENT_PACKAGE_ASSET_ID=?", (int(package_id),)).fetchone()
    if row is None:
        raise KeyError("no refinement package {}".format(package_id))
    return row


def _contained(conn, package_id):
    return conn.execute("SELECT * FROM REFINEMENT_PACKAGE_CONTAINED_PARTICLES_{} ORDER BY POSITION_IN_STACK".format(int(package_id))).fetchall()


def _check_output_path(path, what):
    path = os.path.abspath(os.path.expanduser(str(path or "").strip()))
    if not path or path.endswith(os.sep):
        raise ValueError("{} file name is required".format(what))
    parent = os.path.dirname(path)
    if not os.path.isdir(parent):
        raise ValueError("the folder for the {} file does not exist: {}".format(what, parent))
    if os.path.isdir(path):
        raise ValueError("{} path is a folder: {}".format(what, path))
    return path


def _with_ext(path, ext, force=False):
    root, current = os.path.splitext(path)
    if not current or force:
        return root + ext if current else path + ext
    return path


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_package(conn, package_id, refinement_id, class_number, fmt, stack_path, metadata_path, log=None):
    """ExportRefinementPackageWizard::OnFinished(). Returns the files written."""
    fmt = str(fmt or "").lower()
    if fmt not in EXPORT_FORMATS:
        raise ValueError("format must be one of {}".format(", ".join(EXPORT_FORMATS)))
    pkg = _package(conn, package_id)
    ref = refinements.refinement_row(conn, refinement_id)
    if ref is None or int(ref["REFINEMENT_PACKAGE_ASSET_ID"]) != int(package_id):
        raise ValueError("refinement {} does not belong to package {}".format(refinement_id, package_id))
    class_number = int(class_number or 1)
    if not 1 <= class_number <= int(ref["NUMBER_OF_CLASSES"] or 1):
        raise ValueError("class {} is not one of refinement {}'s {} classes".format(class_number, refinement_id, ref["NUMBER_OF_CLASSES"]))
    if not pkg["STACK_FILENAME"] or not os.path.isfile(pkg["STACK_FILENAME"]):
        raise ValueError("the package's particle stack is missing: {}".format(pkg["STACK_FILENAME"]))
    stack_path = _check_output_path(stack_path, "particle stack")
    metadata_path = _check_output_path(metadata_path, "parameter")
    rows = refinements.load_rows(conn, refinement_id, class_number)
    particles = _contained(conn, package_id)
    if len(rows) != len(particles):
        raise ValueError("refinement {} has {} particles but the package holds {}".format(refinement_id, len(rows), len(particles)))
    files = []
    if fmt == "frealign":
        metadata_path = _with_ext(metadata_path, ".par")
        stack_path = _with_ext(stack_path, ".mrc")
        write_frealign_par(metadata_path, rows)
        files.append(metadata_path)
        shutil.copyfile(pkg["STACK_FILENAME"], stack_path)
        files.append(stack_path)
    elif fmt == "cistem":
        metadata_path = _with_ext(metadata_path, ".star")
        stack_path = _with_ext(stack_path, ".mrc")
        starfile.write_star(metadata_path, rows, starfile.REFINEMENT_KEYS,
                            comments=["Refinement #{} ({}), class {}, package #{} ({})".format(refinement_id, ref["NAME"], class_number, package_id, pkg["NAME"])])
        files.append(metadata_path)
        shutil.copyfile(pkg["STACK_FILENAME"], stack_path)
        files.append(stack_path)
    else:
        stack_path = _with_ext(stack_path, ".mrcs")
        metadata_path = _with_ext(metadata_path, ".star", force=True)
        files.extend(write_relion(conn, pkg, particles, rows, stack_path, metadata_path, relion3=(fmt == "relion3"), log=log))
    return {"format": fmt, "files": files, "particles": len(rows), "refinement_id": int(refinement_id), "class_number": class_number}


def write_frealign_par(path, rows):
    """Refinement::WriteSingleClassFrealignParameterFile()."""
    totals = [0.0] * 17
    with open(path, "w") as fh:
        fh.write(_PAR_HEADER + "\n")
        for r in rows:
            values = [r["position_in_stack"], r["psi"], r["theta"], r["phi"], r["x_shift"], r["y_shift"], 0.0, r["image_is_active"],
                      r["defocus_1"], r["defocus_2"], r["defocus_angle"], r["phase_shift"], r["occupancy"], r["logp"], r["sigma"], r["score"], 0.0]
            for i, v in enumerate(values):
                totals[i] += float(v)
            fh.write(_PAR_FORMAT % (int(values[0]), *[float(v) for v in values[1:7]], int(values[7]), *[float(v) for v in values[8:13]],
                                    starfile.myroundint(values[13]), float(values[14]), float(values[15]), float(values[16])) + "\n")
        n = max(len(rows), 1)
        avg = [t / n for t in totals]
        fh.write(_PAR_AVERAGE_FORMAT % (*avg[1:7], int(avg[7]), *avg[8:13], starfile.myroundint(avg[13]), avg[14], avg[15], avg[16]) + "\n")
        fh.write("C  Total particles included, overall score, average occupancy %11i %10.6f %10.6f\n" % (len(rows), avg[15], avg[12]))
    return path


def _star_loop(fh, block, labels):
    fh.write(" \n{}\n \nloop_\n".format(block))
    for n, label in enumerate(labels, 1):
        fh.write("{} #{}\n".format(label, n))


_RELION_PARTICLE_LABELS = ["_rlnMicrographName", "_rlnCoordinateX", "_rlnCoordinateY", "_rlnImageName", "_rlnDefocusU", "_rlnDefocusV",
                           "_rlnDefocusAngle", "_rlnPhaseShift", "_rlnVoltage", "_rlnSphericalAberration", "_rlnAmplitudeContrast",
                           "_rlnMagnification", "_rlnDetectorPixelSize", "_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"]


def write_relion(conn, pkg, particles, rows, stack_path, star_path, relion3=False, log=None):
    """The Relion branch of the export wizard: the particle stack rewritten
    for Relion, the particles star, and for Relion 3.1 the micrograph and
    motion-correction star files."""
    box = int(pkg["STACK_BOX_SIZE"])
    first = particles[0]
    pixel_size = float(first["PIXEL_SIZE"] or pkg["OUTPUT_PIXEL_SIZE"] or 1.0)
    radius_px = float(pkg["PARTICLE_SIZE"] or 0.0) / 2.0 / pixel_size
    white = bool(pkg["STACK_HAS_WHITE_PROTEIN"])
    files = []

    # ---- the stack: inverted to white protein, normalised outside the particle ----
    yy, xx = np.mgrid[0:box, 0:box]
    outside = ((xx - box // 2) ** 2 + (yy - box // 2) ** 2) > radius_px ** 2
    if not outside.any():
        outside = np.ones((box, box), dtype=bool)
    writer = rp.MrcStackWriter(stack_path, box, pixel_size)
    try:
        for i in range(len(particles)):
            img = rp.read_mrc_section(pkg["STACK_FILENAME"], i + 1)
            if not white:
                img = -img
            ring = img[outside]
            mean, var = float(ring.mean()), float(ring.var())
            img = (img - mean) / math.sqrt(var) if var > 0 else img - mean
            writer.append(img)
            if log and (i + 1) % 1000 == 0:
                log("wrote {} of {} particles".format(i + 1, len(particles)))
    finally:
        writer.close()
    files.append(stack_path)

    # ---- the particles star ----
    image_names = {r["IMAGE_ASSET_ID"]: r["FILENAME"] for r in conn.execute("SELECT IMAGE_ASSET_ID, FILENAME FROM IMAGE_ASSETS").fetchall()}
    base = os.path.splitext(star_path)[0]
    with open(star_path, "w") as fh:
        if relion3:
            _star_loop(fh, "data_optics", ["_rlnOpticsGroup", "_rlnOpticsGroupName", "_rlnAmplitudeContrast", "_rlnSphericalAberration",
                                           "_rlnVoltage", "_rlnImagePixelSize", "_rlnImageSize", "_rlnImageDimensionality"])
            fh.write("%i %s %f %f %f %f %i %i\n" % (1, "opticsGroup1", float(first["AMPLITUDE_CONTRAST"] or 0.07), float(first["SPHERICAL_ABERRATION"] or 2.7),
                                                    float(first["MICROSCOPE_VOLTAGE"] or 300.0), pixel_size, box, 2))
            _star_loop(fh, "data_particles", _RELION_PARTICLE_LABELS + ["_rlnOriginXAngst", "_rlnOriginYAngst", "_rlnOpticsGroup", "_rlnRandomSubset"])
        else:
            _star_loop(fh, "data_", _RELION_PARTICLE_LABELS + ["_rlnOriginX", "_rlnOriginY"])
        random_subset = 2
        for i, (p, r) in enumerate(zip(particles, rows)):
            ps = float(p["PIXEL_SIZE"] or pixel_size)
            micrograph = image_names.get(p["PARENT_IMAGE_ASSET_ID"]) if (p["PARENT_IMAGE_ASSET_ID"] or -1) >= 0 else None
            micrograph = micrograph or "unknown.mrc"
            common = "%s %f %f %06d@%s %f %f %f %f %f %f %f %f %f %f %f %f" % (
                micrograph, float(p["X_POSITION"] or 0.0) / ps, float(p["Y_POSITION"] or 0.0) / ps, i + 1,
                os.path.basename(stack_path) if relion3 else stack_path,
                r["defocus_1"], r["defocus_2"], r["defocus_angle"], r["phase_shift"],
                float(p["MICROSCOPE_VOLTAGE"] or 300.0), float(p["SPHERICAL_ABERRATION"] or 2.7), float(p["AMPLITUDE_CONTRAST"] or 0.07),
                10000.0, ps, r["phi"], r["theta"], r["psi"])
            if relion3:
                if int(r["assigned_subset"] or 0) >= 1:
                    random_subset = int(r["assigned_subset"])
                else:
                    random_subset = 1 if random_subset == 2 else 2
                fh.write(common + " %f %f %i %i\n" % (-r["x_shift"], -r["y_shift"], 1, random_subset))
            else:
                fh.write(common + " %f %f\n" % (-r["x_shift"] / ps, -r["y_shift"] / ps))
    files.append(star_path)

    if relion3:
        files.extend(_write_relion3_micrographs(conn, particles, first, pixel_size, base))
    return files


def _write_relion3_micrographs(conn, particles, first, pixel_size, base):
    """`<base>_corrected_micrographs.star` and a `<base>_motioncorr_<image>.star`
    per parent image with a movie alignment in the project."""
    files = []
    image_ids = []
    for p in particles:
        iid = p["PARENT_IMAGE_ASSET_ID"]
        if iid is not None and iid >= 0 and iid not in image_ids:
            image_ids.append(iid)
    micrographs_path = base + "_corrected_micrographs.star"
    with open(micrographs_path, "w") as mf:
        movie_ps = None
        for iid in image_ids:
            image = conn.execute("SELECT * FROM IMAGE_ASSETS WHERE IMAGE_ASSET_ID=?", (iid,)).fetchone()
            movie = conn.execute("SELECT * FROM MOVIE_ASSETS WHERE MOVIE_ASSET_ID=?", (image["PARENT_MOVIE_ID"],)).fetchone() if image else None
            if movie is not None and movie_ps is None:
                movie_ps = float(movie["PIXEL_SIZE"] or pixel_size)
        _star_loop(mf, "data_optics", ["_rlnOpticsGroupName", "_rlnOpticsGroup", "_rlnMicrographOriginalPixelSize", "_rlnVoltage",
                                       "_rlnSphericalAberration", "_rlnAmplitudeContrast", "_rlnMicrographPixelSize"])
        mf.write("%s %i %f %f %f %f %f\n" % ("opticsGroup1", 1, movie_ps if movie_ps is not None else pixel_size, float(first["MICROSCOPE_VOLTAGE"] or 300.0),
                                             float(first["SPHERICAL_ABERRATION"] or 2.7), float(first["AMPLITUDE_CONTRAST"] or 0.07), pixel_size))
        _star_loop(mf, "data_micrographs", ["_rlnMicrographName", "_rlnMicrographMetadata", "_rlnOpticsGroup"])
        for iid in image_ids:
            image = conn.execute("SELECT * FROM IMAGE_ASSETS WHERE IMAGE_ASSET_ID=?", (iid,)).fetchone()
            if image is None:
                continue
            movie = conn.execute("SELECT * FROM MOVIE_ASSETS WHERE MOVIE_ASSET_ID=?", (image["PARENT_MOVIE_ID"],)).fetchone()
            alignment = conn.execute("SELECT * FROM MOVIE_ALIGNMENT_LIST WHERE ALIGNMENT_ID=?", (image["ALIGNMENT_ID"],)).fetchone()
            shifts_table = "MOVIE_ALIGNMENT_PARAMETERS_{}".format(image["ALIGNMENT_ID"])
            if movie is None or alignment is None or not rp._table_exists(conn, shifts_table):
                continue  # an imported micrograph: no motion correction to describe
            mc_path = "{}_motioncorr_{:06d}.star".format(base, int(iid))
            mf.write("%s %s %i\n" % (image["FILENAME"], mc_path, 1))
            mps = float(movie["PIXEL_SIZE"] or pixel_size)
            with open(mc_path, "w") as cf:
                cf.write(" \ndata_general\n \n")
                cf.write("_rlnImageSizeX %i\n" % int(movie["X_SIZE"] or 0))
                cf.write("_rlnImageSizeY %i\n" % int(movie["Y_SIZE"] or 0))
                cf.write("_rlnImageSizeZ %i\n" % int(movie["NUMBER_OF_FRAMES"] or 0))
                cf.write("_rlnMicrographMovieName %s\n" % movie["FILENAME"])
                cf.write("_rlnMicrographGainName %s\n" % (movie["GAIN_FILENAME"] or ""))
                cf.write("_rlnMicrographBinning %f\n" % float(movie["OUTPUT_BINNING_FACTOR"] or 1.0))
                cf.write("_rlnMicrographOriginalPixelSize %f\n" % mps)
                cf.write("_rlnMicrographDoseRate %f\n" % float(movie["DOSE_PER_FRAME"] or 0.0))
                cf.write("_rlnMicrographPreExposure %f\n" % float(alignment["PRE_EXPOSURE_AMOUNT"] or 0.0))
                cf.write("_rlnMicrographVoltage %f\n" % float(movie["VOLTAGE"] or 300.0))
                cf.write("_rlnMicrographStartFrame %i\n" % int(alignment["FIRST_FRAME_TO_SUM"] or 1))
                _star_loop(cf, "data_global_shift", ["_rlnMicrographFrameNumber", "_rlnMicrographShiftX", "_rlnMicrographShiftY"])
                first_x = first_y = 0.0
                for frame, x, y in conn.execute("SELECT * FROM {} ORDER BY 1".format(shifts_table)).fetchall():
                    if int(frame) == 1:
                        first_x, first_y = float(x), float(y)
                    # Relion wants pixels of the original movie; unblur reported angstroms.
                    cf.write("%i %f %f\n" % (int(frame), (float(x) - first_x) / mps, (float(y) - first_y) / mps))
            files.append(mc_path)
    files.insert(0, micrographs_path)
    return files


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def stack_details(path):
    """GetMRCDetails(): x, y and section count of a stack, or a ValueError."""
    path = str(path or "").strip()
    if not path or not os.path.isfile(path):
        raise ValueError("cannot read the stack file: {}".format(path or "(none)"))
    try:
        head = rp.read_mrc_header(path) if hasattr(rp, "read_mrc_header") else None
    except Exception:
        head = None
    if head is None:
        import volumes
        try:
            head = volumes.read_mrc_header(path)
        except Exception as exc:
            raise ValueError("cannot read the stack file: {}".format(exc))
    if head["nx"] <= 0 or head["ny"] <= 0 or head["nz"] <= 0:
        raise ValueError("cannot read the stack file - aborting")
    return head


def read_frealign_par(path):
    """FrealignParameterFile::ReadFile(): rows keyed as star rows. Lines
    starting with C are comments; 16-column lines are the old format without
    the phase shift."""
    rows = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("C"):
                continue
            tok = s.split()
            try:
                v = [float(t) for t in tok]
            except ValueError:
                continue
            if len(v) == 16:
                v.insert(11, 0.0)
            if len(v) < 17:
                continue
            rows.append({"position_in_stack": int(v[0]), "psi": v[1], "theta": v[2], "phi": v[3], "x_shift": v[4], "y_shift": v[5],
                         "image_is_active": int(v[7]), "defocus_1": v[8], "defocus_2": v[9], "defocus_angle": v[10], "phase_shift": v[11],
                         "occupancy": v[12], "logp": v[13], "sigma": v[14], "score": v[15]})
    return rows


_RELION_COLUMNS = {"_rlnAngleRot": "phi", "_rlnAngleTilt": "theta", "_rlnAnglePsi": "psi", "_rlnCoordinateX": "x_coord", "_rlnCoordinateY": "y_coord",
                   "_rlnOriginX": "x_shift", "_rlnOriginY": "y_shift", "_rlnOriginXAngst": "x_shift_angst", "_rlnOriginYAngst": "y_shift_angst",
                   "_rlnDefocusU": "defocus_1", "_rlnDefocusV": "defocus_2", "_rlnDefocusAngle": "defocus_angle", "_rlnPhaseShift": "phase_shift",
                   "_rlnRandomSubset": "assigned_subset", "_rlnImageName": "image_name", "_rlnMicrographName": "micrograph_name"}


def read_relion_star(path):
    """BasicStarFileReader::ReadFile(): the particle rows of a Relion star
    file (the `data_particles` block of a 3.1 file, else the first loop that
    carries `_rlnAngleRot`). Requires the three angles, the origins and
    DefocusU; the rest default. Positions in the stack are the line numbers,
    shifts in pixels unless the Angst columns are present, phase shifts in
    degrees (converted to radians as the reader does)."""
    blocks, current = [], None
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith(";"):
                continue
            if s.startswith("data_"):
                current = {"name": s, "labels": [], "rows": []}
                blocks.append(current)
            elif current is None:
                continue
            elif s == "loop_":
                current["labels"] = []
            elif s.startswith("_"):
                current["labels"].append(s.split()[0])
            else:
                current["rows"].append(s.split())
    wanted = [b for b in blocks if b["name"] == "data_particles"] or [b for b in blocks if "_rlnAngleRot" in b["labels"]]
    if not wanted:
        raise ValueError("Couldn't find _rlnAngleRot in star file ({})".format(path))
    block = wanted[0]
    labels = block["labels"]
    for needed in ("_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi", "_rlnDefocusU"):
        if needed not in labels:
            raise ValueError("Couldn't find {} in star file ({})".format(needed, path))
    if not any(l in labels for l in ("_rlnOriginX", "_rlnOriginXAngst")):
        raise ValueError("Couldn't find _rlnOriginX in star file ({})".format(path))
    in_angst = "_rlnOriginXAngst" in labels
    index = {_RELION_COLUMNS[l]: i for i, l in enumerate(labels) if l in _RELION_COLUMNS}
    rows = []
    for n, tok in enumerate(block["rows"], 1):
        if len(tok) < len(labels):
            continue

        def val(key, default=0.0):
            i = index.get(key)
            if i is None:
                return default
            try:
                return float(tok[i])
            except ValueError:
                raise ValueError("Error: Converting to a number ({})".format(tok[i]))
        row = {"position_in_stack": n, "phi": val("phi"), "theta": val("theta"), "psi": val("psi"),
               "x_shift": val("x_shift_angst") if in_angst else val("x_shift"), "y_shift": val("y_shift_angst") if in_angst else val("y_shift"),
               "defocus_1": val("defocus_1"), "defocus_angle": val("defocus_angle"), "phase_shift": math.radians(val("phase_shift")),
               "x_coord": val("x_coord"), "y_coord": val("y_coord"), "assigned_subset": int(val("assigned_subset", -1))}
        row["defocus_2"] = val("defocus_2", row["defocus_1"])
        rows.append(row)
    return rows, in_angst


def import_package(conn, project_id, params, log=None):
    """ImportRefinementPackageWizard::OnFinished(). `params`: format
    (cistem | frealign | relion), stack_path, metadata_path, symmetry,
    molecular_weight_kda, largest_dimension_a, protein_is_white, name, and
    for Frealign / Relion: pixel_size_a, voltage_kv, cs_mm,
    amplitude_contrast (a cisTEM star carries those per particle; Cs comes
    from the form in every case, as the wizard's does)."""
    fmt = str(params.get("format") or "cistem").lower()
    if fmt not in IMPORT_FORMATS:
        raise ValueError("format must be one of {}".format(", ".join(IMPORT_FORMATS)))
    stack_path = os.path.abspath(os.path.expanduser(str(params.get("stack_path") or "").strip()))
    metadata_path = os.path.abspath(os.path.expanduser(str(params.get("metadata_path") or "").strip()))
    head = stack_details(stack_path)
    if head["nx"] != head["ny"]:
        raise ValueError("Only square images are currently supported - aborting")
    if not os.path.isfile(metadata_path):
        raise ValueError("cannot read the parameter file: {}".format(metadata_path or "(none)"))
    n_images, box = int(head["nz"]), int(head["nx"])

    symmetry = str(params.get("symmetry") or "C1").upper()
    if symmetry not in rp.SYMMETRIES:
        raise ValueError("unknown symmetry {!r}".format(symmetry))
    molecular_weight = float(params.get("molecular_weight_kda") or 300.0)
    largest_dimension = float(params.get("largest_dimension_a") or 150.0)
    white = bool(params.get("protein_is_white"))
    cs = float(params.get("cs_mm") or 2.7)

    def form_float(key, what):
        v = params.get(key)
        if v in (None, ""):
            raise ValueError("{} is required for a {} import".format(what, fmt))
        return float(v)

    if fmt == "cistem":
        star = starfile.read_star(metadata_path)
        if len(star) != n_images:
            raise ValueError("Number of images in stack ({}) is different from the number of lines in the star file ({}) - aborting".format(n_images, len(star)))
        pixel_size = float(star[0].get("pixel_size") or 0.0) or form_float("pixel_size_a", "pixel size")
        rows = []
        contained = []
        for i, s in enumerate(star):
            ps = float(s.get("pixel_size") or pixel_size)
            pos = int(s.get("position_in_stack") or (i + 1))
            contained.append({"position_id": pos, "image_id": -1, "position_in_stack": pos, "x": 0.0, "y": 0.0, "pixel_size": ps,
                              "defocus1": s.get("defocus_1", 0.0), "defocus2": s.get("defocus_2", 0.0), "defocus_angle": s.get("defocus_angle", 0.0),
                              "phase_shift": s.get("phase_shift", 0.0), "cs": cs, "voltage": s.get("voltage", 300.0),
                              "amplitude_contrast": s.get("amplitude_contrast", 0.07), "subset": int(s.get("assigned_subset") or 0)})
            rows.append((pos, s.get("psi", 0.0), s.get("theta", 0.0), s.get("phi", 0.0), s.get("x_shift", 0.0), s.get("y_shift", 0.0),
                         s.get("defocus_1", 0.0), s.get("defocus_2", 0.0), s.get("defocus_angle", 0.0), s.get("phase_shift", 0.0),
                         s.get("occupancy", 0.0), s.get("logp", 0.0), s.get("sigma", 0.0), s.get("score", 0.0), int(s.get("image_is_active", 1)),
                         ps, s.get("voltage", 300.0), s.get("cs", cs), s.get("amplitude_contrast", 0.07),
                         s.get("beam_tilt_x", 0.0), s.get("beam_tilt_y", 0.0), s.get("image_shift_x", 0.0), s.get("image_shift_y", 0.0),
                         int(s.get("assigned_subset") or 0)))
        label = "cisTEM Import"
    else:
        pixel_size = form_float("pixel_size_a", "pixel size")
        voltage = form_float("voltage_kv", "microscope voltage")
        amplitude_contrast = form_float("amplitude_contrast", "amplitude contrast")
        if fmt == "frealign":
            par = read_frealign_par(metadata_path)
            if len(par) != n_images:
                raise ValueError("Number of images in stack ({}) is different from the number of lines in the par file ({}) - aborting".format(n_images, len(par)))
            label = "Frealign Import"
        else:
            par, in_angst = read_relion_star(metadata_path)
            if len(par) != n_images:
                raise ValueError("Number of images({}) in stack is different from the number of parameters read from the star file({}) - aborting".format(n_images, len(par)))
            label = "Relion Import"
        rows, contained = [], []
        chunk = max(n_images // 10, 1)
        for i, r in enumerate(par):
            pos = int(r["position_in_stack"])
            if fmt == "frealign":
                # the wizard splits the stack in ten and alternates the halves
                subset = 1 if (i // chunk) % 2 else 2
                x = y = 0.0
                xs, ys = r["x_shift"], r["y_shift"]
                occupancy, logp, sigma, score, active = r["occupancy"], r["logp"], r["sigma"], r["score"], r["image_is_active"]
            else:
                subset = int(r["assigned_subset"])
                x, y = r["x_coord"] * pixel_size, r["y_coord"] * pixel_size
                xs = -r["x_shift"] if in_angst else -r["x_shift"] * pixel_size
                ys = -r["y_shift"] if in_angst else -r["y_shift"] * pixel_size
                occupancy, logp, sigma, score, active = 100.0, 0.0, 10.0, 0.0, 1
            contained.append({"position_id": pos, "image_id": -1, "position_in_stack": pos, "x": x, "y": y, "pixel_size": pixel_size,
                              "defocus1": r["defocus_1"], "defocus2": r["defocus_2"], "defocus_angle": r["defocus_angle"], "phase_shift": r["phase_shift"],
                              "cs": cs, "voltage": voltage, "amplitude_contrast": amplitude_contrast, "subset": subset})
            rows.append((pos, r["psi"], r["theta"], r["phi"], xs, ys, r["defocus_1"], r["defocus_2"], r["defocus_angle"], r["phase_shift"],
                         occupancy, logp, sigma, score, active, pixel_size, voltage, cs, amplitude_contrast, 0.0, 0.0, 0.0, 0.0, subset))
    if len({c["position_in_stack"] for c in contained}) != len(contained):
        raise ValueError("the parameter file repeats a position in the stack")

    package_id = conn.execute("SELECT COALESCE(MAX(REFINEMENT_PACKAGE_ASSET_ID), 0) + 1 FROM REFINEMENT_PACKAGE_ASSETS").fetchone()[0]
    refinement_id = refinements.next_refinement_id(conn)
    name = (params.get("name") or "").strip() or "Refinement Package #{} ({})".format(package_id, label)
    rp.insert_package(conn, package_id, name, stack_path, box, pixel_size, symmetry, molecular_weight, largest_dimension, 1, contained,
                      refinement_id, white_protein=white)
    rp.insert_initial_refinement(conn, refinement_id, package_id, "Imported Parameters", [rows], box, pixel_size, molecular_weight, angular=True)
    if log:
        log("imported {} particles as package #{} ({})".format(len(contained), package_id, name))
    return {"refinement_package_asset_id": package_id, "name": name, "particles": len(contained), "stack_filename": stack_path,
            "box_size": box, "output_pixel_size": pixel_size, "refinement_id": refinement_id, "format": fmt}
