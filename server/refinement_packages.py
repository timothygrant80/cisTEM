"""Refinement packages: a particle stack plus everything a refinement needs
to know about each particle -- cisTEM's MyNewRefinementPackageWizard and
MyRefinementPackageAssetPanel, in Python.

cisTEM builds a new package inside the GUI process (MyNewRefinementPackageWizard::
OnFinished): for every particle position in the chosen group it reads the
parent image, replaces outliers beyond 6 sigma with the mean, cuts a box
centred on the pick (padding beyond the image with the image's edge
average), zero-floats and normalises the box to unit variance, inverts it
if the protein is white, and appends it to Assets/ParticleStacks/
particle_stack_<n>.mrc. It then records the package
(REFINEMENT_PACKAGE_ASSETS + four per-package tables) and a first
refinement named "Random Parameters" (REFINEMENT_LIST + per-refinement
tables) whose angles are random, whose CTF values come from each image's
active estimate, and whose resolution statistics are the synthetic curve
ResolutionStatistics::GenerateDefaultStatistics() derives from the
molecular weight. This module does the same, table for table, so a real
cisTEM opening the project sees a package it made itself.

Two of the wizard's sources exist here: a particle position group (the
"New Refinement Package" path) and 2D class-average selections
(classification.py's selection manager) -- the latter re-cuts the members
of the selected classes out of their parent images at the box size asked
for, optionally re-centred by the classification's shifts and with
near-duplicate picks removed, exactly as the wizard's class-selection path
does. Templates, and combining or importing packages, are not mirrored.
"""

import math
import os
import random
import struct
import time
from pathlib import Path

import numpy as np

import db

SYMMETRIES = ("C1", "C2", "C3", "C4", "D2", "D3", "D4", "I", "I2", "O", "T", "T2")  # SymmetryComboBox, in order

_MRC_DTYPES = {0: np.int8, 1: np.int16, 2: np.float32, 6: np.uint16, 12: np.float16}


# ---------------------------------------------------------------------------
# MRC in and out
# ---------------------------------------------------------------------------

def read_mrc_section(path, section=1):
    """One 2D section of an MRC file as float32, rows in file order (the
    first row is the bottom of the image, y up -- cisTEM's convention)."""
    with open(path, "rb") as fh:
        head = fh.read(1024)
        if len(head) < 1024:
            raise ValueError("{} is shorter than an MRC header".format(path))
        endian = ">" if head[212:214] == b"\x11\x11" else "<"
        nx, ny, nz, mode = struct.unpack_from(endian + "iiii", head, 0)
        nsymbt = struct.unpack_from(endian + "i", head, 92)[0]
        if mode not in _MRC_DTYPES:
            raise ValueError("unsupported MRC mode {} in {}".format(mode, path))
        if not (1 <= section <= max(nz, 1)):
            raise ValueError("section {} is not in {} ({} sections)".format(section, path, nz))
        dtype = np.dtype(_MRC_DTYPES[mode]).newbyteorder(endian)
        fh.seek(1024 + max(nsymbt, 0) + (section - 1) * nx * ny * dtype.itemsize)
        data = np.frombuffer(fh.read(nx * ny * dtype.itemsize), dtype=dtype)
        if data.size != nx * ny:
            raise ValueError("{} ends before section {}".format(path, section))
        return data.reshape(ny, nx).astype(np.float32)


class MrcStackWriter:
    """Appends float32 (mode 2) sections to a new MRC stack and writes the
    header on close, with the pixel size in the cell dimensions the way
    cisTEM's MRCFile does."""

    def __init__(self, path, box_size, pixel_size):
        self.path, self.box, self.pixel_size = path, int(box_size), float(pixel_size)
        self.count = 0
        self._sum = self._sumsq = 0.0
        self._min, self._max = float("inf"), float("-inf")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(path, "wb")
        self._fh.write(b"\x00" * 1024)

    def append(self, image):
        arr = np.ascontiguousarray(image, dtype="<f4")
        if arr.shape != (self.box, self.box):
            raise ValueError("particle is {}x{}, box is {}".format(arr.shape[1], arr.shape[0], self.box))
        self._fh.write(arr.tobytes())
        self.count += 1
        self._sum += float(arr.sum()); self._sumsq += float((arr.astype(np.float64) ** 2).sum())
        self._min = min(self._min, float(arr.min())); self._max = max(self._max, float(arr.max()))

    def close(self):
        n = max(self.count, 1) * self.box * self.box
        mean = self._sum / n
        rms = math.sqrt(max(self._sumsq / n - mean * mean, 0.0))
        b, nz = self.box, self.count
        head = bytearray(1024)
        struct.pack_into("<iiii", head, 0, b, b, nz, 2)                       # nx ny nz mode
        struct.pack_into("<iii", head, 28, b, b, nz)                          # mx my mz
        struct.pack_into("<fff", head, 40, b * self.pixel_size, b * self.pixel_size, nz * self.pixel_size)  # cella
        struct.pack_into("<fff", head, 52, 90.0, 90.0, 90.0)                  # cellb
        struct.pack_into("<iii", head, 64, 1, 2, 3)                           # mapc mapr maps
        struct.pack_into("<fff", head, 76, self._min if nz else 0.0, self._max if nz else 0.0, mean)
        struct.pack_into("<ii", head, 88, 0, 0)                               # ispg nsymbt
        head[208:212] = b"MAP "
        head[212:216] = b"\x44\x44\x00\x00"                                   # little-endian machine stamp
        struct.pack_into("<f", head, 216, rms)
        struct.pack_into("<i", head, 220, 1)
        label = "cisTEM3 particle stack, {} particles, {:.4f} A/px".format(nz, self.pixel_size).encode()[:80]
        head[224:224 + len(label)] = label
        self._fh.seek(0); self._fh.write(bytes(head)); self._fh.close()


# ---------------------------------------------------------------------------
# cisTEM's helpers
# ---------------------------------------------------------------------------

def closest_factorized_upper(wanted, largest_factor=3, enforce_even=True):
    """functions.cpp ReturnClosestFactorizedUpper(): the smallest number >=
    wanted whose prime factors are all <= largest_factor (even if asked)."""
    n = int(wanted)
    if enforce_even and n % 2:
        n += 1
    step = 2 if enforce_even else 1
    while True:
        remainder = n
        for factor in range(2, largest_factor + 1):
            while remainder % factor == 0 and remainder > 1:
                remainder //= factor
        if remainder == 1:
            return n
        n += step


def default_box_size(largest_dimension_a, pixel_size):
    """The wizard's guess: 2.2 x the largest dimension in pixels, rounded up
    to an even number with prime factors <= 3."""
    return closest_factorized_upper(int(float(largest_dimension_a) / float(pixel_size) * 2.2), 3, True)


def kda_to_angstrom3(kda):
    # functions.h: protein at 0.81 Da per cubic Angstrom.
    return float(kda) * 1000.0 / 0.81


def default_statistics(molecular_weight_kda, pixel_size, box_size):
    """ResolutionStatistics::Init() + GenerateDefaultStatistics(): the
    synthetic FSC / SSNR curve a new package starts with, one row per shell
    -> [(shell, resolution, fsc, part_fsc, part_ssnr, rec_ssnr)]."""
    number_of_bins = int(box_size) // 2 + 1
    number_of_bins2 = 2 * (number_of_bins - 1)
    extended = int((number_of_bins2 / 2 + 1) * math.sqrt(3.0)) + 1
    diameter = 2.0 * (3.0 * kda_to_angstrom3(molecular_weight_kda) / 4.0 / math.pi) ** (1.0 / 3.0)
    rows = [(0, 0.0, 1.0, 1.0, 1000.0, 1000.0)]
    for i in range(1, extended + 1):
        resolution = float(pixel_size) / float(i) * float(number_of_bins2)
        ssnr = molecular_weight_kda ** 1.5 / 2200.0 * (800.0 * math.exp(-3.5 * diameter / resolution) + math.exp(-25.0 / resolution))
        fsc = ssnr / (2.0 + ssnr)
        rows.append((i, resolution, fsc, fsc, ssnr, ssnr))
    return rows


def assign_subsets(parent_image_ids):
    """The wizard's half-set rule: fewer than 500 particles alternate;
    fewer than 20 micrographs split the stack into ~10 chunks; otherwise by
    the parity of the parent image id."""
    n = len(parent_image_ids)
    images = len(set(parent_image_ids))
    if n < 500:
        return [1 if i % 2 == 0 else 2 for i in range(n)]
    if images < 20:
        chunk = max(n // 10, 1)
        return [1 if (i // chunk) % 2 else 2 for i in range(n)]
    return [1 if pid % 2 else 2 for pid in parent_image_ids]


def _cut_box(image, edge_value, cx, cy, box):
    """Image::ClipInto() for a real-space box centred on pixel (cx, cy) --
    column, row -- padded with `edge_value` where it reaches past the image."""
    half = box // 2
    r0, c0 = cy - half, cx - half
    out = np.full((box, box), edge_value, dtype=np.float32)
    ry0, ry1 = max(r0, 0), min(r0 + box, image.shape[0])
    rx0, rx1 = max(c0, 0), min(c0 + box, image.shape[1])
    if ry1 > ry0 and rx1 > rx0:
        out[ry0 - r0:ry1 - r0, rx0 - c0:rx1 - c0] = image[ry0:ry1, rx0:rx1]
    return out


def _edge_average(image):
    # Image::ReturnAverageOfRealValuesOnEdges(): the mean over the four borders.
    edges = np.concatenate([image[0, :], image[-1, :], image[1:-1, 0], image[1:-1, -1]])
    return float(edges.mean())


def _zero_float_and_normalize(box):
    mean, var = float(box.mean()), float(box.var())
    if var <= 0.0:
        return box - mean
    return (box - mean) / math.sqrt(var)


# ---------------------------------------------------------------------------
# The package
# ---------------------------------------------------------------------------

def _particles_of_group(conn, group_id):
    return conn.execute(
        "SELECT pp.PARTICLE_POSITION_ASSET_ID, pp.PARENT_IMAGE_ASSET_ID, pp.X_POSITION, pp.Y_POSITION, "
        "ia.FILENAME, ia.PIXEL_SIZE, ia.VOLTAGE, ia.SPHERICAL_ABERRATION, ia.PROTEIN_IS_WHITE, "
        "ce.DEFOCUS1, ce.DEFOCUS2, ce.DEFOCUS_ANGLE, ce.ADDITIONAL_PHASE_SHIFT, ce.AMPLITUDE_CONTRAST, ce.TILT_ANGLE, ce.TILT_AXIS "
        "FROM PARTICLE_POSITION_ASSETS pp "
        "JOIN PARTICLE_POSITION_GROUP_MEMBERS m ON m.PARTICLE_POSITION_ASSET_ID = pp.PARTICLE_POSITION_ASSET_ID AND m.GROUP_ID = ? "
        "JOIN IMAGE_ASSETS ia ON ia.IMAGE_ASSET_ID = pp.PARENT_IMAGE_ASSET_ID "
        "LEFT JOIN ESTIMATED_CTF_PARAMETERS ce ON ce.CTF_ESTIMATION_ID = ia.CTF_ESTIMATION_ID "
        "ORDER BY pp.PARENT_IMAGE_ASSET_ID, pp.PARTICLE_POSITION_ASSET_ID", (int(group_id),)).fetchall()


def group_defaults(conn, group_id, largest_dimension_a=150.0):
    """What the wizard prefills from the group: the first particle's image
    pixel size (the output pixel size) and the box size derived from it."""
    row = conn.execute(
        "SELECT ia.PIXEL_SIZE FROM PARTICLE_POSITION_ASSETS pp "
        "JOIN PARTICLE_POSITION_GROUP_MEMBERS m ON m.PARTICLE_POSITION_ASSET_ID = pp.PARTICLE_POSITION_ASSET_ID AND m.GROUP_ID = ? "
        "JOIN IMAGE_ASSETS ia ON ia.IMAGE_ASSET_ID = pp.PARENT_IMAGE_ASSET_ID ORDER BY pp.PARTICLE_POSITION_ASSET_ID LIMIT 1", (int(group_id),)).fetchone()
    pixel_size = float(row["PIXEL_SIZE"]) if row and row["PIXEL_SIZE"] else None
    return {"pixel_size": pixel_size,
            "box_size": default_box_size(largest_dimension_a, pixel_size) if pixel_size else None}


def _particles_of_selections(conn, params):
    """The wizard's class-selection source: the members of the selected
    classes, looked up in their parent package's contained particles for
    where they were picked, then given the same row shape
    _particles_of_group() returns (image, active CTF) so the cutting loop
    needn't know where they came from. `recentre` moves each pick by the
    classification's x/y shift (the wizard's Re-centre picks page);
    `remove_duplicates` then drops picks on the same image closer than
    `duplicate_threshold_a` to one another, keeping the one that moved least
    (the wizard's Remove duplicate picks page; threshold defaults to the
    largest dimension)."""
    import classification  # noqa: E402 -- classification imports this module

    selection_ids = params.get("selection_ids") or []
    if not selection_ids:
        raise ValueError("selection_ids is required")
    members = classification.selection_members(conn, selection_ids)
    if not members:
        raise ValueError("the selected classes hold no particles")
    recentre = bool(params.get("recentre", True))
    remove_duplicates = recentre and bool(params.get("remove_duplicates", True))
    # the wizard's default threshold: a quarter of the largest dimension
    threshold = float(params.get("duplicate_threshold_a") or 0.25 * float(params.get("largest_dimension_a") or 150.0))

    contained_cache = {}
    picks = []
    seen = set()
    for m in members:
        # The same particle in two selections (or two selected classes of
        # different classifications) is one particle.
        key = (int(m["package_id"]), int(m["position_in_stack"]))
        if key in seen:
            continue
        seen.add(key)
        pid = int(m["package_id"])
        if pid not in contained_cache:
            table = "REFINEMENT_PACKAGE_CONTAINED_PARTICLES_{}".format(pid)
            contained_cache[pid] = {r["POSITION_IN_STACK"]: r for r in conn.execute("SELECT * FROM {}".format(table)).fetchall()} \
                if _table_exists(conn, table) else {}
        c = contained_cache[pid].get(m["position_in_stack"])
        if c is None or c["PARENT_IMAGE_ASSET_ID"] is None or c["PARENT_IMAGE_ASSET_ID"] < 0:
            continue  # no parent image to cut from (an imported stack) -- not supported here
        x, y = float(c["X_POSITION"]), float(c["Y_POSITION"])
        shift2 = 0.0
        if recentre:
            x -= float(m["x_shift"]); y -= float(m["y_shift"])
            shift2 = float(m["x_shift"]) ** 2 + float(m["y_shift"]) ** 2
        picks.append({"position_id": c["ORIGINAL_PARTICLE_POSITION_ASSET_ID"], "image_id": c["PARENT_IMAGE_ASSET_ID"], "x": x, "y": y, "shift2": shift2})

    removed = 0
    if remove_duplicates and picks:
        t2 = threshold ** 2
        by_image = {}
        for p in picks:
            by_image.setdefault(p["image_id"], []).append(p)
        kept = []
        for image_picks in by_image.values():
            i = 0
            while i < len(image_picks):
                j = i + 1
                restart = False
                while j < len(image_picks):
                    a, b = image_picks[i], image_picks[j]
                    if (a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 < t2:
                        # Too close: keep whichever moved least, the likelier original pick.
                        if a["shift2"] < b["shift2"]:
                            del image_picks[j]; removed += 1; continue
                        del image_picks[i]; removed += 1; restart = True; break
                    j += 1
                if not restart:
                    i += 1
            kept.extend(image_picks)
        picks = kept

    image_ids = sorted({p["image_id"] for p in picks})
    images = {}
    for iid in image_ids:
        images[iid] = conn.execute(
            "SELECT ia.IMAGE_ASSET_ID, ia.FILENAME, ia.PIXEL_SIZE, ia.VOLTAGE, ia.SPHERICAL_ABERRATION, ia.PROTEIN_IS_WHITE, "
            "ce.DEFOCUS1, ce.DEFOCUS2, ce.DEFOCUS_ANGLE, ce.ADDITIONAL_PHASE_SHIFT, ce.AMPLITUDE_CONTRAST, ce.TILT_ANGLE, ce.TILT_AXIS "
            "FROM IMAGE_ASSETS ia LEFT JOIN ESTIMATED_CTF_PARAMETERS ce ON ce.CTF_ESTIMATION_ID = ia.CTF_ESTIMATION_ID "
            "WHERE ia.IMAGE_ASSET_ID = ?", (iid,)).fetchone()
    picks.sort(key=lambda p: (p["image_id"], p["position_id"] if p["position_id"] is not None else 0))
    rows = []
    for p in picks:
        img = images.get(p["image_id"])
        if img is None:
            continue
        rows.append({"PARTICLE_POSITION_ASSET_ID": p["position_id"], "PARENT_IMAGE_ASSET_ID": p["image_id"], "X_POSITION": p["x"], "Y_POSITION": p["y"],
                     "FILENAME": img["FILENAME"], "PIXEL_SIZE": img["PIXEL_SIZE"], "VOLTAGE": img["VOLTAGE"], "SPHERICAL_ABERRATION": img["SPHERICAL_ABERRATION"],
                     "PROTEIN_IS_WHITE": img["PROTEIN_IS_WHITE"], "DEFOCUS1": img["DEFOCUS1"], "DEFOCUS2": img["DEFOCUS2"], "DEFOCUS_ANGLE": img["DEFOCUS_ANGLE"],
                     "ADDITIONAL_PHASE_SHIFT": img["ADDITIONAL_PHASE_SHIFT"], "AMPLITUDE_CONTRAST": img["AMPLITUDE_CONTRAST"],
                     "TILT_ANGLE": img["TILT_ANGLE"], "TILT_AXIS": img["TILT_AXIS"]})
    return rows, {"members": len(members), "duplicates_removed": removed, "recentred": recentre}


def selection_defaults(conn, selection_ids):
    """What the wizard prefills for a class-selection package -- the parent
    package's values, page by page (MyNewRefinementPackageWizard::OnPageChanged
    with a class-average parent): box size, pixel size, symmetry, molecular
    weight, largest dimension and class count -- and how many particles the
    selections hold."""
    import classification  # noqa: E402

    members = classification.selection_members(conn, selection_ids)
    pkg_ids = sorted({m["package_id"] for m in members})
    pkg = conn.execute("SELECT * FROM REFINEMENT_PACKAGE_ASSETS WHERE REFINEMENT_PACKAGE_ASSET_ID=?", (pkg_ids[0],)).fetchone() if pkg_ids else None
    return {"particle_count": len(members),
            "box_size": pkg["STACK_BOX_SIZE"] if pkg else None,
            "pixel_size": pkg["OUTPUT_PIXEL_SIZE"] if pkg else None,
            "symmetry": pkg["SYMMETRY"] if pkg else None,
            "molecular_weight_kda": pkg["MOLECULAR_WEIGHT"] if pkg else None,
            "largest_dimension_a": pkg["PARTICLE_SIZE"] if pkg else None,
            "number_of_classes": pkg["NUMBER_OF_CLASSES"] if pkg else None,
            "parent_package_ids": pkg_ids}


def create_package(conn, project_id, params, log=None):
    """MyNewRefinementPackageWizard::OnFinished() for a new package from a
    particle position group (`particle_group_id`) or from 2D class-average
    selections (`selection_ids`). Returns the new package's id and particle count."""
    source = {}
    if params.get("selection_ids"):
        particles, source = _particles_of_selections(conn, params)
        if not particles:
            raise ValueError("the selected classes hold no particles that can be cut from an image")
    else:
        group_id = params.get("particle_group_id")
        if group_id is None:
            raise ValueError("particle_group_id or selection_ids is required")
        particles = _particles_of_group(conn, group_id)
        if not particles:
            raise ValueError("the particle position group is empty")
    missing_ctf = sorted({p["PARENT_IMAGE_ASSET_ID"] for p in particles if p["DEFOCUS1"] is None})
    if missing_ctf:
        raise ValueError("image{} {} {} no CTF estimate; run Find CTF first".format(
            "s" if len(missing_ctf) > 1 else "", ", ".join(str(i) for i in missing_ctf[:5]) + ("…" if len(missing_ctf) > 5 else ""),
            "have" if len(missing_ctf) > 1 else "has"))

    symmetry = str(params.get("symmetry") or "C1").upper()
    if symmetry not in SYMMETRIES:
        raise ValueError("unknown symmetry {!r}".format(symmetry))
    molecular_weight = float(params.get("molecular_weight_kda") or 300.0)
    largest_dimension = float(params.get("largest_dimension_a") or 150.0)
    number_of_classes = max(1, int(params.get("number_of_classes") or 1))
    first_pixel_size = float(particles[0]["PIXEL_SIZE"] or 1.0)
    output_pixel_size = float(params.get("output_pixel_size") or first_pixel_size)
    box_size = int(params.get("box_size") or default_box_size(largest_dimension, first_pixel_size))
    if box_size < 16:
        raise ValueError("box size must be at least 16 pixels")

    package_id = conn.execute("SELECT COALESCE(MAX(REFINEMENT_PACKAGE_ASSET_ID), 0) + 1 FROM REFINEMENT_PACKAGE_ASSETS").fetchone()[0]
    refinement_id = conn.execute("SELECT COALESCE(MAX(REFINEMENT_ID), 0) + 1 FROM REFINEMENT_LIST").fetchone()[0]
    name = (params.get("name") or "").strip() or "Refinement Package #{}".format(package_id)
    stack_dir = db.project_dir(project_id) / "Assets" / "ParticleStacks"
    stack_path = str(stack_dir / "particle_stack_{}.mrc".format(package_id))

    # ---- cut the particles ----
    writer = MrcStackWriter(stack_path, box_size, output_pixel_size)
    contained = []
    current_image_id, image, edge = None, None, 0.0
    try:
        for p in particles:
            if p["PARENT_IMAGE_ASSET_ID"] != current_image_id:
                image = read_mrc_section(p["FILENAME"], 1)
                # Image::ReplaceOutliersWithMean(6)
                mean, sigma = float(image.mean()), float(image.std())
                if sigma > 0:
                    image = np.where(np.abs(image - mean) > 6.0 * sigma, np.float32(mean), image)
                edge = _edge_average(image)
                current_image_id = p["PARENT_IMAGE_ASSET_ID"]
                if log:
                    log("cutting particles from image {}".format(current_image_id))
            ps = float(p["PIXEL_SIZE"] or 1.0)
            cx, cy = int(round(p["X_POSITION"] / ps)), int(round(p["Y_POSITION"] / ps))
            box = _zero_float_and_normalize(_cut_box(image, edge, cx, cy, box_size))
            if p["PROTEIN_IS_WHITE"]:
                box = -box
            writer.append(box)
            tilt_angle, tilt_axis = p["TILT_ANGLE"] or 0.0, p["TILT_AXIS"] or 0.0
            d1, d2 = float(p["DEFOCUS1"]), float(p["DEFOCUS2"])
            if tilt_angle or tilt_axis:
                # A tilted specimen: the defocus at this particle follows its height.
                x_rel = (cx - image.shape[1] // 2) * ps
                y_rel = (cy - image.shape[0] // 2) * ps
                a = math.radians(tilt_axis)
                y_rot = math.sin(a) * x_rel + math.cos(a) * y_rel  # RotationMatrix::RotateCoords2D row 2
                height = y_rot * math.tan(math.radians(tilt_angle))
                d1, d2 = d1 + height, d2 + height
            contained.append({
                "position_id": p["PARTICLE_POSITION_ASSET_ID"], "image_id": p["PARENT_IMAGE_ASSET_ID"],
                "position_in_stack": writer.count, "x": float(p["X_POSITION"]), "y": float(p["Y_POSITION"]),
                "pixel_size": ps, "defocus1": d1, "defocus2": d2, "defocus_angle": float(p["DEFOCUS_ANGLE"] or 0.0),
                "phase_shift": float(p["ADDITIONAL_PHASE_SHIFT"] or 0.0), "cs": float(p["SPHERICAL_ABERRATION"] or 2.7),
                "voltage": float(p["VOLTAGE"] or 300.0), "amplitude_contrast": float(p["AMPLITUDE_CONTRAST"] or 0.07),
            })
    finally:
        writer.close()
    subsets = assign_subsets([c["image_id"] for c in contained])
    for c, s in zip(contained, subsets):
        c["subset"] = s

    # ---- the database: Database::AddRefinementPackageAsset() + AddRefinement() ----
    insert_package(conn, package_id, name, stack_path, box_size, output_pixel_size, symmetry, molecular_weight, largest_dimension,
                   number_of_classes, contained, refinement_id)
    rng = random.Random()
    class_rows = []
    for k in range(1, number_of_classes + 1):
        rows = []
        for c in contained:
            # Random Euler angles, uniform over the sphere for theta (the wizard's formula).
            phi = rng.uniform(-1.0, 1.0) * 180.0
            theta = math.degrees(math.acos(max(-1.0, min(1.0, 2.0 * abs(rng.uniform(-1.0, 1.0)) - 1.0))))
            psi = rng.uniform(-1.0, 1.0) * 180.0
            occupancy = 100.0 if number_of_classes == 1 else abs(rng.uniform(-1.0, 1.0) * (200.0 / number_of_classes))
            rows.append((c["position_in_stack"], psi, theta, phi, 0.0, 0.0, c["defocus1"], c["defocus2"], c["defocus_angle"], c["phase_shift"],
                         occupancy, 0.0, 1.0, 0.0, 1, c["pixel_size"], c["voltage"], c["cs"], c["amplitude_contrast"], 0.0, 0.0, 0.0, 0.0, c["subset"]))
        class_rows.append(rows)
    insert_initial_refinement(conn, refinement_id, package_id, "Random Parameters", class_rows, box_size, output_pixel_size, molecular_weight)
    return dict(source, refinement_package_asset_id=package_id, name=name, particles=len(contained), stack_filename=stack_path,
                box_size=box_size, output_pixel_size=output_pixel_size, refinement_id=refinement_id)


def insert_package(conn, package_id, name, stack_path, box_size, output_pixel_size, symmetry, molecular_weight, largest_dimension,
                   number_of_classes, contained, refinement_id, white_protein=False):
    """Database::AddRefinementPackageAsset(): the REFINEMENT_PACKAGE_ASSETS row
    and the package's four tables. `contained` are dicts with position_id,
    image_id, position_in_stack, x, y, pixel_size, defocus1/2, defocus_angle,
    phase_shift, cs, voltage, amplitude_contrast, subset. Every class starts
    at "generate from parameters" (reference -1) and `refinement_id` is the
    package's first refinement, which the caller writes."""
    with conn:
        conn.execute(
            "INSERT INTO REFINEMENT_PACKAGE_ASSETS(REFINEMENT_PACKAGE_ASSET_ID, NAME, STACK_FILENAME, STACK_BOX_SIZE, OUTPUT_PIXEL_SIZE, "
            "SYMMETRY, MOLECULAR_WEIGHT, PARTICLE_SIZE, NUMBER_OF_CLASSES, NUMBER_OF_REFINEMENTS, LAST_REFINEMENT_ID, STACK_HAS_WHITE_PROTEIN) "
            "VALUES (?,?,?,?,?,?,?,?,?,0,?,?)",
            (package_id, name, stack_path, box_size, output_pixel_size, symmetry, molecular_weight, largest_dimension, number_of_classes, refinement_id,
             1 if white_protein else 0))
        conn.execute("CREATE TABLE IF NOT EXISTS REFINEMENT_PACKAGE_CONTAINED_PARTICLES_{}(ORIGINAL_PARTICLE_POSITION_ASSET_ID INTEGER PRIMARY KEY, "
                     "PARENT_IMAGE_ASSET_ID INTEGER, POSITION_IN_STACK INTEGER, X_POSITION REAL, Y_POSITION REAL, PIXEL_SIZE REAL, DEFOCUS_1 REAL, "
                     "DEFOCUS_2 REAL, DEFOCUS_ANGLE REAL, PHASE_SHIFT REAL, SPHERICAL_ABERRATION REAL, MICROSCOPE_VOLTAGE REAL, AMPLITUDE_CONTRAST REAL, "
                     "ASSIGNED_SUBSET INTEGER)".format(package_id))
        conn.execute("CREATE TABLE IF NOT EXISTS REFINEMENT_PACKAGE_CURRENT_REFERENCES_{}(CLASS_NUMBER INTEGER PRIMARY KEY, VOLUME_ASSET_ID INTEGER)".format(package_id))
        conn.execute("CREATE TABLE IF NOT EXISTS REFINEMENT_PACKAGE_REFINEMENTS_LIST_{}(REFINEMENT_NUMBER INTEGER PRIMARY KEY, REFINEMENT_ID INTEGER)".format(package_id))
        conn.execute("CREATE TABLE IF NOT EXISTS REFINEMENT_PACKAGE_CLASSIFICATIONS_LIST_{}(CLASSIFICATION_NUMBER INTEGER PRIMARY KEY, CLASSIFICATION_ID INTEGER)".format(package_id))
        conn.executemany(
            "INSERT INTO REFINEMENT_PACKAGE_CONTAINED_PARTICLES_{} VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)".format(package_id),
            [(c["position_id"], c["image_id"], c["position_in_stack"], c["x"], c["y"], c["pixel_size"], c["defocus1"], c["defocus2"],
              c["defocus_angle"], c["phase_shift"], c["cs"], c["voltage"], c["amplitude_contrast"], c["subset"]) for c in contained])
        # "Generate from parameters" for every class: no reference volume yet.
        conn.executemany("INSERT INTO REFINEMENT_PACKAGE_CURRENT_REFERENCES_{} VALUES (?, -1)".format(package_id),
                         [(k,) for k in range(1, number_of_classes + 1)])
        conn.execute("INSERT INTO REFINEMENT_PACKAGE_REFINEMENTS_LIST_{} VALUES (1, ?)".format(package_id), (refinement_id,))


def insert_initial_refinement(conn, refinement_id, package_id, name, class_rows, box_size, pixel_size, molecular_weight, angular=False):
    """Database::AddRefinement() for a package's first refinement: the
    REFINEMENT_LIST row, a REFINEMENT_DETAILS_<id> row per class with
    ClassRefinementResults' constructor values (occupancy split evenly),
    REFINEMENT_RESULT_<id>_<k> from `class_rows` (tuples in the table's
    column order) and GenerateDefaultStatistics()' curve. The wizard writes
    no angular distribution for random angles; an import does (`angular`)."""
    number_of_classes = len(class_rows)
    number_of_particles = len(class_rows[0]) if class_rows else 0
    now = db.now_epoch()
    with conn:
        conn.execute(
            "INSERT INTO REFINEMENT_LIST(REFINEMENT_ID, REFINEMENT_PACKAGE_ASSET_ID, NAME, RESOLUTION_STATISTICS_ARE_GENERATED, DATETIME_OF_RUN, "
            "STARTING_REFINEMENT_ID, NUMBER_OF_PARTICLES, NUMBER_OF_CLASSES, RESOLUTION_STATISTICS_BOX_SIZE, RESOLUTION_STATISTICS_PIXEL_SIZE, PERCENT_USED) "
            "VALUES (?,?,?,1,?,-1,?,?,?,?,100.0)",
            (refinement_id, package_id, name, now, number_of_particles, number_of_classes, box_size, pixel_size))
        conn.execute("CREATE TABLE IF NOT EXISTS REFINEMENT_DETAILS_{}(CLASS_NUMBER INTEGER PRIMARY KEY, REFERENCE_VOLUME_ASSET_ID INTEGER, LOW_RESOLUTION_LIMIT REAL, "
                     "HIGH_RESOLUTION_LIMIT REAL, MASK_RADIUS REAL, SIGNED_CC_RESOLUTION_LIMIT REAL, GLOBAL_RESOLUTION_LIMIT REAL, GLOBAL_MASK_RADIUS REAL, "
                     "NUMBER_RESULTS_TO_REFINE INTEGER, ANGULAR_SEARCH_STEP REAL, SEARCH_RANGE_X REAL, SEARCH_RANGE_Y REAL, CLASSIFICATION_RESOLUTION_LIMIT REAL, "
                     "SHOULD_FOCUS_CLASSIFY INTEGER, SPHERE_X_COORD REAL, SPHERE_Y_COORD REAL, SPHERE_Z_COORD REAL, SPHERE_RADIUS REAL, SHOULD_REFINE_CTF INTEGER, "
                     "DEFOCUS_SEARCH_RANGE REAL, DEFOCUS_SEARCH_STEP REAL, AVERAGE_OCCUPANCY REAL, ESTIMATED_RESOLUTION REAL, RECONSTRUCTED_VOLUME_ASSET_ID INTEGER, "
                     "RECONSTRUCTION_ID INTEGER, SHOULD_AUTOMASK INTEGER, SHOULD_REFINE_INPUT_PARAMS INTEGER, SHOULD_USE_SUPPLIED_MASK INTEGER, MASK_ASSET_ID INTEGER, "
                     "MASK_EDGE_WIDTH REAL, OUTSIDE_MASK_WEIGHT REAL, SHOULD_LOWPASS_OUTSIDE_MASK INTEGER, MASK_FILTER_RESOLUTION REAL)".format(refinement_id))
        stats = default_statistics(molecular_weight, pixel_size, box_size)
        for k, rows in enumerate(class_rows, 1):
            conn.execute("INSERT INTO REFINEMENT_DETAILS_{} VALUES (?, -1, 0,0,0,0,0,0, 0, 0,0,0,0, 0, 0,0,0,0, 0, 0,0, ?, 0.0, -1, -1, 0, 1, 0, -1, 10.0, 0.0, 0, 30.0)".format(refinement_id),
                         (k, 100.0 / number_of_classes))
            conn.execute("CREATE TABLE IF NOT EXISTS REFINEMENT_RESULT_{}_{}(POSITION_IN_STACK INTEGER PRIMARY KEY, PSI REAL, THETA REAL, PHI REAL, XSHIFT REAL, YSHIFT REAL, "
                         "DEFOCUS1 REAL, DEFOCUS2 REAL, DEFOCUS_ANGLE REAL, PHASE_SHIFT REAL, OCCUPANCY REAL, LOGP REAL, SIGMA REAL, SCORE REAL, IMAGE_IS_ACTIVE INTEGER, "
                         "PIXEL_SIZE REAL, MICROSCOPE_VOLTAGE REAL, MICROSCOPE_CS REAL, AMPLITUDE_CONTRAST REAL, BEAM_TILT_X REAL, BEAM_TILT_Y REAL, IMAGE_SHIFT_X REAL, "
                         "IMAGE_SHIFT_Y REAL, ASSIGNED_SUBSET INTEGER)".format(refinement_id, k))
            conn.executemany("INSERT INTO REFINEMENT_RESULT_{}_{} VALUES ({})".format(refinement_id, k, ",".join("?" * 24)), rows)
            conn.execute("CREATE TABLE IF NOT EXISTS REFINEMENT_RESOLUTION_STATISTICS_{}_{}(SHELL INTEGER PRIMARY KEY, RESOLUTION REAL, FSC REAL, PART_FSC REAL, "
                         "PART_SSNR REAL, REC_SSNR REAL)".format(refinement_id, k))
            conn.executemany("INSERT INTO REFINEMENT_RESOLUTION_STATISTICS_{}_{} VALUES (?,?,?,?,?,?)".format(refinement_id, k), stats)
    if angular:
        import refinements
        refinements.rebuild_angular_distributions(conn, refinement_id)


def list_packages(conn):
    out = []
    for r in conn.execute("SELECT * FROM REFINEMENT_PACKAGE_ASSETS ORDER BY REFINEMENT_PACKAGE_ASSET_ID").fetchall():
        d = {k.lower(): r[k] for k in r.keys()}
        pid = r["REFINEMENT_PACKAGE_ASSET_ID"]
        d["particle_count"] = conn.execute("SELECT COUNT(*) FROM REFINEMENT_PACKAGE_CONTAINED_PARTICLES_{}".format(pid)).fetchone()[0] \
            if _table_exists(conn, "REFINEMENT_PACKAGE_CONTAINED_PARTICLES_{}".format(pid)) else 0
        d["stack_file_exists"] = bool(r["STACK_FILENAME"]) and os.path.isfile(r["STACK_FILENAME"])
        out.append(d)
    return out


def package_particles(conn, package_id, limit=5000):
    table = "REFINEMENT_PACKAGE_CONTAINED_PARTICLES_{}".format(int(package_id))
    if not _table_exists(conn, table):
        return [], 0
    total = conn.execute("SELECT COUNT(*) FROM {}".format(table)).fetchone()[0]
    rows = conn.execute("SELECT * FROM {} ORDER BY POSITION_IN_STACK LIMIT ?".format(table), (limit,)).fetchall()
    return [{k.lower(): r[k] for k in r.keys()} for r in rows], total


def rename_package(conn, package_id, name):
    name = (name or "").strip()
    if not name:
        raise ValueError("a name is required")
    with conn:
        cur = conn.execute("UPDATE REFINEMENT_PACKAGE_ASSETS SET NAME=? WHERE REFINEMENT_PACKAGE_ASSET_ID=?", (name, int(package_id)))
    return cur.rowcount > 0


def delete_package(conn, package_id, remove_stack=True):
    """MyRefinementPackageAssetPanel's Delete: the package row and its
    tables, every refinement of it with theirs, and the stack file."""
    package_id = int(package_id)
    row = conn.execute("SELECT STACK_FILENAME FROM REFINEMENT_PACKAGE_ASSETS WHERE REFINEMENT_PACKAGE_ASSET_ID=?", (package_id,)).fetchone()
    if row is None:
        return False
    with conn:
        for r in conn.execute("SELECT REFINEMENT_ID, NUMBER_OF_CLASSES FROM REFINEMENT_LIST WHERE REFINEMENT_PACKAGE_ASSET_ID=?", (package_id,)).fetchall():
            for k in range(1, (r["NUMBER_OF_CLASSES"] or 1) + 1):
                for prefix in ("REFINEMENT_RESULT", "REFINEMENT_RESOLUTION_STATISTICS", "REFINEMENT_ANGULAR_DISTRIBUTION"):
                    conn.execute("DROP TABLE IF EXISTS {}_{}_{}".format(prefix, r["REFINEMENT_ID"], k))
            conn.execute("DROP TABLE IF EXISTS REFINEMENT_DETAILS_{}".format(r["REFINEMENT_ID"]))
        conn.execute("DELETE FROM REFINEMENT_LIST WHERE REFINEMENT_PACKAGE_ASSET_ID=?", (package_id,))
        # cisTEM's Delete also drops the class selections made on this package's classifications.
        for (sid,) in conn.execute("SELECT SELECTION_ID FROM CLASSIFICATION_SELECTION_LIST WHERE REFINEMENT_PACKAGE_ID=?", (package_id,)).fetchall():
            conn.execute("DROP TABLE IF EXISTS CLASSIFICATION_SELECTION_{}".format(sid))
        conn.execute("DELETE FROM CLASSIFICATION_SELECTION_LIST WHERE REFINEMENT_PACKAGE_ID=?", (package_id,))
        for prefix in ("REFINEMENT_PACKAGE_CONTAINED_PARTICLES", "REFINEMENT_PACKAGE_CURRENT_REFERENCES",
                       "REFINEMENT_PACKAGE_REFINEMENTS_LIST", "REFINEMENT_PACKAGE_CLASSIFICATIONS_LIST"):
            conn.execute("DROP TABLE IF EXISTS {}_{}".format(prefix, package_id))
        conn.execute("DELETE FROM REFINEMENT_PACKAGE_ASSETS WHERE REFINEMENT_PACKAGE_ASSET_ID=?", (package_id,))
    if remove_stack and row["STACK_FILENAME"] and os.path.isfile(row["STACK_FILENAME"]):
        try:
            os.remove(row["STACK_FILENAME"])
        except OSError:
            pass
    return True


def _table_exists(conn, name):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None
