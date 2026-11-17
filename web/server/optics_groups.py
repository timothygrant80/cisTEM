"""Optics groups (web/CLAUDE.md, "Optics groups"): which images share a beam tilt and higher-order
aberration set -- RELION 3.1's model. The group is a column of the image (and its movie), copied
into a package when it is cut; this module reads the acquisition metadata the grouping can be made
from and proposes and applies groupings.

Acquisition metadata, read beside each movie at import (and on demand for older projects):
  * EPU's XML, written next to every movie (``<name>.xml`` for ``<name>_fractions.tiff``,
    ``<name>_Fractions.mrc`` or ``<name>_EER.eer``): ``BeamShift`` (``_x``, ``_y``, in EPU's
    units, which are what RELION's AFIS grouping scripts cluster) and ``acquisitionDateTime``.
  * SerialEM's mdoc (``<movie>.mdoc`` or ``<stem>.mdoc``): ``ImageShift = x y`` (microns) and
    ``DateTime``.

Groupings (preview() / apply()):
  * ``image_groups``: every chosen image group becomes an optics group.
  * ``filename``: a regular expression with one capture; the distinct captures become groups
    (EPU's hole-template position ``_Data_<x>_<y>_`` is the image-shift position).
  * ``time``: by calendar day, or a new group whenever the gap between successive acquisition
    times exceeds a number of hours.
  * ``beam_shift``: k-means on the (x, y) beam shifts, k asked for or chosen by the largest
    jump in the sorted nearest-neighbour distances (the AFIS positions are well separated
    clusters, so the jump is unmistakable).
"""
import datetime as _dt
import math
import os
import re
import xml.etree.ElementTree as ET

import numpy as np

METHODS = ("image_groups", "filename", "time", "beam_shift")

FILENAME_PRESETS = {
    "epu_position": {"label": "EPU hole-template position (_Data_<x>_<y>_)", "pattern": r"_Data_(\d+_\d+)_"},
    "epu_gridsquare": {"label": "EPU grid square (GridSquare_<n>)", "pattern": r"(GridSquare_\d+)"},
    "epu_date": {"label": "EPU date in the name (_<yyyymmdd>_)", "pattern": r"_(\d{8})_\d{6}"},
    "serialem_prefix": {"label": "SerialEM name before the number (<prefix>_<n>)", "pattern": r"^(.*)_\d+$"},
}

# ---------------------------------------------------------------------- sidecars

_SIDECAR_SUFFIXES = ("_fractions", "_Fractions", "_EER", "_eer", "_frames", "_Frames")


def sidecar_candidates(movie_path):
    """The EPU XML and SerialEM mdoc files that could belong to a movie, in the order tried."""
    d, name = os.path.split(str(movie_path))
    stem = name.rsplit(".", 1)[0] if "." in name else name
    stems = [stem]
    for suffix in _SIDECAR_SUFFIXES:
        if stem.endswith(suffix):
            stems.append(stem[:-len(suffix)])
    out = []
    for s in stems:
        out.append(os.path.join(d, s + ".xml"))
    out.append(os.path.join(d, name + ".mdoc"))
    for s in stems:
        out.append(os.path.join(d, s + ".mdoc"))
    return out


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _epoch(text):
    """An ISO 8601 time (EPU's acquisitionDateTime) as epoch seconds, else None."""
    if not text:
        return None
    t = text.strip()
    try:
        t = re.sub(r"(\.\d{6})\d+", r"\1", t)          # EPU writes seven fractional digits; fromisoformat takes six
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        d = _dt.datetime.fromisoformat(t)
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=_dt.timezone.utc)
    return int(d.timestamp())


def read_epu_xml(path):
    """BeamShift (x, y) and acquisitionDateTime from an EPU XML, as {beam_shift_x, beam_shift_y, acquisition_time}
    with None for what is missing; {} when the file is not EPU's."""
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return {}
    out = {"beam_shift_x": None, "beam_shift_y": None, "acquisition_time": None}
    found = False
    for el in root.iter():
        name = _local(el.tag)
        if name == "BeamShift":
            vals = {_local(c.tag): c.text for c in el}
            try:
                out["beam_shift_x"], out["beam_shift_y"] = float(vals.get("_x")), float(vals.get("_y"))
                found = True
            except (TypeError, ValueError):
                pass
        elif name == "acquisitionDateTime":
            out["acquisition_time"] = _epoch(el.text)
            found = found or out["acquisition_time"] is not None
    return out if found else {}


_MDOC_DATE_FORMATS = ("%d-%b-%y %H:%M:%S", "%d-%b-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d-%b-%y  %H:%M:%S")


def read_mdoc(path):
    """ImageShift (x, y) and DateTime from a SerialEM mdoc (the first section's), as read_epu_xml's dict."""
    out = {"beam_shift_x": None, "beam_shift_y": None, "acquisition_time": None}
    found = False
    try:
        with open(path, errors="replace") as fh:
            for line in fh:
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if key == "ImageShift" and out["beam_shift_x"] is None:
                    parts = value.split()
                    if len(parts) >= 2:
                        try:
                            out["beam_shift_x"], out["beam_shift_y"] = float(parts[0]), float(parts[1])
                            found = True
                        except ValueError:
                            pass
                elif key == "DateTime" and out["acquisition_time"] is None:
                    text = " ".join(value.split())
                    for fmt in _MDOC_DATE_FORMATS:
                        try:
                            d = _dt.datetime.strptime(text, fmt.replace("  ", " "))
                            out["acquisition_time"] = int(d.replace(tzinfo=_dt.timezone.utc).timestamp())
                            found = True
                            break
                        except ValueError:
                            continue
                if found and out["beam_shift_x"] is not None and out["acquisition_time"] is not None:
                    break
    except OSError:
        return {}
    return out if found else {}


def read_sidecar(movie_path):
    """The acquisition metadata beside a movie: the first EPU XML or mdoc among sidecar_candidates() that
    yields anything, else {} (the import then records nothing and the file's mtime stands for the time)."""
    for p in sidecar_candidates(movie_path):
        if not os.path.isfile(p):
            continue
        meta = read_epu_xml(p) if p.lower().endswith(".xml") else read_mdoc(p)
        if meta:
            return meta
    return {}


def file_time(path):
    try:
        return int(os.stat(path).st_mtime)
    except OSError:
        return None


def metadata_for_movie(path):
    """What the import records: the sidecar's values, the file's mtime when the sidecar gives no time."""
    meta = read_sidecar(path)
    if meta.get("acquisition_time") is None:
        meta["acquisition_time"] = file_time(path)
    return {"beam_shift_x": meta.get("beam_shift_x"), "beam_shift_y": meta.get("beam_shift_y"), "acquisition_time": meta.get("acquisition_time")}


def read_metadata(conn):
    """Fill the beam shift and acquisition time of every movie and image that has none (projects from
    before the columns, or movies imported without a sidecar that has since appeared). Returns the counts."""
    movies_read = images_read = 0
    with conn:
        for r in conn.execute("SELECT MOVIE_ASSET_ID, FILENAME FROM MOVIE_ASSETS WHERE BEAM_SHIFT_X IS NULL OR ACQUISITION_TIME IS NULL").fetchall():
            meta = metadata_for_movie(r["FILENAME"])
            if meta["beam_shift_x"] is not None or meta["acquisition_time"] is not None:
                conn.execute("UPDATE MOVIE_ASSETS SET BEAM_SHIFT_X=COALESCE(BEAM_SHIFT_X, ?), BEAM_SHIFT_Y=COALESCE(BEAM_SHIFT_Y, ?), "
                             "ACQUISITION_TIME=COALESCE(ACQUISITION_TIME, ?) WHERE MOVIE_ASSET_ID=?",
                             (meta["beam_shift_x"], meta["beam_shift_y"], meta["acquisition_time"], r["MOVIE_ASSET_ID"]))
                movies_read += 1
        # images from a movie take the movie's; imported micrographs read their own sidecar
        images_read += conn.execute(
            "UPDATE IMAGE_ASSETS SET BEAM_SHIFT_X=(SELECT BEAM_SHIFT_X FROM MOVIE_ASSETS m WHERE m.MOVIE_ASSET_ID=IMAGE_ASSETS.PARENT_MOVIE_ID), "
            "BEAM_SHIFT_Y=(SELECT BEAM_SHIFT_Y FROM MOVIE_ASSETS m WHERE m.MOVIE_ASSET_ID=IMAGE_ASSETS.PARENT_MOVIE_ID), "
            "ACQUISITION_TIME=(SELECT ACQUISITION_TIME FROM MOVIE_ASSETS m WHERE m.MOVIE_ASSET_ID=IMAGE_ASSETS.PARENT_MOVIE_ID) "
            "WHERE PARENT_MOVIE_ID > 0 AND (BEAM_SHIFT_X IS NULL OR ACQUISITION_TIME IS NULL) "
            "AND EXISTS (SELECT 1 FROM MOVIE_ASSETS m WHERE m.MOVIE_ASSET_ID=IMAGE_ASSETS.PARENT_MOVIE_ID AND (m.BEAM_SHIFT_X IS NOT NULL OR m.ACQUISITION_TIME IS NOT NULL))").rowcount
        for r in conn.execute("SELECT IMAGE_ASSET_ID, FILENAME FROM IMAGE_ASSETS WHERE (PARENT_MOVIE_ID IS NULL OR PARENT_MOVIE_ID <= 0) "
                              "AND (BEAM_SHIFT_X IS NULL OR ACQUISITION_TIME IS NULL)").fetchall():
            meta = metadata_for_movie(r["FILENAME"])
            if meta["beam_shift_x"] is not None or meta["acquisition_time"] is not None:
                conn.execute("UPDATE IMAGE_ASSETS SET BEAM_SHIFT_X=COALESCE(BEAM_SHIFT_X, ?), BEAM_SHIFT_Y=COALESCE(BEAM_SHIFT_Y, ?), "
                             "ACQUISITION_TIME=COALESCE(ACQUISITION_TIME, ?) WHERE IMAGE_ASSET_ID=?",
                             (meta["beam_shift_x"], meta["beam_shift_y"], meta["acquisition_time"], r["IMAGE_ASSET_ID"]))
                images_read += 1
    return {"movies_read": movies_read, "images_read": images_read}


# ---------------------------------------------------------------------- the groupings

def _images(conn, image_group_id):
    sql = ("SELECT ia.IMAGE_ASSET_ID, ia.NAME, ia.FILENAME, ia.PARENT_MOVIE_ID, ia.OPTICS_GROUP, ia.BEAM_SHIFT_X, ia.BEAM_SHIFT_Y, ia.ACQUISITION_TIME "
           "FROM IMAGE_ASSETS ia")
    if image_group_id is None or int(image_group_id) == 0:
        return conn.execute(sql + " ORDER BY ia.IMAGE_ASSET_ID").fetchall()
    return conn.execute(sql + " JOIN IMAGE_GROUP_MEMBERS m ON m.IMAGE_ASSET_ID = ia.IMAGE_ASSET_ID AND m.GROUP_ID = ? ORDER BY ia.IMAGE_ASSET_ID",
                        (int(image_group_id),)).fetchall()


def _by_image_groups(conn, images, group_ids):
    ids = [int(g) for g in (group_ids or []) if int(g) != 0]
    if not ids:
        raise ValueError("choose at least one image group")
    names = {r["GROUP_ID"]: r["GROUP_NAME"] for r in conn.execute("SELECT GROUP_ID, GROUP_NAME FROM IMAGE_GROUP_LIST")}
    member_of = {}
    for n, gid in enumerate(ids, 1):
        if gid not in names:
            raise ValueError("image group {} does not exist".format(gid))
        for r in conn.execute("SELECT IMAGE_ASSET_ID FROM IMAGE_GROUP_MEMBERS WHERE GROUP_ID=?", (gid,)):
            member_of.setdefault(r["IMAGE_ASSET_ID"], n)      # an image in two chosen groups goes with the first
    labels = {n: names[gid] for n, gid in enumerate(ids, 1)}
    assignment = {r["IMAGE_ASSET_ID"]: member_of.get(r["IMAGE_ASSET_ID"], 0) for r in images}
    return assignment, labels


def _by_filename(images, pattern):
    if not pattern:
        raise ValueError("a pattern with one capture group is needed")
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        raise ValueError("bad pattern: {}".format(exc))
    if rx.groups < 1:
        raise ValueError("the pattern needs a capture group (in parentheses) for the part that names the group")
    captures = {}
    assignment = {}
    for r in images:
        base = os.path.basename(r["FILENAME"] or r["NAME"] or "")
        stem = base.rsplit(".", 1)[0] if "." in base else base
        m = rx.search(stem) or rx.search(base)
        if not m or m.group(1) is None:
            assignment[r["IMAGE_ASSET_ID"]] = 0
            continue
        key = m.group(1)
        if key not in captures:
            captures[key] = len(captures) + 1
        assignment[r["IMAGE_ASSET_ID"]] = captures[key]
    labels = {n: key for key, n in captures.items()}
    return assignment, labels


def _by_time(images, mode="day", gap_hours=4.0):
    times = [(r["IMAGE_ASSET_ID"], r["ACQUISITION_TIME"]) for r in images]
    if all(t is None for _, t in times):
        raise ValueError("no image has an acquisition time; read the EPU / mdoc metadata first")
    assignment, labels = {}, {}
    if mode == "day":
        days = {}
        for iid, t in times:
            if t is None:
                assignment[iid] = 0
                continue
            day = _dt.datetime.fromtimestamp(int(t), _dt.timezone.utc).strftime("%Y-%m-%d")
            if day not in days:
                days[day] = 0
            days[day] += 1
        for n, day in enumerate(sorted(days), 1):
            labels[n] = day
        number_of = {day: n for n, day in labels.items()}
        for iid, t in times:
            if t is not None:
                assignment[iid] = number_of[_dt.datetime.fromtimestamp(int(t), _dt.timezone.utc).strftime("%Y-%m-%d")]
        return assignment, labels
    gap = float(gap_hours or 0.0) * 3600.0
    if gap <= 0:
        raise ValueError("the gap must be a positive number of hours")
    timed = sorted((int(t), iid) for iid, t in times if t is not None)
    n = 0
    last = None
    for t, iid in timed:
        if last is None or t - last > gap:
            n += 1
            labels[n] = _dt.datetime.fromtimestamp(t, _dt.timezone.utc).strftime("from %Y-%m-%d %H:%M")
        assignment[iid] = n
        last = t
    for iid, t in times:
        if t is None:
            assignment[iid] = 0
    return assignment, labels


def kmeans(points, k, iterations=50, seed=0):
    """Plain k-means on (n, 2) points: labels 0..k-1 and the centres, started from k-means++ seeds."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    k = max(1, min(int(k), n))
    rng = np.random.default_rng(seed)
    centres = [pts[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min([np.sum((pts - c) ** 2, axis=1) for c in centres], axis=0)
        total = d2.sum()
        centres.append(pts[rng.choice(n, p=d2 / total)] if total > 0 else pts[rng.integers(n)])
    centres = np.array(centres)
    labels = np.zeros(n, dtype=int)
    for _ in range(iterations):
        d2 = np.stack([np.sum((pts - c) ** 2, axis=1) for c in centres], axis=1)
        new = d2.argmin(axis=1)
        if np.array_equal(new, labels) and _ > 0:
            break
        labels = new
        for j in range(k):
            if np.any(labels == j):
                centres[j] = pts[labels == j].mean(axis=0)
    return labels, centres


def suggest_k(points, sample_size=1500, minimum_jump=3.0, seed=0):
    """The number of clusters in well-separated data (the AFIS positions): the sorted pairwise distances of a
    sample fall into the within-cluster ones and the between-cluster ones with a jump between, and the
    clusters are the connected components under a threshold inside the largest jump. 1 when no jump is at
    least `minimum_jump`-fold (one cluster, or a continuum) or for fewer than 4 points."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 4:
        return 1
    if n > sample_size:
        pts = pts[np.random.default_rng(seed).choice(n, sample_size, replace=False)]
        n = len(pts)
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(n, 1)
    d = np.sort(dist[iu])
    d = d[d > 0]
    if d.size < 2:
        return 1
    ratios = d[1:] / d[:-1]
    i = int(np.argmax(ratios))
    if ratios[i] < minimum_jump:
        return 1
    threshold = math.sqrt(d[i] * d[i + 1])
    # connected components under the threshold (union-find)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    close = np.argwhere((dist < threshold) & (dist > 0))
    for a, b in close:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    return len({find(a) for a in range(n)})


def _by_beam_shift(images, k=None):
    with_shift = [r for r in images if r["BEAM_SHIFT_X"] is not None and r["BEAM_SHIFT_Y"] is not None]
    if not with_shift:
        raise ValueError("no image has a beam shift; read the EPU / mdoc metadata first")
    pts = np.array([[r["BEAM_SHIFT_X"], r["BEAM_SHIFT_Y"]] for r in with_shift])
    k = int(k) if k else suggest_k(pts)
    labels, centres = kmeans(pts, k)
    # number the clusters by their centre (x then y) so the numbering is stable across runs
    order = sorted(range(len(centres)), key=lambda j: (round(float(centres[j][0]), 6), round(float(centres[j][1]), 6)))
    number_of = {j: n for n, j in enumerate(order, 1)}
    assignment = {r["IMAGE_ASSET_ID"]: 0 for r in images}
    for r, lab in zip(with_shift, labels.tolist()):
        assignment[r["IMAGE_ASSET_ID"]] = number_of[lab]
    labels_out = {number_of[j]: "beam shift {:.4g}, {:.4g}".format(float(centres[j][0]), float(centres[j][1])) for j in range(len(centres))}
    return assignment, labels_out


def preview(conn, params):
    """A proposed grouping of the images in an image group (`image_group_id`, All Images by default) by
    `method`: {assignment: {image_id: group}, groups: [{group, name, images}], unassigned, points} --
    `points` the (x, y, group) beam shifts for the beam-shift scatter plot when there are any."""
    method = str(params.get("method") or "")
    if method not in METHODS:
        raise ValueError("method must be one of {}".format(", ".join(METHODS)))
    images = _images(conn, params.get("image_group_id"))
    if not images:
        raise ValueError("the image group has no images")
    if method == "image_groups":
        assignment, labels = _by_image_groups(conn, images, params.get("group_ids"))
    elif method == "filename":
        pattern = params.get("pattern")
        preset = params.get("preset")
        if preset and preset in FILENAME_PRESETS and not pattern:
            pattern = FILENAME_PRESETS[preset]["pattern"]
        assignment, labels = _by_filename(images, pattern)
    elif method == "time":
        assignment, labels = _by_time(images, params.get("mode") or "day", params.get("gap_hours") or 4.0)
    else:
        assignment, labels = _by_beam_shift(images, params.get("k"))
    counts = {}
    for g in assignment.values():
        counts[g] = counts.get(g, 0) + 1
    groups = [{"group": g, "name": labels.get(g, "group {}".format(g)), "images": counts.get(g, 0)} for g in sorted(labels)]
    points = [[r["BEAM_SHIFT_X"], r["BEAM_SHIFT_Y"], assignment.get(r["IMAGE_ASSET_ID"], 0)] for r in images
              if r["BEAM_SHIFT_X"] is not None and r["BEAM_SHIFT_Y"] is not None]
    return {"method": method, "assignment": {str(k): v for k, v in assignment.items()}, "groups": groups,
            "unassigned": counts.get(0, 0), "points": points, "image_count": len(images)}


def apply(conn, params):
    """Write a preview's grouping: the images' OPTICS_GROUP (and their movies'), the names in OPTICS_GROUPS.
    With `replace` (default) every image outside the scope goes to group 0 and the names table is rebuilt,
    so the project's groups are exactly these; otherwise the new groups are numbered after the existing
    ones and the other images keep theirs."""
    result = preview(conn, params)
    replace = params.get("replace", True)
    assignment = {int(k): int(v) for k, v in result["assignment"].items()}
    groups = result["groups"]
    with conn:
        offset = 0
        if replace:
            conn.execute("UPDATE IMAGE_ASSETS SET OPTICS_GROUP=0")
            conn.execute("UPDATE MOVIE_ASSETS SET OPTICS_GROUP=0")
            conn.execute("DELETE FROM OPTICS_GROUPS")
        else:
            offset = int(conn.execute("SELECT COALESCE(MAX(GROUP_ID), 0) FROM OPTICS_GROUPS").fetchone()[0])
        for g in groups:
            conn.execute("INSERT OR REPLACE INTO OPTICS_GROUPS(GROUP_ID, NAME, SOURCE) VALUES (?, ?, ?)", (g["group"] + offset, g["name"], result["method"]))
            g["group"] += offset
        conn.executemany("UPDATE IMAGE_ASSETS SET OPTICS_GROUP=? WHERE IMAGE_ASSET_ID=?",
                         [(g + offset if g else 0, iid) for iid, g in assignment.items()])
        conn.executemany("UPDATE MOVIE_ASSETS SET OPTICS_GROUP=? WHERE MOVIE_ASSET_ID=(SELECT PARENT_MOVIE_ID FROM IMAGE_ASSETS WHERE IMAGE_ASSET_ID=?)",
                         [(g + offset if g else 0, iid) for iid, g in assignment.items()])
    return {"method": result["method"], "groups": groups, "unassigned": result["unassigned"], "image_count": result["image_count"]}


def current(conn):
    """The project's optics groups with their image counts, and how many images carry metadata."""
    counts = {r[0]: r[1] for r in conn.execute("SELECT COALESCE(OPTICS_GROUP, 0), COUNT(*) FROM IMAGE_ASSETS GROUP BY COALESCE(OPTICS_GROUP, 0)")}
    names = {r["GROUP_ID"]: r for r in conn.execute("SELECT * FROM OPTICS_GROUPS")}
    ids = sorted(set(counts) | set(names))
    groups = [{"group": g, "name": names[g]["NAME"] if g in names else "group {}".format(g), "source": names[g]["SOURCE"] if g in names else None,
               "images": counts.get(g, 0)} for g in ids if g != 0]
    meta = conn.execute("SELECT COUNT(*), SUM(BEAM_SHIFT_X IS NOT NULL), SUM(ACQUISITION_TIME IS NOT NULL) FROM IMAGE_ASSETS").fetchone()
    return {"groups": groups, "unassigned": counts.get(0, 0), "image_count": int(meta[0] or 0),
            "images_with_beam_shift": int(meta[1] or 0), "images_with_time": int(meta[2] or 0),
            "filename_presets": [{"key": k, "label": v["label"], "pattern": v["pattern"]} for k, v in FILENAME_PRESETS.items()]}
