#!/usr/bin/env python3
"""Time refine2d and refine3d over thread counts, for two builds, on synthetic data made here.

    benchmark_refine.py --new /path/to/new/bin [--old /path/to/old/bin] [--threads 1,8,32,128] [--work DIR]
                        [--mkl /opt/intel/oneapi/setvars.sh] [--box 256] [--classes 50] [--seed 12345]

Makes a synthetic volume, a particle stack of noisy CTF-applied projections (project3d from the NEW build,
whose star-file path works) and 2D starting class averages, then for every build and thread count runs one
refine2d round (box/4 pixels at an 8 A limit), one refine3d local refinement at 3 A and one global search at
8 A, with the particle count scaled by the thread count so each run takes seconds rather than minutes.
Prints a table and writes results.csv in the work directory. Needs numpy; nothing else beyond the binaries.
The old build may not know cisTEM_RANDOM_SEED, so its random starts differ run to run: the numbers are
timings, not an output comparison (that was done bit for bit at a fixed seed before the change)."""
import argparse
import csv
import os
import struct
import subprocess
import sys
import time

import numpy as np


def write_mrc(path, data, pixel_size):
    """data: (nz, ny, nx) float32 -> MRC mode 2 with the pixel size in the cell."""
    data = np.ascontiguousarray(data, dtype="<f4")
    nz, ny, nx = data.shape
    header = bytearray(1024)
    struct.pack_into("<iiii", header, 0, nx, ny, nz, 2)
    struct.pack_into("<iii", header, 28, nx, ny, nz)
    struct.pack_into("<fff", header, 40, nx * pixel_size, ny * pixel_size, nz * pixel_size)
    struct.pack_into("<fff", header, 52, 90.0, 90.0, 90.0)
    struct.pack_into("<iii", header, 64, 1, 2, 3)
    struct.pack_into("<fff", header, 76, float(data.min()), float(data.max()), float(data.mean()))
    header[208:212] = b"MRC "
    header[212:216] = b"\x44\x44\x00\x00"
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(data.tobytes())


STAR_COLUMNS = [("_cisTEMPositionInStack", "{:d}"), ("_cisTEMImageActivity", "{:d}"), ("_cisTEMAnglePsi", "{:.2f}"), ("_cisTEMAngleTheta", "{:.2f}"), ("_cisTEMAnglePhi", "{:.2f}"),
                ("_cisTEMXShift", "{:.2f}"), ("_cisTEMYShift", "{:.2f}"), ("_cisTEMDefocus1", "{:.1f}"), ("_cisTEMDefocus2", "{:.1f}"), ("_cisTEMDefocusAngle", "{:.2f}"),
                ("_cisTEMPhaseShift", "{:.2f}"), ("_cisTEMOccupancy", "{:.2f}"), ("_cisTEMLogP", "{:d}"), ("_cisTEMSigma", "{:.4f}"), ("_cisTEMScore", "{:.2f}"), ("_cisTEMScoreChange", "{:.2f}"),
                ("_cisTEMPixelSize", "{:.5f}"), ("_cisTEMMicroscopeVoltagekV", "{:.2f}"), ("_cisTEMMicroscopeCsMM", "{:.2f}"), ("_cisTEMAmplitudeContrast", "{:.4f}"),
                ("_cisTEMBeamTiltX", "{:.3f}"), ("_cisTEMBeamTiltY", "{:.3f}"), ("_cisTEMImageShiftX", "{:.3f}"), ("_cisTEMImageShiftY", "{:.3f}"), ("_cisTEMAssignedSubset", "{:d}")]


def write_star(path, rows):
    with open(path, "w") as fh:
        fh.write("# synthetic particles for benchmark_refine.py\n\ndata_\n\nloop_\n")
        for i, (label, _fmt) in enumerate(STAR_COLUMNS):
            fh.write("{} #{}\n".format(label, i + 1))
        for r in rows:
            fh.write(" ".join(fmt.format(v) for (_label, fmt), v in zip(STAR_COLUMNS, r)) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", required=True, help="directory with the new refine2d, refine3d and project3d")
    ap.add_argument("--old", help="directory with the old refine2d and refine3d (optional)")
    ap.add_argument("--threads", default="1,8,32,128")
    ap.add_argument("--work", default=os.path.abspath("benchmark_refine_work"))
    ap.add_argument("--mkl", default="/opt/intel/oneapi/setvars.sh", help="sourced before each program; '' to skip")
    ap.add_argument("--box", type=int, default=256)
    ap.add_argument("--classes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--per-thread-2d", type=int, default=100, help="refine2d particles per thread (minimum 400)")
    ap.add_argument("--per-thread-3d", type=int, default=100, help="refine3d local particles per thread (minimum 200)")
    ap.add_argument("--per-thread-global", type=int, default=10, help="refine3d global-search particles per thread (minimum 10)")
    args = ap.parse_args()
    args.new = os.path.abspath(args.new)
    if args.old:
        args.old = os.path.abspath(args.old)
    threads = [int(t) for t in args.threads.split(",")]
    W = args.work
    os.makedirs(W, exist_ok=True)
    box, ps = args.box, 1.0
    n2 = [max(400, args.per_thread_2d * t) for t in threads]
    n3 = [max(200, args.per_thread_3d * t) for t in threads]
    ng = [max(10, args.per_thread_global * t) for t in threads]
    n_total = max(n2 + n3 + ng)

    def run(bin_dir, prog, answers, log, threads_env=None):
        env = dict(os.environ, cisTEM_RANDOM_SEED=str(args.seed))
        if threads_env:
            env["OMP_NUM_THREADS"] = str(threads_env)
        prefix = "source {} > /dev/null 2>&1; ".format(args.mkl) if args.mkl else ""
        t0 = time.time()
        p = subprocess.run(["bash", "-c", prefix + "cd {}; exec {}".format(W, os.path.join(bin_dir, prog))], input="\n".join(str(a) for a in answers) + "\n",
                           capture_output=True, text=True, env=env)
        dt = time.time() - t0
        with open(os.path.join(W, log), "w") as fh:
            fh.write(p.stdout + p.stderr)
        if p.returncode != 0:
            print("{} failed (exit {}), see {}".format(prog, p.returncode, os.path.join(W, log)))
            print(p.stdout[-600:])
            sys.exit(1)
        return dt

    stack = os.path.join(W, "stack.mrc")
    star = os.path.join(W, "particles.star")
    vol = os.path.join(W, "volume.mrc")
    classes0 = os.path.join(W, "classes0.mrc")
    if not os.path.exists(classes0):
        print("making {} particles at box {} ...".format(n_total, box))
        rng = np.random.default_rng(args.seed)
        c = box // 2
        zz, yy, xx = np.mgrid[:box, :box, :box].astype(np.float32)
        v = np.zeros((box, box, box), np.float32)
        for _ in range(40):
            p = rng.normal(0, box / 10, 3)
            s = rng.uniform(box / 64, box / 28)
            v += np.exp(-((zz - c - p[0]) ** 2 + (yy - c - p[1]) ** 2 + (xx - c - p[2]) ** 2) / (2 * s * s))
        write_mrc(vol, v, ps)
        rows = []
        for i in range(n_total):
            d1 = 15000 + rng.uniform(-5000, 5000)
            rows.append((i + 1, 1, rng.uniform(0, 360), float(np.degrees(np.arccos(rng.uniform(-1, 1)))), rng.uniform(0, 360), 0.0, 0.0, d1, d1 - 300, rng.uniform(0, 180),
                         0.0, 100.0, 0, 1.0, 20.0, 0.0, ps, 300.0, 2.7, 0.07, 0.0, 0.0, 0.0, 0.0, 1 + i % 2))
        write_star(star, rows)
        run(args.new, "project3d", [vol, "Yes", star, 1, n_total, "Yes", "No", stack, ps, box * 0.4, 0.05, 1.0, "C1", "No", "Yes", max(threads)], "project3d.log")
        common2d = [ps, box * 0.4, 300.0, 8.0, 0.0, 30.0, 1.0, 2.0, "Yes", "No", "No", "No", "Yes", "No", "/dev/null", max(threads)]
        run(args.new, "refine2d", [stack, star, "/dev/null", os.path.join(W, "start.star"), classes0, args.classes, 1, n2[0], 1.0] + common2d, "refine2d_start.log")
        print("data ready")

    builds = [("new", args.new)] + ([("old", args.old)] if args.old else [])
    results = []
    print("{:5s} {:>7s} {:>8s} {:>9s} {:>9s} {:>11s}".format("build", "threads", "program", "particles", "seconds", "particles/s"))
    for t_i, t in enumerate(threads):
        for name, bin_dir in builds:
            tag = "{}_{}".format(name, t)
            common2d = [ps, box * 0.4, 300.0, 8.0, 0.0, 30.0, 1.0, 2.0, "Yes", "No", "No", "No", "Yes", "No", "/dev/null", t]
            dt = run(bin_dir, "refine2d", [stack, star, classes0, os.path.join(W, "r2d_{}.star".format(tag)), os.path.join(W, "r2d_{}.mrc".format(tag)), 0, 1, n2[t_i], 1.0] + common2d, "refine2d_{}.log".format(tag))
            results.append((name, t, "refine2d", n2[t_i], dt))
            print("{:5s} {:7d} {:>8s} {:9d} {:9.1f} {:11.1f}".format(name, t, "refine2d", n2[t_i], dt, n2[t_i] / dt), flush=True)
            for prog_label, n, glob, hires, smask in (("3d-local", n3[t_i], False, 3.0, 0.0), ("3d-global", ng[t_i], True, 8.0, box * 0.4)):
                answers = [stack, star, vol, "/dev/null", "No", "/dev/null", os.path.join(W, "r3d_{}_{}.star".format(prog_label, tag)), "/dev/null", "C1", 1, n, 1.0, ps, 300.0,
                           0.0, box * 0.4, 300.0, hires, 0.0, hires, smask, 8.0, 0.0, 20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 500.0, 50.0, 1.0,
                           "Yes" if glob else "No", "No" if glob else "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "No", "No", "No", "Yes", "No", "No", "Yes", "Yes", t]
                dt = run(bin_dir, "refine3d", answers, "refine3d_{}_{}.log".format(prog_label, tag))
                results.append((name, t, prog_label, n, dt))
                print("{:5s} {:7d} {:>8s} {:9d} {:9.1f} {:11.1f}".format(name, t, prog_label, n, dt, n / dt), flush=True)
    with open(os.path.join(W, "results.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["build", "threads", "program", "particles", "seconds", "particles_per_second"])
        for r in results:
            w.writerow(list(r) + [r[3] / r[4]])
    # scaling and new/old summary
    print("\nparticles per second relative to 1 thread (ideal = thread count), and new/old speed at each thread count:")
    by = {(b, t, p): n / dt for b, t, p, n, dt in results}
    for p in ("refine2d", "3d-local", "3d-global"):
        line = "{:>9s}:".format(p)
        for t in threads:
            parts = []
            for name, _ in builds:
                base = by.get((name, threads[0], p))
                rate = by.get((name, t, p))
                parts.append("{} x{:.1f}".format(name, rate / base if base and rate else float("nan")))
            if args.old and (("old", t, p) in by):
                parts.append("new/old {:.2f}".format(by[("new", t, p)] / by[("old", t, p)]))
            line += "  [{} thr: {}]".format(t, ", ".join(parts))
        print(line)
    print("results in", os.path.join(W, "results.csv"))


if __name__ == "__main__":
    main()
