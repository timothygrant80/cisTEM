"""Blush regularisation of a refinement's reference between rounds.

Kimanius et al. (2024), Nat. Methods 21:1216, "Data-driven regularization
lowers the size barrier of cryo-EM structure determination": a U-Net trained
to denoise unfiltered cryo-EM reconstructions replaces the Wiener filter as
the regulariser of a 3D refinement. This is a port of the parts of RELION's
`relion_blush` package (https://github.com/3dem/relion-blush, MIT License,
Copyright (c) 2023 3dem) that RELION's refinement path runs -- its
`refine3d()` in command_line.py and the helpers in util.py -- re-expressed
on cisTEM's outputs:

  * RELION hands Blush the raw Fourier data and weights of each half map and
    the Python forms the unfiltered reconstruction; here merge3d's two half
    maps (FinalizeSimple: data / weights, unfiltered) are averaged, which is
    the unfiltered reconstruction of the whole data set.
  * The map is resampled to the model's 1.5 A voxel, masked at the mask
    radius, normalised, and the network run on 64-voxel blocks at a stride of
    20 with a weighted stitch (apply_model), exactly as RELION does.
  * The result is masked, resampled back and given RELION's "spectral
    trailing": a soft cut at the shell where the FSC falls below 1/7, so the
    denoised map contributes nothing the data do not support. When that
    shell lies beyond the denoiser's own Nyquist (3 A), the input's higher
    frequencies are mixed back in instead. cisTEM refines every particle
    against one merged reference, so the denoised map is shared by both half
    sets, as cisTEM's reference already is; cisTEM's FSC is computed by
    merge3d from the half maps before any denoising, so it stays an honest
    measure of whether Blush improved the alignments.
  * Alternatively the already Wiener-filtered reference can be denoised, as
    pull request 541's libtorch port does; that skips the trailing cut since
    the input is filtered already (`input_is_filtered=True`).

The model (blush_model.py, RELION's definition verbatim) loads RELION's
published checkpoint, blush_v1.0.ckpt.gz from Zenodo record 10072731, looked
for at $CISTEM_BLUSH_WEIGHTS, then server/data/models/blush_v1.0.ckpt(.gz).
torch is an optional dependency: `availability()` says whether the feature
can run and why not, and the panels show the option accordingly. Inference
runs on CUDA when torch sees a device, else on the CPU with the requested
thread count.
"""

import gzip
import io
import math
import os
import sys
import threading
from pathlib import Path

import numpy as np

MODEL_VOXEL_SIZE = 1.5   # the v1.0 weights' training voxel size (the checkpoint says so too)
BLOCK_SIZE = 64
DEFAULT_STRIDE = 20      # RELION's command-line default
MASK_EDGE_A = 10.0       # the soft edge of the radial mask, as RELION's classification path uses
FSC_THRESHOLD = 1.0 / 7.0

WEIGHTS_CANDIDATES = [
    Path(__file__).parent / "data" / "models" / "blush_v1.0.ckpt.gz",
    Path(__file__).parent / "data" / "models" / "blush_v1.0.ckpt",
]

_lock = threading.Lock()
_models = {}
_availability = None


class BlushUnavailable(RuntimeError):
    pass


def weights_path():
    env = os.environ.get("CISTEM_BLUSH_WEIGHTS")
    if env:
        return Path(env)
    for p in WEIGHTS_CANDIDATES:
        if p.is_file():
            return p
    return WEIGHTS_CANDIDATES[0]


def availability(refresh=False):
    """{available, reason, device, weights, torch}: whether Blush can run here.
    Cached after the first call (importing torch takes a second); `refresh`
    looks again, e.g. after the weights were copied in."""
    global _availability
    with _lock:
        if _availability is not None and not refresh:
            return dict(_availability)
        info = {"available": False, "reason": None, "device": None, "weights": str(weights_path()), "torch": None}
        try:
            import torch  # noqa: F401
            info["torch"] = torch.__version__
            info["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            info["reason"] = "PyTorch is not installed in the server's Python (pip install torch)"
        if info["torch"] and not weights_path().is_file():
            info["reason"] = "the Blush weights were not found at {} (blush_v1.0.ckpt.gz from Zenodo record 10072731)".format(weights_path())
        info["available"] = info["reason"] is None
        _availability = info
        return dict(info)


def load_checkpoint(path=None):
    """RELION's composed checkpoint: {model_state_dict, model_definition, block_size, voxel_size, no_mask}."""
    import torch
    path = Path(path or weights_path())
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as fh:
        data = fh.read()
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)


def load_model(device=None, path=None):
    """The network with RELION's weights, in eval mode on `device`; cached per device."""
    import torch
    from blush_model import BlushModel
    info = availability()
    if not info["available"]:
        raise BlushUnavailable(info["reason"])
    device = device or info["device"]
    key = (device, str(path or weights_path()))
    with _lock:
        if key in _models:
            return _models[key]
        ck = load_checkpoint(path)
        if int(ck.get("block_size", BLOCK_SIZE)) != BLOCK_SIZE or abs(float(ck.get("voxel_size", MODEL_VOXEL_SIZE)) - MODEL_VOXEL_SIZE) > 1e-6:
            raise BlushUnavailable("unexpected Blush checkpoint: block size {} at {} A (expected {} at {} A)".format(
                ck.get("block_size"), ck.get("voxel_size"), BLOCK_SIZE, MODEL_VOXEL_SIZE))
        model = BlushModel()
        model.load_state_dict(ck["model_state_dict"], strict=True)
        model.eval()
        model.to(device)
        _models[key] = model
        return model


# --- the pieces of relion_blush/util.py the refine path uses -------------------------------------------------------

def block_starts(span, block_size, stride):
    """ScanningBlockIterator.get_range(): block start offsets along one axis, the last block pushed to the edge."""
    starts = list(range(0, span - block_size, stride))
    if not starts or starts[-1] < span - block_size:
        starts.append(span - block_size)
    return starts


def make_weight_box(size, margin=10):
    """RELION's stitching weight for one block: a cosine falling to the edge
    inside a zero margin (clipped to 1e-6), so overlapping blocks blend."""
    margin = margin - 1 if margin > 0 else 1
    s = size - margin * 2
    ls = np.linspace(-(s // 2), s // 2, s)
    z, y, x = np.meshgrid(ls, ls, ls, indexing="ij")
    r = np.max([np.abs(x), np.abs(y), np.abs(z)], axis=0)
    r = np.cos(r / np.max(r) * np.pi / 2)
    w = np.zeros((size, size, size), dtype=np.float32)
    w[margin:size - margin, margin:size - margin, margin:size - margin] = r
    return np.clip(w, 1e-6, None).astype(np.float32)


def radial_mask(box_size, radius, edge_width=0.0):
    """RELION's get_radial_mask(): 1 inside `radius` (voxels), a cosine edge of `edge_width` voxels about it, 0 outside."""
    ls = np.linspace(-box_size / 2.0, box_size / 2.0, box_size, dtype=np.float32)
    z, y, x = np.meshgrid(ls, ls, ls, indexing="ij")
    r = np.sqrt(x * x + y * y + z * z)
    scale = np.ones_like(r)
    if edge_width > 0:
        lo, hi = radius - edge_width // 2, radius + edge_width // 2
        scale[r > hi] = 0.0
        band = (r >= lo) & (r <= hi)
        scale[band] = 0.5 + 0.5 * np.cos(np.pi * (r[band] - lo) / edge_width)
    else:
        scale[r > radius] = 0.0
    return scale.astype(np.float32)


def fft(grid):
    return np.fft.rfftn(np.fft.fftshift(grid)).astype(np.complex64)


def ifft(grid_ft):
    return np.fft.ifftshift(np.fft.irfftn(grid_ft)).astype(np.float32)


def rescaled_box_size(box_size, in_voxel, target_voxel):
    """RELION's rescaled_boxsize_from_voxelsize(): the even box that brings `in_voxel` closest to `target_voxel`."""
    out = int(round(box_size * in_voxel / target_voxel))
    if out % 2 != 0:
        vs1 = in_voxel * box_size / (out + 1)
        vs2 = in_voxel * box_size / (out - 1)
        out += 1 if abs(vs1 - target_voxel) < abs(vs2 - target_voxel) else -1
    return out, in_voxel * box_size / out


def rescale_fourier(box, out_size):
    """RELION's rescale_fourier(): crop or zero-pad a half-plane transform about the origin to `out_size` cubed."""
    if out_size % 2 != 0:
        raise ValueError("bad output size {}".format(out_size))
    ibox = np.fft.ifftshift(box, axes=(0, 1))
    obox = np.zeros((out_size, out_size, out_size // 2 + 1), dtype=box.dtype)
    si = np.array(ibox.shape) // 2
    so = np.array(obox.shape) // 2
    if so[0] < si[0]:
        obox = ibox[si[0] - so[0]:si[0] + so[0], si[1] - so[1]:si[1] + so[1], :obox.shape[2]]
    elif so[0] > si[0]:
        obox[so[0] - si[0]:so[0] + si[0], so[1] - si[1]:so[1] + si[1], :ibox.shape[2]] = ibox
    else:
        obox = ibox
    return np.fft.ifftshift(obox, axes=(0, 1))


def resample_fourier(volume, in_voxel, target_voxel):
    """The volume on a grid of (nearly) `target_voxel`, by Fourier crop or pad, with RELION's density scaling; (volume, voxel)."""
    df = fft(volume)
    out_size, out_voxel = rescaled_box_size(volume.shape[0], in_voxel, target_voxel)
    df_out = rescale_fourier(df, out_size) * (out_size / volume.shape[0]) ** 3
    return ifft(df_out), out_voxel


def fourier_shells(shape):
    """RELION's get_fourier_shells() for a half-plane transform of `shape`: the radius of every voxel in shells."""
    z, y, x = shape
    Z, Y, X = np.meshgrid(np.linspace(-z // 2, z // 2 - 1, z), np.linspace(-y // 2, y // 2 - 1, y), np.linspace(0, x - 1, x), indexing="ij")
    return np.fft.ifftshift(np.sqrt(X ** 2 + Y ** 2 + Z ** 2), axes=(0, 1))


def crossover_grid(crossover_index, size, filter_edge_width=3):
    """RELION's get_crossover_grid(): 1 below shell `crossover_index`, a cosine edge of `filter_edge_width` shells, 0 above."""
    r = fourier_shells((size, size, size // 2 + 1))
    half = filter_edge_width // 2
    lo = float(np.clip(crossover_index - half, 0, size / 2.0 + 1))
    hi = float(np.clip(crossover_index + half, 0, size / 2.0 + 1))
    scale = np.zeros_like(r, dtype=np.float32)
    scale[r < lo] = 1.0
    if hi > lo:
        band = (r >= lo) & (r <= hi)
        scale[band] = 0.5 + 0.5 * np.cos(np.pi * (r[band] - lo) / (hi - lo))
    return scale


def fsc_crossing_index(fsc, threshold=FSC_THRESHOLD):
    """RELION's res_from_fsc(fsc) - 1: the last shell before the FSC first falls below `threshold`.
    `fsc` is indexed by shell (shell 0 the origin)."""
    fsc = np.asarray(fsc, dtype=float)
    i = int(np.argmax(fsc < threshold))          # the first shell below the threshold (0 when none is)
    res_index = i - 1 if i > 0 else len(fsc) - 1
    return res_index - 1


def fsc_by_shell(stats, box_size, key="part_fsc"):
    """cisTEM's statistics rows ({shell, fsc, part_fsc, ...}, shells 1..box/2) as an array indexed by shell, shell 0 = 1."""
    out = np.ones(box_size // 2 + 1, dtype=float)
    for row in stats:
        s = int(row.get("shell", 0))
        if 0 < s < out.size:
            out[s] = float(row.get(key, 1.0))
    return out


def local_std(volume_t, size=10):
    """RELION's get_local_std_torch(): the standard deviation under a Gaussian of half-width `size` voxels, separably."""
    import torch
    grid = volume_t.unsqueeze(0).unsqueeze(0).clone()
    grid2 = grid.square()
    ls = torch.linspace(-1.5, 1.5, 2 * size + 1)
    kernel = torch.exp(-ls.square()).to(grid.device)
    kernel = (kernel / kernel.sum())[None, None, :, None, None]
    for _ in range(3):
        grid = torch.nn.functional.conv3d(grid.permute(0, 1, 4, 2, 3), kernel, padding="same")
        grid2 = torch.nn.functional.conv3d(grid2.permute(0, 1, 4, 2, 3), kernel, padding="same")
    return torch.sqrt(torch.clip(grid2 - grid.square(), min=0))[0, 0]


def _standardise(volume, input_mask, block_size, stride):
    """The network's two input channels for a whole volume (standardised
    density and normalised local standard deviation, both masked), the mask and
    the padding RELION applies when the volume is smaller than a block; numpy."""
    import torch
    vol = torch.as_tensor(np.ascontiguousarray(volume, dtype=np.float32))
    mask = torch.as_tensor(np.ascontiguousarray(input_mask, dtype=np.float32))
    shape = list(vol.shape)
    pad = None
    if any(n <= block_size for n in shape):   # the block must fit: pad small volumes
        new_shape = [max(n, block_size + stride) for n in shape]
        si = [n // 2 for n in shape]
        so = [n // 2 for n in new_shape]
        pad = [so[i] - si[i] for i in range(3)]
        v = torch.zeros(new_shape); v[pad[0]:so[0] + si[0], pad[1]:so[1] + si[1], pad[2]:so[2] + si[2]] = vol; vol = v
        m = torch.zeros(new_shape); m[pad[0]:so[0] + si[0], pad[1]:so[1] + si[1], pad[2]:so[2] + si[2]] = mask; mask = m
        shape = new_shape
    std_layer = local_std(vol, 10)
    std_layer = std_layer / std_layer.mean()
    mean, std = float(vol.mean()), float(vol.std())
    vol = (vol - mean) / (std + 1e-8) * mask
    std_layer = std_layer * mask
    inputs = torch.stack([vol, std_layer], 0).numpy()
    return inputs, mask.numpy(), mean, std, pad, shape


def _wanted_blocks(mask, shape, block_size, stride):
    """The block origins RELION processes: every block whose mask mean is at least 0.3."""
    starts = [(z, y, x) for z in block_starts(shape[0], block_size, stride) for y in block_starts(shape[1], block_size, stride) for x in block_starts(shape[2], block_size, stride)]
    return [c for c in starts if mask[c[0]:c[0] + block_size, c[1]:c[1] + block_size, c[2]:c[2] + block_size].mean() >= 0.3]


def _run_blocks(model, inputs, coords, weight, block_size, batch_size, device, infer, count, on_batch=None):
    """The network over `coords` in batches, each block's output blended into
    `infer` and the weights into `count` (torch tensors on `device`).
    `on_batch(n)` after each batch may return False to stop."""
    import torch
    with torch.no_grad():
        for b in range(0, len(coords), batch_size):
            batch = coords[b:b + batch_size]
            blocks = torch.stack([inputs[:, z:z + block_size, y:y + block_size, x:x + block_size] for (z, y, x) in batch], 0)
            out, _mask_logit = model(blocks[:, 0], blocks[:, 1])
            for i, (z, y, x) in enumerate(batch):
                infer[z:z + block_size, y:y + block_size, x:x + block_size] += out[i] * weight
                count[z:z + block_size, y:y + block_size, x:x + block_size] += weight
            if on_batch is not None and on_batch(len(batch)) is False:
                return False
    return True


def _attach_shared(name):
    """Attach to a shared-memory block another process created, without this
    process's resource tracker unlinking it at exit (Python's issue 38119:
    SharedMemory(name=) registers the block as this process's own)."""
    from multiprocessing import shared_memory, resource_tracker
    shm = shared_memory.SharedMemory(name=name)
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:  # noqa: BLE001
        pass
    return shm


def _worker_main(spec_path):
    """`python3 blush.py --worker <spec.json>`: one process of the CPU pool.
    Its share of the blocks goes into its own shared-memory sums; progress is
    one line per batch on stdout, then `done`, or `error <message>`."""
    import json
    import torch
    with open(spec_path) as fh:
        spec = json.load(fh)
    shms = []
    try:
        if spec["threads"] > 0:
            torch.set_num_threads(int(spec["threads"]))
        shape = tuple(spec["shape"])
        shm = _attach_shared(spec["inputs"]); shms.append(shm)
        inputs = torch.from_numpy(np.ndarray((2,) + shape, dtype=np.float32, buffer=shm.buf))
        out_shm = _attach_shared(spec["out"]); shms.append(out_shm)
        count_shm = _attach_shared(spec["count"]); shms.append(count_shm)
        infer = torch.from_numpy(np.ndarray(shape, dtype=np.float32, buffer=out_shm.buf))
        count = torch.from_numpy(np.ndarray(shape, dtype=np.float32, buffer=count_shm.buf))
        infer.zero_(); count.zero_()
        model = load_model("cpu", spec["weights"])
        weight = torch.as_tensor(make_weight_box(spec["block_size"], 10))
        coords = [tuple(c) for c in spec["coords"]]

        def on_batch(n):
            sys.stdout.write("progress {}\n".format(n)); sys.stdout.flush()
            return True

        _run_blocks(model, inputs, coords, weight, spec["block_size"], spec["batch_size"], "cpu", infer, count, on_batch)
        sys.stdout.write("done\n"); sys.stdout.flush()
    except Exception as exc:  # noqa: BLE001
        sys.stdout.write("error {}: {}\n".format(type(exc).__name__, exc)); sys.stdout.flush()
    finally:
        for m in shms:
            try:
                m.close()
            except Exception:  # noqa: BLE001
                pass


def _apply_model_processes(inputs, coords, shape, block_size, batch_size, processes, threads, weights, progress):
    """The blocks shared out over `processes` worker processes (`python3
    blush.py --worker`, each with its own copy of the model and `threads`
    PyTorch threads), their weighted sums added up here. One PyTorch process
    cannot use a large machine's cores on one 64-voxel block, so this is how a
    128-core server is kept busy. Plain subprocesses rather than the
    multiprocessing module's spawn, which re-imports the launching program's
    main module in every worker (the Flask server, here)."""
    import json
    import queue as queue_module
    import subprocess
    import tempfile
    from multiprocessing import shared_memory
    processes = max(1, min(int(processes), max(1, math.ceil(len(coords) / batch_size))))
    chunks = [coords[i::processes] for i in range(processes)]
    n_vox = int(np.prod(shape))
    shm = shared_memory.SharedMemory(create=True, size=inputs.nbytes)
    np.ndarray(inputs.shape, dtype=np.float32, buffer=shm.buf)[...] = inputs
    outs, counts, procs, readers = [], [], [], []
    messages = queue_module.Queue()
    tmp = tempfile.mkdtemp(prefix="blush_")
    infer = np.zeros(shape, dtype=np.float32)
    count = np.zeros(shape, dtype=np.float32)

    def read(index, proc):
        for line in proc.stdout:
            messages.put((index, line.decode("utf-8", "replace").strip()))
        messages.put((index, "exit {}".format(proc.wait())))

    try:
        for i, chunk in enumerate(chunks):
            o = shared_memory.SharedMemory(create=True, size=n_vox * 4)
            c = shared_memory.SharedMemory(create=True, size=n_vox * 4)
            outs.append(o); counts.append(c)
            spec = os.path.join(tmp, "worker_{}.json".format(i))
            with open(spec, "w") as fh:
                json.dump({"inputs": shm.name, "out": o.name, "count": c.name, "shape": list(shape), "coords": [list(map(int, cc)) for cc in chunk],
                           "block_size": block_size, "batch_size": batch_size, "threads": int(threads), "weights": str(weights)}, fh)
            proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker", spec], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    cwd=os.path.dirname(os.path.abspath(__file__)))
            procs.append(proc)
            t = threading.Thread(target=read, args=(i, proc), daemon=True, name="blush-reader-{}".format(i)); t.start(); readers.append(t)
        total, done, finished, error, cancelled = len(coords), 0, set(), None, False
        while len(finished) < len(procs) and error is None:
            try:
                index, line = messages.get(timeout=1.0)
            except queue_module.Empty:
                dead = [i for i, pr in enumerate(procs) if pr.poll() is not None and i not in finished and not readers[i].is_alive()]
                if dead:
                    error = "worker {} exited with code {}".format(dead[0], procs[dead[0]].returncode)
                continue
            if line.startswith("progress "):
                done += int(line.split()[1])
                if progress is not None and progress(done, total) is False:
                    cancelled = True
                    break
            elif line == "done":
                finished.add(index)
            elif line.startswith("error "):
                error = "worker {}: {}".format(index, line[6:])
            elif line.startswith("exit ") and index not in finished:
                err = procs[index].stderr.read().decode("utf-8", "replace").strip() if procs[index].stderr else ""
                error = "worker {} exited with code {}{}".format(index, line.split()[1], ": " + err[-500:] if err else "")
        if cancelled:
            raise BlushCancelled()
        if error:
            raise RuntimeError("Blush: " + error)
        for o, c in zip(outs, counts):
            infer += np.ndarray(shape, dtype=np.float32, buffer=o.buf)
            count += np.ndarray(shape, dtype=np.float32, buffer=c.buf)
        return infer, count
    finally:
        for pr in procs:
            if pr.poll() is None:
                pr.terminate()
        for pr in procs:
            try:
                pr.wait(timeout=10)
            except Exception:  # noqa: BLE001
                pr.kill()
        for m in [shm] + outs + counts:
            try:
                m.close(); m.unlink()
            except Exception:  # noqa: BLE001
                pass
        try:
            for f in os.listdir(tmp):
                os.remove(os.path.join(tmp, f))
            os.rmdir(tmp)
        except OSError:
            pass


def apply_model(model, volume, input_mask, device, stride=DEFAULT_STRIDE, block_size=BLOCK_SIZE, batch_size=1, progress=None,
                processes=1, threads=0, weights=None):
    """RELION's apply_model(): the network over every block whose mask mean is
    at least 0.3, blended with the weight box; the volume is standardised on
    the way in and restored on the way out. `progress(done, total)` is called
    after each batch and may return False to stop (BlushCancelled). On the CPU
    with `processes` > 1 the blocks are shared out over that many spawned
    processes of `threads` PyTorch threads each (_apply_model_processes); a
    GPU takes every block itself."""
    import torch
    inputs, mask, mean, std, pad, shape = _standardise(volume, input_mask, block_size, stride)
    coords = _wanted_blocks(mask, shape, block_size, stride)
    if device == "cpu" and processes > 1 and len(coords) > 1:
        infer_np, count_np = _apply_model_processes(inputs, coords, shape, block_size, batch_size, processes, threads, weights or weights_path(), progress)
        infer = torch.from_numpy(infer_np)
        count = torch.from_numpy(count_np)
        mask_t = torch.from_numpy(mask)
    else:
        inputs_t = torch.from_numpy(inputs).to(device)
        mask_t = torch.from_numpy(mask).to(device)
        weight = torch.as_tensor(make_weight_box(block_size, 10)).to(device)
        infer = torch.zeros(shape, device=device)
        count = torch.zeros(shape, device=device)
        total = len(coords)
        done = [0]

        def on_batch(n):
            done[0] += n
            return progress(done[0], total) if progress is not None else True

        if _run_blocks(model, inputs_t, coords, weight, block_size, batch_size, device, infer, count, on_batch) is False:
            raise BlushCancelled()
    covered = count > 0
    infer[covered] /= count[covered]
    infer[count < 1e-1] = 0.0
    infer = infer * mask_t * (std + 1e-8) + mean
    if pad is not None:
        si = [n // 2 for n in volume.shape]
        so = [n // 2 for n in shape]
        infer = infer[pad[0]:so[0] + si[0], pad[1]:so[1] + si[1], pad[2]:so[2] + si[2]]
    return infer.cpu().numpy().astype(np.float32)


class BlushCancelled(RuntimeError):
    pass


def denoise(volume, pixel_size, mask_radius_a, fsc=None, input_is_filtered=False, stride=DEFAULT_STRIDE, batch_size=1,
            threads=0, processes=1, device=None, model=None, progress=None):
    """RELION's refine3d() on one cisTEM map. `volume` (z, y, x) at `pixel_size`
    A; `mask_radius_a` the refinement's mask radius; `fsc` the round's FSC
    indexed by shell (fsc_by_shell), None for no spectral trailing. Returns the
    denoised map at the input's size and pixel size.

    The radial mask is `mask_radius_a` plus half a 10 A cosine edge, clipped to
    the box (RELION's classification path derives it from the particle
    diameter the same way; its refinement path's formula is dimensionally odd
    and in practice clips to the box). With `input_is_filtered` the trailing
    cut is skipped, as the Wiener filter has already applied one; the mixing
    of the input's frequencies beyond the denoiser's 3 A Nyquist applies in
    both modes, since the network cannot produce them. On the CPU, `processes`
    > 1 shares the blocks over that many spawned processes of `threads`
    PyTorch threads each (0: PyTorch's default); on a GPU both are ignored."""
    import torch
    device = device or availability()["device"] or "cpu"
    pooled = device == "cpu" and processes > 1 and model is None   # the blocks run in worker processes
    if threads and threads > 0 and not pooled:
        torch.set_num_threads(int(threads))
    if not pooled:
        model = model or load_model(device)
    volume = np.asarray(volume, dtype=np.float32)
    n = volume.shape[0]
    denoise_input, voxel_nv = resample_fourier(volume, pixel_size, MODEL_VOXEL_SIZE)
    n_nv = denoise_input.shape[0]

    def mask_for(size, voxel):
        edge = MASK_EDGE_A / voxel
        radius = min(mask_radius_a / voxel + edge / 2.0, (size - edge) / 2.0 + 1)
        return radial_mask(size, radius, edge_width=edge)

    mask_nv = mask_for(n_nv, voxel_nv)
    mask_orig = mask_for(n, pixel_size)
    denoised_nv = apply_model(model, denoise_input, mask_nv, device, stride=stride, block_size=BLOCK_SIZE, batch_size=batch_size, progress=progress,
                              processes=processes if pooled else 1, threads=threads)
    denoised_nv *= mask_nv
    denoised_df_nv = fft(denoised_nv)
    denoised_df = rescale_fourier(denoised_df_nv, n) * (n / n_nv) ** 3
    denoiser_limit = denoised_df_nv.shape[-1] - 3          # the last shell the 1.5 A grid can carry, less the edge
    if fsc is not None:
        max_index = fsc_crossing_index(fsc)
    else:
        max_index = denoiser_limit + 1 if input_is_filtered else denoiser_limit
    if max_index > denoiser_limit:
        # The data reach beyond the denoiser's Nyquist: keep its low frequencies and the input's high ones.
        grid = crossover_grid(denoiser_limit, n, 3)
        out_df = denoised_df * grid + fft(volume * mask_orig) * (1.0 - grid)
    elif input_is_filtered:
        out_df = denoised_df
    else:
        out_df = denoised_df * crossover_grid(max_index, n, 3)
    return ifft(out_df)


def denoise_file(input_paths, output_path, pixel_size, mask_radius_a, fsc_stats=None, input_is_filtered=False, **kwargs):
    """Read one map, or average several (the half maps) into the unfiltered
    reconstruction, denoise it and write `output_path`; returns the output."""
    import volumes
    acc = None
    ps = None
    for p in input_paths:
        vol, file_ps = volumes.read_mrc_volume(p)
        ps = ps or file_ps or pixel_size
        acc = vol if acc is None else acc + vol
    vol = acc / float(len(input_paths))
    ps = pixel_size or ps
    fsc = fsc_by_shell(fsc_stats, vol.shape[0]) if fsc_stats else None
    out = denoise(vol, ps, mask_radius_a, fsc=fsc, input_is_filtered=input_is_filtered, **kwargs)
    volumes.write_mrc_volume(output_path, out, ps)
    return out


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        _worker_main(sys.argv[2])
    else:
        sys.stderr.write("usage: blush.py --worker <spec.json>   (a worker of the CPU pool; the module is otherwise imported)\n")
        sys.exit(2)
