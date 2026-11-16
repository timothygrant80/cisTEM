"""Higher-order aberrations: RELION 3.1's Zernike model (Zivanov, Nakane &
Scheres, IUCrJ 7:253, 2020) fitted on the server from what refine_ctf
accumulates (web/CLAUDE.md, "Higher-order aberrations").

The basis, its ordering and the sign conventions mirror src/core/zernike.cpp
and CTF::SetZernikeCoefficients in the cisTEM tree, so a coefficient list here
is what the programs read from an aberration file (src/core/aberration_table.h)
and what RELION writes as rlnOddZernike / rlnEvenZernike: evaluated at the
spatial frequency in 1/A, the odd terms the phase RELION puts on the
projection, the even terms what RELION adds to its CTF argument (cisTEM
subtracts them from its own, which is the negative of RELION's).

refine_ctf writes the aberration sums (its argument 32, in local mode and
from its master alike) as an MRC stack of seven (ny, nx/2+1) real images, one
value per pixel of cisTEM's Fourier half plane:
  - sections 1-5, the even-aberration sums: sum a^2|V|^2, ab|V|^2, b^2|V|^2,
    a Re(X V*), b Re(X V*) over the particles, with a = -sin(gamma),
    b = cos(gamma) for the CTF each particle was refined with (its current
    even coefficients included), so the phase offset delta they determine per
    pixel is the *residual* to add to the current even set;
  - sections 6-7, the real and imaginary parts of the phase-difference sum S
    (the normalised sum of CTF-applied projection x conj(image), what the
    phase-difference image holds before its back-transform): its phase is
    *minus* the total odd aberration in RELION's convention (RELION's phase
    is the image's relative to the projection; Image::ConjugateMultiplyPixelWise
    is a * conj(b) with the projection as a), the projections carrying no odd
    phase. fit_odd takes the conjugate first, so its result is RELION's set.

Each fit is a weighted linear least squares on the per-pixel phase estimates
followed by damped Newton steps on the exact cost (RELION polishes with
Nelder-Mead instead); both take well under a second for a 512 box, which is
why the fit lives here rather than in a job of its own. The weights (|S| for
the odd fit, the curvature of the per-pixel cost for the even one) are
normalised ring by ring: the raw ones fall by orders of magnitude with
resolution and would let the lowest ring, where masking biases the phases,
decide everything; the band is the refinement's low- and high-resolution
limits. A second pass then weights each ring by the coherence of its
residual to the first fit (the resultant length R of the unit phasors,
R^2/(1-R^2) capped), so rings beyond the data's resolution drop out on
their own instead of diluting the fit at equal weight. On a small disc the m = 0 even terms (Z0^0, Z2^0, Z4^0: 1, rho^2,
rho^4) are nearly collinear, so their coefficients trade off against each
other while the phase surface they make is well determined; judge a fit by
its fitted_phase, and read the Cs error and four-fold astigmatism from it.
"""
import math

import numpy as np

import volumes


def wavelength_a(voltage_kv):
    """ReturnWavelenthInAngstroms()."""
    v = 1000.0 * voltage_kv
    return 12.2639 / math.sqrt(v + 0.97845e-6 * v * v)


# ---------------------------------------------------------------- the basis

def number_of_odd_coefficients(order):
    l = (order - 1) // 2 + 1
    return l * l + l


def number_of_even_coefficients(order):
    l = order // 2
    return (l + 1) * (l + 1)


def odd_index_to_mn(index):
    k = int((math.sqrt(1 + 4 * index) - 1.0) / 2.0)
    i0 = k * k + k
    n = 2 * k + 1
    return 2 * (index - i0) - n, n


def even_index_to_mn(index):
    k = int(math.isqrt(index))
    return 2 * (index - k * k - k), 2 * k


def radial_polynomial(m, n, rho):
    """R_n^m(rho), m >= 0, unnormalised as in RELION."""
    rho = np.asarray(rho, dtype=np.float64)
    out = np.zeros_like(rho)
    if (n - m) % 2 == 1:
        return out
    for k in range((n - m) // 2 + 1):
        coefficient = ((-1) ** k) * math.factorial(n - k) / (
            math.factorial(k) * math.factorial((n + m) // 2 - k) * math.factorial((n - m) // 2 - k))
        out += coefficient * rho ** (n - 2 * k)
    return out


def zernike(m, n, rho, phi):
    """Z_n^m: R cos(m phi) for m >= 0, R sin(|m| phi) for m < 0."""
    if m >= 0:
        return radial_polynomial(m, n, rho) * np.cos(m * np.asarray(phi, dtype=np.float64))
    return radial_polynomial(-m, n, rho) * np.sin(-m * np.asarray(phi, dtype=np.float64))


def basis(order, odd, rho, phi):
    """The odd (orders 1, 3, ..) or even (0, 2, ..) polynomials up to `order`,
    stacked as (K, *rho.shape)."""
    rho = np.asarray(rho, dtype=np.float64)
    count = number_of_odd_coefficients(order) if odd else number_of_even_coefficients(order)
    out = np.empty((count,) + rho.shape)
    for i in range(count):
        m, n = odd_index_to_mn(i) if odd else even_index_to_mn(i)
        out[i] = zernike(m, n, rho, phi)
    return out


def half_plane_frequencies(nx, ny, pixel_size):
    """rho (1/A) and azimuth of every pixel of cisTEM's Fourier half plane,
    (ny rows, nx/2 + 1 columns): column i is x frequency i/(nx p), row j the
    y frequency j/(ny p) for j < ny/2 and (j - ny)/(ny p) from ny/2 on, which
    is np.fft.rfft2's layout."""
    fx = np.arange(nx // 2 + 1) / (nx * pixel_size)
    fy = np.fft.fftfreq(ny, d=pixel_size)
    FX, FY = np.meshgrid(fx, fy)
    rho = np.hypot(FX, FY)
    phi = np.arctan2(FY, FX)
    phi[(FX == 0) & (FY == 0)] = 0.0
    return rho, phi


# ------------------------------------------- beam tilt <-> odd coefficients

def odd_coefficients_for_beam_tilt(tilt_x_mrad, tilt_y_mrad, shift_x_a, shift_y_a, cs_mm, voltage_kv, order=3):
    """The odd coefficients equivalent to cisTEM's beam tilt (mrad) and
    particle shift (A): CTF::OddZernikeCoefficientsForBeamTilt, which is
    RELION's TiltHelper::insertTilt."""
    scale = 2.0 * math.pi * cs_mm * 1.0e7 * wavelength_a(voltage_kv) ** 2
    c = np.zeros(number_of_odd_coefficients(order))
    c[4] = -scale * tilt_x_mrad * 1.0e-3 / 3.0
    c[3] = -scale * tilt_y_mrad * 1.0e-3 / 3.0
    c[1] = 2.0 * c[4] + 2.0 * math.pi * shift_x_a
    c[0] = 2.0 * c[3] + 2.0 * math.pi * shift_y_a
    return c


def beam_tilt_from_odd(coefficients, cs_mm, voltage_kv):
    """The beam tilt (mrad) and particle shift (A) an odd set contains, and the
    set with them taken out (trefoil and higher): TiltHelper::extractTilt."""
    c = np.array(coefficients, dtype=np.float64)
    if c.size < 6:
        c = np.concatenate([c, np.zeros(6 - c.size)])
    scale = 2.0 * math.pi * cs_mm * 1.0e7 * wavelength_a(voltage_kv) ** 2
    tilt_x = -3.0 * c[4] / scale * 1.0e3
    tilt_y = -3.0 * c[3] / scale * 1.0e3
    shift_x = (c[1] - 2.0 * c[4]) / (2.0 * math.pi)
    shift_y = (c[0] - 2.0 * c[3]) / (2.0 * math.pi)
    rest = c.copy()
    rest[[0, 1, 3, 4]] = 0.0
    return tilt_x, tilt_y, shift_x, shift_y, rest


# ------------------------------------------------------------------ fitting

def resolution_mask(rho, resolution_limit_a=None, low_resolution_limit_a=None):
    mask = rho > 0.0
    if resolution_limit_a:
        mask &= rho <= 1.0 / float(resolution_limit_a)
    if low_resolution_limit_a:
        mask &= rho >= 1.0 / float(low_resolution_limit_a)
    return mask


RING_WIDTH = 0.01   # 1/A


def ring_normalised(weight, rho, mask):
    """The weights scaled to a mean of one in every ring of RING_WIDTH inside
    the mask (zero outside it), so each ring counts the same."""
    out = np.zeros_like(weight, dtype=np.float64)
    inside = mask & (weight > 0)
    if not inside.any():
        return out
    ring = np.floor(rho / RING_WIDTH).astype(int)
    for r in np.unique(ring[inside]):
        pixels = inside & (ring == r)
        mean = weight[pixels].mean()
        if mean > 0:
            out[pixels] = weight[pixels] / mean
    return out


RIDGE = 1.0e-6


def _ridge(matrix, ridge=RIDGE):
    """The Tikhonov term for a normal matrix: a fraction of its own diagonal,
    so every coefficient is shrunk by the same relative amount (a uniform
    ridge would flatten the weakly determined high-order terms, whose basis
    columns are tiny on a small disc, by half). On a small disc the m = 0
    even terms (1, rho^2, rho^4 ...) and the odd orders 1 and 3 (rho, rho^3)
    are nearly collinear, so without it the coefficients run away along the
    degenerate direction while the phase surface stays the same; the ridge
    picks the smallest set."""
    return ridge * np.diag(np.diag(matrix))


def _weighted_lstsq(Z, target, weight):
    K = Z.shape[0]
    w = np.maximum(weight, 0.0).ravel()
    keep = w > 0.0
    A = Z.reshape(K, -1).T[keep]
    b = target.ravel()[keep]
    normal = A.T @ (A * w[keep, None])
    return np.linalg.solve(normal + _ridge(normal), A.T @ (b * w[keep]))


def _wrap(phase):
    return (phase + math.pi) % (2.0 * math.pi) - math.pi


STAGES = (0.5, 0.75, 1.0)
COHERENCE_WEIGHT_CAP = 100.0


def ring_coherence(residual_phase, rho, mask):
    """The resultant length R of the unit phasors of a residual phase, ring by
    ring (RING_WIDTH): 1 for a perfectly fitted ring, about 0 for noise. Returns
    (per-pixel map of the ring's R, {ring index: R})."""
    out = np.zeros_like(rho, dtype=np.float64)
    rings = {}
    inside = mask & np.isfinite(residual_phase)
    ring = np.floor(rho / RING_WIDTH).astype(int)
    for r in np.unique(ring[inside]):
        pixels = inside & (ring == r)
        R = float(np.abs(np.exp(1j * residual_phase[pixels]).mean()))
        rings[int(r)] = R
        out[pixels] = R
    return out, rings


def coherence_weight(R):
    """Inverse variance of a ring's phase from its resultant length, R^2 / (1 - R^2),
    capped: a ring beyond the data's resolution gets next to nothing."""
    R = np.clip(R, 0.0, 1.0)
    return np.minimum(R * R / np.maximum(1.0 - R * R, 1.0e-12), COHERENCE_WEIGHT_CAP)


def _progressive_fit(Z, measured, weight, mask, rho, evaluate_within, constant_index=None):
    """Both fits share this: the per-pixel phase estimates are wrapped, so the
    fit starts on the inner half of the band, where the phase is smallest,
    and grows outwards in stages, at each one unwrapping the measured phase
    around the current model (which extrapolates to within pi of the truth
    in the new ring) before a weighted linear fit and Newton steps on the
    exact cost over the stage's pixels. With `constant_index` (the even set's
    Z0^0, determined only modulo 2 pi) the first stage tries the linear fit
    around every constant on a 10-degree grid and keeps the lowest cost, so a
    phase offset beyond pi does not leave the wrapped estimates split in two."""
    kmin = float(rho[mask].min()) if mask.any() else 0.0
    kmax = float(rho[mask].max()) if mask.any() else 0.0
    c = None
    for fraction in STAGES:
        stage = mask & (rho <= kmin + fraction * (kmax - kmin) + 1.0e-12)
        evaluate = evaluate_within(stage)
        if c is not None:
            model = np.tensordot(c, Z, 1)
            c = _weighted_lstsq(Z, model + _wrap(measured - model), weight * stage)
        elif constant_index is None:
            c = _weighted_lstsq(Z, measured, weight * stage)
        else:
            best = None
            for offset in np.arange(-math.pi, math.pi, math.pi / 18.0):
                trial = _weighted_lstsq(Z, offset + _wrap(measured - offset), weight * stage)
                cost = evaluate(trial)[0]
                if best is None or cost < best[0]:
                    best = (cost, trial)
            c = best[1]
        c = _newton(evaluate, c)
    return c


def _coherence_pass(Z, measured, weight, mask, rho, evaluate_within, c):
    """The second pass of either fit: each ring reweighted by the coherence of
    its residual to the first fit (coherence_weight), so rings beyond the
    data's resolution, which equal ring weights would let dilute the fit,
    drop out on their own. Returns the refitted coefficients, the reweighted
    weights and the per-ring coherence of the first fit."""
    residual = _wrap(measured - np.tensordot(c, Z, 1))
    coherence_map, rings = ring_coherence(residual, rho, mask)
    reweighted = weight * coherence_weight(coherence_map)
    if not (reweighted > 0).any():
        return c, weight, rings
    model = np.tensordot(c, Z, 1)
    c = _weighted_lstsq(Z, model + _wrap(measured - model), reweighted)
    c = _newton(evaluate_within(mask, reweighted / np.where(weight > 0, weight, 1.0)), c)
    return c, reweighted, rings


def _newton(evaluate_cost, c, iterations=50):
    """Damped (Levenberg-Marquardt style) Newton descent on evaluate(c) ->
    (cost, gradient, hessian), with the same ridge as the linear fits so the
    degenerate directions stay put."""
    c = np.array(c, dtype=np.float64)
    cost0, g0, H0 = evaluate_cost(c)
    ridge = _ridge(H0)

    def evaluate(c):
        cost, g, H = evaluate_cost(c)
        return cost + 0.5 * c @ ridge @ c, g + ridge @ c, H + ridge

    cost, g, H = evaluate(c)
    lam = 1.0e-3
    for _ in range(iterations):
        diag = np.diag(np.maximum(np.abs(np.diag(H)), 1.0e-30))
        accepted = False
        for _attempt in range(12):
            try:
                step = np.linalg.solve(H + lam * diag, g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            c_new = c - step
            cost_new, g_new, H_new = evaluate(c_new)
            if np.isfinite(cost_new) and cost_new < cost:
                accepted = True
                break
            lam *= 10.0
        if not accepted:
            break
        converged = cost - cost_new <= 1.0e-12 * max(abs(cost), 1.0e-300)
        c, cost, g, H = c_new, cost_new, g_new, H_new
        lam = max(lam / 3.0, 1.0e-12)
        if converged:
            break
    return c


def fit_odd(complex_sum, rho, phi, order=3, resolution_limit_a=None, current=None, low_resolution_limit_a=None):
    """Fit the odd (antisymmetric) aberrations to refine_ctf's phase-difference
    sum (ny, nx/2+1), whose phase is minus the aberration (module docstring):
    with S its conjugate, maximise sum |S| cos(phase(S) - Z c), i.e.
    sum Re(conj(S) exp(i Z c)). `current` is the odd set the phase already
    accounted for (removed first, added back); the result is the total set.
    Returns a dict: coefficients (total), residual (what was fitted),
    measured_phase and fitted_phase (half-plane maps of the aberration, NaN
    outside the fit's pixels), weight (|S| inside them)."""
    S = np.conj(np.asarray(complex_sum, dtype=np.complex128))
    Z = basis(order, True, rho, phi)
    mask = resolution_mask(rho, resolution_limit_a, low_resolution_limit_a)
    weight = ring_normalised(np.abs(S), rho, mask)
    if current is not None:
        current = np.asarray(current, dtype=np.float64)
        if current.size != Z.shape[0]:
            raise ValueError("current odd set has {} coefficients, the fit {}".format(current.size, Z.shape[0]))
        S = S * np.exp(-1j * np.tensordot(current, Z, 1))
    S = S * mask
    theta = np.angle(S)

    def evaluate_within(stage, factor=1.0):
        Ss = S * (weight * factor * stage / np.where(np.abs(S) > 0, np.abs(S), 1.0))     # unit phasors, ring-normalised weights

        def evaluate(c):
            e = np.conj(Ss) * np.exp(1j * np.tensordot(c, Z, 1))
            cost = -np.sum(e.real)
            gradient = np.tensordot(Z, e.imag, 2)
            hessian = np.tensordot(Z * e.real, Z, axes=([1, 2], [1, 2]))
            return cost, gradient, hessian
        return evaluate

    c = _progressive_fit(Z, theta, weight, mask, rho, evaluate_within)
    c, weight, coherence = _coherence_pass(Z, theta, weight, mask, rho, evaluate_within, c)
    fitted = np.tensordot(c, Z, 1)
    total = c + (current if current is not None else 0.0)
    measured = np.where(mask, theta, np.nan)
    return {"coefficients": total, "residual": c, "measured_phase": measured,
            "fitted_phase": np.where(mask, fitted, np.nan), "weight": weight, "order": order,
            "coherence": {round(r * RING_WIDTH, 4): R for r, R in coherence.items()}}


def per_pixel_even_phase(Axx, Axy, Ayy, bx, by):
    """The phase offset delta that minimises the quadratic form
    Axx cos^2 + 2 Axy sin cos + Ayy sin^2 - 2 bx cos - 2 by sin at every pixel,
    and the curvature there (the fit's weight): a 2-degree scan, then Newton."""
    best = np.full(Axx.shape, np.inf)
    delta = np.zeros(Axx.shape)
    for k in range(180):
        d = -math.pi + 2.0 * math.pi * k / 180.0
        s, c = math.sin(d), math.cos(d)
        f = Axx * c * c + 2.0 * Axy * s * c + Ayy * s * s - 2.0 * bx * c - 2.0 * by * s
        better = f < best
        best = np.where(better, f, best)
        delta = np.where(better, d, delta)
    for _ in range(4):
        s, c = np.sin(delta), np.cos(delta)
        f1 = 2.0 * (Ayy - Axx) * s * c + 2.0 * Axy * (c * c - s * s) + 2.0 * bx * s - 2.0 * by * c
        f2 = 2.0 * (Ayy - Axx) * (c * c - s * s) - 8.0 * Axy * s * c + 2.0 * bx * c + 2.0 * by * s
        step = np.where(f2 > 0.0, f1 / np.where(f2 > 0.0, f2, 1.0), 0.0)
        delta = delta - np.clip(step, -0.1, 0.1)
    s, c = np.sin(delta), np.cos(delta)
    curvature = 2.0 * (Ayy - Axx) * (c * c - s * s) - 8.0 * Axy * s * c + 2.0 * bx * c + 2.0 * by * s
    return delta, np.maximum(curvature, 0.0)


def fit_even(sums, rho, phi, order=4, resolution_limit_a=None, low_resolution_limit_a=None):
    """Fit the even (symmetric) aberrations to refine_ctf's five sums
    (5, ny, nx/2+1): the per-pixel phase offsets, weighted by their curvature,
    give the linear start, then Newton on the exact cost
    sum_pixels f_p(Z c). Returns the *residual* set (to add to the one the
    particles were refined with), with the same maps as fit_odd. Z0^0, the
    constant, is only determined modulo 2 pi: the set is returned with the
    phase at DC in (-pi, pi]."""
    Axx, Axy, Ayy, bx, by = (np.asarray(s, dtype=np.float64) for s in sums)
    Z = basis(order, False, rho, phi)
    mask = resolution_mask(rho, resolution_limit_a, low_resolution_limit_a)
    delta, curvature = per_pixel_even_phase(Axx, Axy, Ayy, bx, by)
    weight = ring_normalised(curvature, rho, mask)
    scale = np.where(curvature > 0, weight / np.where(curvature > 0, curvature, 1.0), 0.0)   # per-pixel factor that makes the cost ring-normalised

    def evaluate_within(stage, factor=1.0):
        m = scale * factor * stage

        def evaluate(c):
            p = np.tensordot(c, Z, 1)
            s, co = np.sin(p), np.cos(p)
            f = (Axx * co * co + 2.0 * Axy * s * co + Ayy * s * s - 2.0 * bx * co - 2.0 * by * s) * m
            f1 = (2.0 * (Ayy - Axx) * s * co + 2.0 * Axy * (co * co - s * s) + 2.0 * bx * s - 2.0 * by * co) * m
            f2 = (2.0 * (Ayy - Axx) * (co * co - s * s) - 8.0 * Axy * s * co + 2.0 * bx * co + 2.0 * by * s) * m
            return np.sum(f), np.tensordot(Z, f1, 2), np.tensordot(Z * f2, Z, axes=([1, 2], [1, 2]))
        return evaluate

    c = _progressive_fit(Z, delta, weight, mask, rho, evaluate_within, constant_index=0)
    c, weight, coherence = _coherence_pass(Z, delta, weight, mask, rho, evaluate_within, c)
    # the m = 0 terms all carry a constant (Z0^0 = 1, Z2^0 = -1, Z4^0 = 1, ... at rho = 0): keep the phase at
    # DC in (-pi, pi] by moving whole turns out of Z0^0, which changes nothing modulo 2 pi
    dc = float(np.tensordot(c, Z, 1)[rho == 0.0].mean()) if (rho == 0.0).any() else sum(c[i] * radial_polynomial(0, n, 0.0) for i, (m, n) in ((i, even_index_to_mn(i)) for i in range(c.size)) if m == 0)
    c[0] -= 2.0 * math.pi * round(dc / (2.0 * math.pi))
    fitted = np.tensordot(c, Z, 1)
    return {"coefficients": c, "residual": c, "measured_phase": np.where(mask, delta, np.nan),
            "fitted_phase": np.where(mask, fitted, np.nan), "weight": weight, "order": order,
            "coherence": {round(r * RING_WIDTH, 4): R for r, R in coherence.items()}}


# --------------------------------------------------------- refine_ctf's files

def load_sums(path):
    """refine_ctf's aberration sums: (the five even sums as (5, ny, nx/2+1),
    the complex odd sum S as (ny, nx/2+1))."""
    stack, _ps = volumes.read_mrc_volume(path)
    if stack.shape[0] != 7:
        raise ValueError("{} has {} sections, the aberration sums have 7".format(path, stack.shape[0]))
    stack = stack.astype(np.float64)
    return stack[:5], stack[5] + 1j * stack[6]


def fit_sums(sums_path, box_size, pixel_size, odd_order=3, even_order=4, resolution_limit_a=None, low_resolution_limit_a=None,
             current_odd=None, current_even=None):
    """Both fits from refine_ctf's sums file. The even result's `coefficients`
    is the residual added to `current_even` (zeros when None); the odd
    result's is the total set."""
    rho, phi = half_plane_frequencies(box_size, box_size, pixel_size)
    even_sums, S = load_sums(sums_path)
    odd = fit_odd(S, rho, phi, odd_order, resolution_limit_a, current_odd, low_resolution_limit_a)
    even = fit_even(even_sums, rho, phi, even_order, resolution_limit_a, low_resolution_limit_a)
    base = np.zeros(number_of_even_coefficients(even_order)) if current_even is None else np.asarray(current_even, dtype=np.float64)
    if base.size != even["coefficients"].size:
        raise ValueError("current even set has {} coefficients, the fit {}".format(base.size, even["coefficients"].size))
    even["coefficients"] = base + even["residual"]
    return odd, even


def full_grid(half_plane, odd):
    """A half-plane map (ny, nx/2+1) as the centred full grid (ny, nx) for
    display: the other half from Friedel symmetry, with the sign flipped for
    an odd (antisymmetric) quantity."""
    ny, nxh = half_plane.shape
    nx = 2 * (nxh - 1)
    full = np.empty((ny, nx), dtype=half_plane.dtype)
    for i in range(nx):
        fx = i - nx // 2                                   # x frequency index of column i on the centred grid
        if fx >= 0:
            full[:, i] = half_plane[:, fx]                 # rows still in fft order
        else:
            source = half_plane[:, -fx]                    # (fx, fy) from (-fx, -fy): row j -> row -j
            source = np.concatenate([source[:1], source[1:][::-1]])
            full[:, i] = -source if odd else source
    return np.fft.fftshift(full, axes=0)
