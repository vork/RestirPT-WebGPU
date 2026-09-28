"""Replicate-based comparison statistics (plan §7.3, docs/math.md#validation-stats, val §3).

Pure numpy/scipy. No image IO and no plotting here; compare.py does both.

Conventions
- An image is float HxWxC, row 0 = top. Source images are RGB (alpha dropped) or 1-channel (Y).
- The test channels are Y (Rec.709 luminance 0.2126R + 0.7152G + 0.0722B) plus R, G, B.
- Uncertainty always comes from REPLICATES of the statistic (Cycles K seeds, PT batches B,
  ReSTIR runs R): every aggregate (tile, mask region, global mean) is computed per replicate first,
  then its mean and SE across replicates. Per-pixel variances are never summed.
- "ours" is X (mean X̄, SE_X, ν_X = B−1); "ref" is R (R̄, SE_R, ν_R = K−1). Δ = X̄ − R̄.

See validation/tools/README.md for the design decisions taken where the docs leave freedom.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats as st

LUMA_709 = np.array([0.2126, 0.7152, 0.0722])
DEFAULT_CHANNELS: tuple[str, ...] = ("Y", "R", "G", "B")
TILE_SIZES: tuple[int, ...] = (16, 32, 64)

# Stage defaults (plan §7.3): relative δ for global / tile / mask aggregates and the gate tile size.
STAGE_DEFAULTS: dict[str, dict] = {
    "A": dict(delta_global=0.005, delta_tile=0.02, delta_mask=0.02, tile=32),
    "B": dict(delta_global=0.002, delta_tile=0.01, delta_mask=0.01, tile=32),
    "dyn": dict(delta_global=0.002, delta_tile=0.02, delta_mask=0.03, tile=64),
}

# Rejection-type checks (controlled by α_u) vs equivalence checks (controlled by power / sizing).
REJECTION_CHECKS = ("sidak_tiles", "chi2_red", "mean_t", "ks_ad")
EQUIVALENCE_CHECKS = ("tost_global", "tost_tiles", "tost_masks")
ALL_CHECKS = ("min_replicates",) + EQUIVALENCE_CHECKS + REJECTION_CHECKS


# --------------------------------------------------------------------------------------------
# Channels and aggregation
# --------------------------------------------------------------------------------------------

def channelize(img: np.ndarray, channels: Sequence[str] = DEFAULT_CHANNELS) -> np.ndarray:
    """(..., 3 RGB | 1 Y) -> (..., len(channels)) with channels drawn from {Y, R, G, B}."""
    a = np.asarray(img, dtype=np.float64)
    c = a.shape[-1]
    out = []
    for ch in channels:
        if ch == "Y":
            out.append(a[..., 0] if c == 1 else a[..., :3] @ LUMA_709)
        elif ch in ("R", "G", "B"):
            if c < 3:
                raise ValueError(f"channel {ch} requested but the image has {c} channel(s)")
            out.append(a[..., "RGB".index(ch)])
        else:
            raise ValueError(f"unknown channel {ch!r} (expected Y, R, G or B)")
    return np.stack(out, axis=-1)


def tile_grid(h: int, w: int, tile: int) -> tuple[int, int]:
    return -(-h // tile), -(-w // tile)


def tile_pixel_counts(h: int, w: int, tile: int) -> np.ndarray:
    """(Th, Tw) pixel count per tile; edge tiles may be partial."""
    th, tw = tile_grid(h, w, tile)
    rows = np.minimum(tile, h - np.arange(th) * tile)
    cols = np.minimum(tile, w - np.arange(tw) * tile)
    return np.outer(rows, cols).astype(np.float64)


def tile_sums(img: np.ndarray, tile: int) -> np.ndarray:
    """(..., H, W, C) -> (..., Th, Tw, C) sums over each tile (zero-padded edge tiles)."""
    a = np.asarray(img, dtype=np.float64)
    *lead, h, w, c = a.shape
    th, tw = tile_grid(h, w, tile)
    ph, pw = th * tile - h, tw * tile - w
    if ph or pw:
        pad = [(0, 0)] * len(lead) + [(0, ph), (0, pw), (0, 0)]
        a = np.pad(a, pad)
    a = a.reshape(*lead, th, tile, tw, tile, c)
    return a.sum(axis=(-4, -2))


def tile_means(img: np.ndarray, tile: int) -> np.ndarray:
    h, w = np.shape(img)[-3:-1]
    return tile_sums(img, tile) / tile_pixel_counts(h, w, tile)[..., None]


def mask_means(img: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """(..., H, W, C) x (M, H, W) bool -> (..., M, C) mean over each mask."""
    m = np.asarray(masks, dtype=np.float64)
    counts = m.sum(axis=(1, 2))
    if np.any(counts == 0):
        raise ValueError("empty mask region")
    return np.einsum("...hwc,mhw->...mc", np.asarray(img, dtype=np.float64), m) / counts[:, None]


@dataclass
class Replicates:
    """Per-replicate aggregates of one image set (one row per replicate).

    tiles[s] : (N, Th, Tw, C) per-replicate tile MEANS for tile size s
    glob     : (N, C) per-replicate image means
    masks    : (N, M, C) per-replicate mask-region means, or None
    pixel_sum / pixel_sumsq : (H, W, C) Σ_b x and Σ_b x² over replicates (per-pixel maps only;
               NaN where unavailable, e.g. Y from RGB-only ensemble sums)
    prefix   : {N: (H, W, C) Σ over the first N replicates} for curve mode (N = 2^k)
    """

    channels: tuple[str, ...]
    height: int
    width: int
    tiles: dict[int, np.ndarray]
    glob: np.ndarray
    masks: np.ndarray | None = None
    mask_names: tuple[str, ...] = ()
    pixel_sum: np.ndarray | None = None
    pixel_sumsq: np.ndarray | None = None
    prefix: dict[int, np.ndarray] = field(default_factory=dict)
    count: int | None = None  # replicate count behind pixel_sum when rows are absent

    @property
    def n(self) -> int:
        return int(self.glob.shape[0])

    def subset(self, idx: Sequence[int] | np.ndarray) -> "Replicates":
        idx = np.asarray(idx)
        return Replicates(
            channels=self.channels, height=self.height, width=self.width,
            tiles={s: v[idx] for s, v in self.tiles.items()}, glob=self.glob[idx],
            masks=None if self.masks is None else self.masks[idx], mask_names=self.mask_names,
        )

    def pixel_summary(self) -> "Summary | None":
        if self.pixel_sum is None or self.pixel_sumsq is None:
            return None
        return summarize_sums(self.pixel_sum, self.pixel_sumsq, self.count or self.n)

    def pixel_mean(self) -> np.ndarray | None:
        return None if self.pixel_sum is None else self.pixel_sum / (self.count or self.n)


class ReplicateAccumulator:
    """Streaming builder: add one replicate image at a time (keeps O(H·W·C) memory)."""

    def __init__(self, channels: Sequence[str] = DEFAULT_CHANNELS, tile_sizes: Iterable[int] = TILE_SIZES,
                 masks: np.ndarray | None = None, mask_names: Sequence[str] = (),
                 checkpoints: Iterable[int] = ()):
        self.channels = tuple(channels)
        self.tile_sizes = tuple(sorted(set(int(s) for s in tile_sizes)))
        self.masks = None if masks is None else np.asarray(masks, dtype=bool)
        self.mask_names = tuple(mask_names) or tuple(f"mask{i}" for i in range(0 if masks is None else len(masks)))
        self.checkpoints = set(int(c) for c in checkpoints)
        self._tiles: dict[int, list] = {s: [] for s in self.tile_sizes}
        self._glob: list = []
        self._masks: list = []
        self._sum = self._sumsq = None
        self._prefix: dict[int, np.ndarray] = {}
        self.shape: tuple[int, int] | None = None

    def add(self, img: np.ndarray) -> None:
        x = channelize(img, self.channels)
        h, w = x.shape[:2]
        if self.shape is None:
            self.shape = (h, w)
            self._sum = np.zeros_like(x)
            self._sumsq = np.zeros_like(x)
            if self.masks is not None and self.masks.shape[1:] != (h, w):
                raise ValueError(f"mask shape {self.masks.shape[1:]} != image shape {(h, w)}")
        elif self.shape != (h, w):
            raise ValueError(f"image shape {(h, w)} != first replicate {self.shape}")
        for s in self.tile_sizes:
            self._tiles[s].append(tile_means(x, s))
        self._glob.append(x.mean(axis=(0, 1)))
        if self.masks is not None:
            self._masks.append(mask_means(x, self.masks))
        self._sum += x
        self._sumsq += x * x
        n = len(self._glob)
        if n in self.checkpoints:
            self._prefix[n] = self._sum.copy()

    def finish(self) -> Replicates:
        if not self._glob:
            raise ValueError("no replicates added")
        h, w = self.shape
        return Replicates(
            channels=self.channels, height=h, width=w,
            tiles={s: np.stack(v) for s, v in self._tiles.items()}, glob=np.stack(self._glob),
            masks=np.stack(self._masks) if self._masks else None, mask_names=self.mask_names,
            pixel_sum=self._sum, pixel_sumsq=self._sumsq, prefix=dict(self._prefix),
        )


def aggregate_stack(stack: np.ndarray, channels: Sequence[str] = DEFAULT_CHANNELS,
                    tile_sizes: Iterable[int] = TILE_SIZES, masks: np.ndarray | None = None,
                    mask_names: Sequence[str] = (), checkpoints: Iterable[int] = ()) -> Replicates:
    """(N, H, W, C_src) replicate images -> Replicates."""
    acc = ReplicateAccumulator(channels, tile_sizes, masks, mask_names, checkpoints)
    for img in stack:
        acc.add(img)
    return acc.finish()


def replicates_from_sums(data: Mapping[str, np.ndarray], channels: Sequence[str] = DEFAULT_CHANNELS) -> Replicates:
    """Ensemble aggregates (the `ensembleStats` read-back; format in README) -> Replicates.

    Keys: tiles16/tiles32/tiles64 (R,Th,Tw,C) per-run tile SUMS; global (R,C) per-run image sums;
    masks (R,M,C) per-run mask sums + mask_pixels (M,); pixel_sum/pixel_sumsq (H,W,C); count;
    optional channels (C,) names (default R,G,B), mask_names (M,), height/width.
    """
    src = tuple(str(c) for c in np.asarray(data["channels"]).tolist()) if "channels" in data else ("R", "G", "B")
    if "pixel_sum" in data:
        h, w = np.asarray(data["pixel_sum"]).shape[:2]
    else:
        h, w = int(data["height"]), int(data["width"])

    def pick(a: np.ndarray, linear: bool = True) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64)
        out = []
        for ch in channels:
            if ch in src:
                out.append(a[..., src.index(ch)])
            elif ch == "Y" and linear and all(c in src for c in "RGB"):
                out.append(a[..., [src.index(c) for c in "RGB"]] @ LUMA_709)
            elif ch == "Y" and not linear:
                out.append(np.full(a.shape[:-1], np.nan))  # Y² is not linear in RGB sums
            else:
                raise ValueError(f"channel {ch} not derivable from ensemble channels {src}")
        return np.stack(out, axis=-1)

    tiles = {}
    for s in TILE_SIZES:
        if f"tiles{s}" in data:
            tiles[s] = pick(data[f"tiles{s}"]) / tile_pixel_counts(h, w, s)[..., None]
    glob = pick(data["global"]) / (h * w)
    masks = None
    names: tuple[str, ...] = ()
    if "masks" in data and np.size(data["masks"]):
        mp = np.asarray(data["mask_pixels"], dtype=np.float64)
        masks = pick(data["masks"]) / mp[:, None]
        names = tuple(str(x) for x in np.asarray(data["mask_names"]).tolist()) if "mask_names" in data \
            else tuple(f"mask{i}" for i in range(len(mp)))
    ps = pick(data["pixel_sum"]) if "pixel_sum" in data else None
    pq = pick(data["pixel_sumsq"], linear=False) if "pixel_sumsq" in data else None
    count = int(np.asarray(data["count"])) if "count" in data else int(glob.shape[0])
    return Replicates(tuple(channels), h, w, tiles, glob, masks, names, ps, pq, {}, count)


# --------------------------------------------------------------------------------------------
# Replicate summaries and the Welch test
# --------------------------------------------------------------------------------------------

@dataclass
class Summary:
    mean: np.ndarray
    se: np.ndarray
    nu: np.ndarray
    n: int
    const: np.ndarray  # True where all replicates are identical (exact zero variance)


def summarize(values: np.ndarray) -> Summary:
    """Replicate axis 0 -> mean, SE = s/√n, ν = n−1."""
    v = np.asarray(values, dtype=np.float64)
    n = v.shape[0]
    if n < 2:
        raise ValueError("need >= 2 replicates for a replicate-based SE")
    mean = v.mean(axis=0)
    const = np.ptp(v, axis=0) == 0
    var = np.where(const, 0.0, v.var(axis=0, ddof=1))
    se = np.sqrt(var / n)
    return Summary(mean, se, np.full(mean.shape, n - 1.0), n, const)


def summarize_sums(s: np.ndarray, sq: np.ndarray, n: int) -> Summary:
    if n < 2:
        raise ValueError("need >= 2 replicates for a replicate-based SE")
    mean = s / n
    var = np.maximum((sq - s * s / n) / (n - 1), 0.0)
    return Summary(mean, np.sqrt(var / n), np.full(np.shape(mean), n - 1.0), n, var == 0)


def welch(mx, sex, nux, mr, ser, nur):
    """Welch t with Welch–Satterthwaite ν. Returns (Δ, SE_Δ, t, ν), all arrays.

    0/0 guards: SE_Δ = 0 gives t = 0 when Δ = 0 else ±inf, and ν = NaN (undefined).
    ν_X or ν_R = inf (an exact/analytic side with SE = 0) is allowed.
    """
    mx, sex, nux, mr, ser, nur = (np.asarray(a, dtype=np.float64) for a in (mx, sex, nux, mr, ser, nur))
    d = mx - mr
    vx, vr = sex ** 2, ser ** 2
    v = vx + vr
    se = np.sqrt(v)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, d / np.where(se > 0, se, 1.0), np.where(d == 0, 0.0, np.sign(d) * np.inf))
        t = np.where(np.isnan(se) | np.isnan(d), np.nan, t)
        tx = np.where(vx > 0, vx ** 2 / nux, 0.0)
        tr = np.where(vr > 0, vr ** 2 / nur, 0.0)
        den = tx + tr
        # den = 0 with v > 0 happens only when every non-zero variance has ν = inf (known SE): normal limit
        nu = np.where(v > 0, np.where(den > 0, v ** 2 / np.where(den > 0, den, 1.0), np.inf), np.nan)
    return d, se, t, nu


def p_two_sided(t, nu) -> np.ndarray:
    t, nu = np.asarray(t, dtype=np.float64), np.asarray(nu, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        p = 2.0 * st.t.sf(np.abs(t), np.where(np.isnan(nu), 1.0, nu))
    return np.where(np.isnan(nu) | np.isnan(t), np.nan, np.minimum(p, 1.0))


def welch_summaries(x: Summary, r: Summary):
    return welch(x.mean, x.se, x.nu, r.mean, r.se, r.nu)


# --------------------------------------------------------------------------------------------
# Equivalence (TOST), MDB, multiplicity, sizing
# --------------------------------------------------------------------------------------------

def t_quantile(q: float, nu) -> np.ndarray:
    nu = np.asarray(nu, dtype=np.float64)
    return np.where(np.isnan(nu), np.nan, st.t.ppf(q, np.where(np.isnan(nu), 1.0, nu)))


def tost(delta, se_d, nu, rbar, rbar_image, rel_margin, *, alpha: float = 0.01, dark_frac: float = 0.05,
         zero_tol: float = 1e-6) -> dict[str, np.ndarray]:
    """TOST equivalence at α per side (the (1−2α) CI of Δ lies inside ±margin).

    margin = δ·|R̄_A|                 if R̄_A ≥ dark_frac·R̄_image (relative)
           = δ·dark_frac·|R̄_image|   otherwise (absolute, "dark tile")
    pass ⇔ |Δ| + t_{1−α,ν}·SE_Δ < margin.
    Zero-variance aggregates (SE_Δ = 0): pass ⇔ |Δ| ≤ zero_tol·|R̄_image|.
    Every ratio is guarded (no 0/0): `rel` = Δ/denominator and `ratio` = bound/margin.
    """
    delta, se_d, nu, rbar = (np.asarray(a, dtype=np.float64) for a in (delta, se_d, nu, rbar))
    ri = abs(float(rbar_image))
    dark = rbar < dark_frac * ri
    denom = np.where(dark, dark_frac * ri, np.abs(rbar))
    margin = rel_margin * denom
    zero_var = se_d == 0
    tcrit = np.where(zero_var, 0.0, t_quantile(1.0 - alpha, np.where(zero_var, np.inf, nu)))
    bound = np.abs(delta) + tcrit * se_d
    ok_zero = np.abs(delta) <= zero_tol * ri
    ok = np.where(zero_var, ok_zero, bound < margin)
    ok = ok & np.isfinite(bound)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(denom > 0, delta / np.where(denom > 0, denom, 1.0), np.where(delta == 0, 0.0, np.sign(delta) * np.inf))
        se_rel = np.where(denom > 0, se_d / np.where(denom > 0, denom, 1.0), np.where(se_d == 0, 0.0, np.inf))
        ratio = np.where(margin > 0, bound / np.where(margin > 0, margin, 1.0), np.where(bound == 0, 0.0, np.inf))
    return dict(dark=dark, zero_var=zero_var, denom=denom, margin=margin, tcrit=tcrit, bound=bound,
                rel=rel, se_rel=se_rel, ratio=ratio, passed=ok)


def mdb(se_rel, alpha: float = 0.01, power: float = 0.9) -> np.ndarray:
    """Minimum detectable (relative) bias ≈ (z_{1−α/2} + z_{power})·SE_Δ/R̄."""
    return (st.norm.ppf(1 - alpha / 2) + st.norm.ppf(power)) * np.asarray(se_rel, dtype=np.float64)


def sidak_alpha(alpha: float, m: int) -> float:
    """Per-test level α′ = 1 − (1 − α)^{1/m} (computed without cancellation)."""
    if m < 1:
        raise ValueError("m must be >= 1")
    return float(-np.expm1(np.log1p(-alpha) / m))


def sidak_threshold(alpha: float, m: int, nu) -> np.ndarray:
    """Two-sided |t| threshold t* = t_ν⁻¹(1 − α′/2)."""
    return t_quantile(1.0 - sidak_alpha(alpha, m) / 2.0, nu)


def alpha_unit(n_units: int, alpha_suite: float = 0.01) -> float:
    """Suite FWER: α_u = 1 − (1 − α_suite)^{1/n_units}  (= 1 − 0.99^{1/n_units})."""
    return sidak_alpha(alpha_suite, max(int(n_units), 1))


def bh_qvalues(p) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values (q-values); NaN entries stay NaN and are not counted."""
    p = np.asarray(p, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    flat = p.ravel()
    ok = np.flatnonzero(~np.isnan(flat))
    if ok.size == 0:
        return out
    pv = flat[ok]
    order = np.argsort(pv)
    m = pv.size
    q = pv[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.minimum(q, 1.0)
    o = out.ravel()
    o[ok] = res
    return o.reshape(p.shape)


def bh_reject(p, q: float = 0.01) -> np.ndarray:
    qv = bh_qvalues(p)
    return np.where(np.isnan(qv), False, qv <= q)


def sizing_target(delta: float, m: int = 1, nu: float = np.inf, alpha: float = 0.01) -> float:
    """Largest relative SE_Δ that sizes an aggregate family: δ / (t_{1−α,ν} + z_{1−0.005/m})."""
    return float(delta / (st.t.ppf(1 - alpha, nu) + st.norm.ppf(1 - 0.005 / max(m, 1))))


def required_replicates(sd_rel: float, delta: float, m: int = 1, alpha: float = 0.01, *,
                        se_other_rel: float = 0.0, nu_other: float = np.inf,
                        sd_other_rel_same_n: float | None = None, n_min: int = 2,
                        n_max: int = 10_000_000) -> int | None:
    """Smallest replicate count n with SE_Δ(n) ≤ sizing_target(δ, m, ν(n)).

    sd_rel: per-replicate relative SD of the aggregate on the side being sized (from a pilot:
            sd = SE_pilot·√n_pilot, see `pilot_sd`). SE(n) = sd/√n.
    se_other_rel/nu_other: the other side's fixed relative SE and dof (e.g. an existing reference).
    sd_other_rel_same_n: size BOTH sides with the same n instead (the other side's per-replicate SD).
    Returns None when no n ≤ n_max can reach the target (enlarge the aggregate; never loosen δ).
    """
    def ok(n: int) -> bool:
        sx = sd_rel / math.sqrt(n)
        if sd_other_rel_same_n is not None:
            so, no = sd_other_rel_same_n / math.sqrt(n), n - 1.0
        else:
            so, no = se_other_rel, nu_other
        _, se, _, nu = welch(0.0, sx, n - 1.0, 0.0, so, no)
        nu = float(nu) if np.isfinite(nu) else np.inf
        return float(se) <= sizing_target(delta, m, nu, alpha)

    if not ok(n_max):
        return None
    lo, hi = n_min, n_max
    if ok(lo):
        return lo
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi


def pilot_sd(pilot_se: float, pilot_n: int) -> float:
    """Per-replicate SD implied by a pilot's SE with pilot_n replicates."""
    return float(pilot_se) * math.sqrt(pilot_n)


# --------------------------------------------------------------------------------------------
# Distribution checks over the aggregate family {t_A}
# --------------------------------------------------------------------------------------------

MIN_FAMILY = 8  # below this many finite t_A the distribution checks are "not applicable"


def _family(t, nu):
    t, nu = np.asarray(t, dtype=np.float64).ravel(), np.asarray(nu, dtype=np.float64).ravel()
    ok = np.isfinite(t) & np.isfinite(nu) & (nu > 0)
    return t[ok], nu[ok]


def chi2_red(t, nu) -> dict:
    """χ²_red = mean t_A² ; E = mean ν/(ν−2) ; pass ⇔ χ²_red < E·(1 + 4·√(2/m)).

    Tiles with ν ≤ 2 (infinite t variance) are excluded. `p_moment` is a normal approximation that
    uses the exact t(ν) variance of t² (ν > 4), reported for information only.
    """
    t, nu = _family(t, nu)
    keep = nu > 2
    t, nu = t[keep], nu[keep]
    m = t.size
    if m < MIN_FAMILY:
        return dict(applicable=False, m=int(m), passed=True)
    stat = float(np.mean(t * t))
    e = nu / (nu - 2)
    expect = float(np.mean(e))
    bound = expect * (1 + 4 * math.sqrt(2 / m))
    v4 = nu > 4
    p_mom = None
    if np.all(v4):
        var = 3 * nu ** 2 / ((nu - 2) * (nu - 4)) - e ** 2
        z = (stat - expect) / math.sqrt(float(np.mean(var)) / m)
        p_mom = float(st.norm.sf(z))
    return dict(applicable=True, m=int(m), stat=stat, expect=expect, bound=bound,
                p_moment=p_mom, passed=bool(stat < bound))


def mean_t(t, nu, alpha: float) -> dict:
    """Mean of t_A; under H0 ≈ 0 with SE √(mean ν/(ν−2)/m) (→ 1/√m for large ν)."""
    t, nu = _family(t, nu)
    keep = nu > 2
    t, nu = t[keep], nu[keep]
    m = t.size
    if m < MIN_FAMILY:
        return dict(applicable=False, m=int(m), passed=True)
    mt = float(np.mean(t))
    se = math.sqrt(float(np.mean(nu / (nu - 2))) / m)
    z = mt / se
    p = float(2 * st.norm.sf(abs(z)))
    return dict(applicable=True, m=int(m), mean=mt, se=se, z=z, p=p, alpha=alpha, passed=bool(p >= alpha))


def _ad_inf_cdf(z: float) -> float:
    """Asymptotic Anderson–Darling CDF (Marsaglia & Marsaglia 2004, ADinf)."""
    if z <= 0:
        return 0.0
    if z < 2:
        return math.exp(-1.2337141 / z) / math.sqrt(z) * (
            2.00012 + (.247105 - (.0649821 - (.0347962 - (.011672 - .00168691 * z) * z) * z) * z) * z)
    return math.exp(-math.exp(1.0776 - (2.30695 - (.43424 - (.082433 - (.008056 - .0003146 * z) * z) * z) * z) * z))


def ks_ad(t, nu, alpha: float) -> dict:
    """KS and Anderson–Darling of {t_A} against t(ν_A) (probability-integral transform u = F_ν(t)).

    Two tests, Bonferroni-combined: pass ⇔ min(p_KS, p_AD) ≥ α/2.
    """
    t, nu = _family(t, nu)
    m = t.size
    if m < MIN_FAMILY:
        return dict(applicable=False, m=int(m), passed=True)
    u = st.t.cdf(t, nu)
    ks = st.kstest(u, "uniform")
    lc = np.sort(st.t.logcdf(t, nu))            # ln u_(i)
    ls = np.sort(st.t.logsf(t, nu))             # ascending ln(1 − u) = ln(1 − u_(n+1−i))
    i = np.arange(1, m + 1)
    a2 = float(-m - np.mean((2 * i - 1) * (lc + ls)))
    p_ad = float(min(max(1.0 - _ad_inf_cdf(a2), 0.0), 1.0))
    p_ks = float(ks.pvalue)
    return dict(applicable=True, m=int(m), ks_stat=float(ks.statistic), p_ks=p_ks, ad_stat=a2, p_ad=p_ad,
                alpha=alpha, passed=bool(min(p_ks, p_ad) >= alpha / 2))


# --------------------------------------------------------------------------------------------
# Gate evaluation (one "unit" = scene × config × gate, run on each channel)
# --------------------------------------------------------------------------------------------

@dataclass
class GateSpec:
    stage: str = "A"
    delta_global: float = 0.005
    delta_tile: float = 0.02
    delta_mask: float = 0.02
    tile: int = 32
    channels: tuple[str, ...] = DEFAULT_CHANNELS
    alpha_tost: float = 0.01
    n_units: int = 1
    alpha_suite: float = 0.01
    dark_frac: float = 0.05
    zero_tol: float = 1e-6
    power: float = 0.9
    fdr_q: float = 0.01
    min_replicates: int = 16
    gating: dict[str, bool] = field(default_factory=lambda: {c: True for c in ALL_CHECKS})

    @classmethod
    def for_stage(cls, stage: str, **overrides) -> "GateSpec":
        if stage not in STAGE_DEFAULTS:
            raise ValueError(f"unknown stage {stage!r}; expected one of {sorted(STAGE_DEFAULTS)}")
        kw = dict(STAGE_DEFAULTS[stage])
        kw.update({k: v for k, v in overrides.items() if v is not None})
        g = {c: True for c in ALL_CHECKS}
        g.update(kw.pop("gating", None) or {})
        return cls(stage=stage, gating=g, **kw)

    @property
    def alpha_u(self) -> float:
        return alpha_unit(self.n_units, self.alpha_suite)

    def with_(self, **kw) -> "GateSpec":
        return replace(self, **kw)

    def to_json(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["channels"] = list(self.channels)
        d["alpha_u"] = self.alpha_u
        return d


@dataclass
class GateResult:
    passed: bool
    checks: list[dict]
    summary: dict
    arrays: dict[str, dict[str, np.ndarray]]


def _fl(x) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else (None if math.isnan(x) else (1e308 if x > 0 else -1e308))


def evaluate_gate(ours: Replicates, ref: Replicates, spec: GateSpec) -> GateResult:
    """Run every §7.3 check of one unit. Rejection-type checks use α_u; TOST uses α_tost per side."""
    if ours.channels != ref.channels:
        raise ValueError(f"channel mismatch {ours.channels} vs {ref.channels}")
    if (ours.height, ours.width) != (ref.height, ref.width):
        raise ValueError(f"image size mismatch {(ours.height, ours.width)} vs {(ref.height, ref.width)}")
    if spec.tile not in ours.tiles or spec.tile not in ref.tiles:
        raise ValueError(f"tile size {spec.tile} not aggregated")
    au = spec.alpha_u
    checks: list[dict] = []
    arrays: dict[str, dict[str, np.ndarray]] = {}
    per_ch: dict[str, dict] = {}

    def add(name, ch, ok, value=None, threshold=None, **detail):
        checks.append(dict(name=name, channel=ch, passed=bool(ok), gating=bool(spec.gating.get(name, True)),
                           value=value, threshold=threshold, **detail))

    nmin = min(ours.n, ref.n)
    add("min_replicates", "all", nmin >= spec.min_replicates, value=nmin, threshold=spec.min_replicates,
        n_ours=ours.n, n_ref=ref.n)

    gx, gr = summarize(ours.glob), summarize(ref.glob)
    tx, tr = summarize(ours.tiles[spec.tile]), summarize(ref.tiles[spec.tile])
    mx = mr = None
    if ours.masks is not None and ref.masks is not None:
        mx, mr = summarize(ours.masks), summarize(ref.masks)
    nt = int(np.prod(tx.mean.shape[:2]))

    for ci, ch in enumerate(spec.channels):
        if ch not in ours.channels:
            raise ValueError(f"channel {ch} not aggregated")
        k = ours.channels.index(ch)
        rimg = float(gr.mean[k])
        # ---- global
        d, se, t, nu = welch(gx.mean[k], gx.se[k], gx.nu[k], gr.mean[k], gr.se[k], gr.nu[k])
        to = tost(d, se, nu, gr.mean[k], rimg, spec.delta_global, alpha=spec.alpha_tost,
                  dark_frac=spec.dark_frac, zero_tol=spec.zero_tol)
        g_mdb = float(mdb(to["se_rel"], spec.alpha_tost, spec.power))
        glob = dict(ours=_fl(gx.mean[k]), ref=_fl(gr.mean[k]), delta=_fl(d), rel=_fl(to["rel"]), se=_fl(se),
                    se_rel=_fl(to["se_rel"]), t=_fl(t), nu=_fl(nu), p=_fl(p_two_sided(t, nu)),
                    bound_rel=_fl(to["ratio"] * spec.delta_global), ratio=_fl(to["ratio"]), mdb=_fl(g_mdb),
                    delta_margin=spec.delta_global, zero_var=bool(to["zero_var"]), passed=bool(to["passed"]),
                    sizing_target=sizing_target(spec.delta_global, 1, float(nu) if np.isfinite(nu) else np.inf,
                                                spec.alpha_tost))
        glob["sizing_ratio"] = _fl(to["se_rel"] / glob["sizing_target"])
        glob["replicate_multiplier_needed"] = _fl(max(1.0, float(to["se_rel"] / glob["sizing_target"]) ** 2))
        add("tost_global", ch, to["passed"], value=glob["ratio"], threshold=1.0, rel=glob["rel"],
            bound_rel=glob["bound_rel"], delta=spec.delta_global, mdb=glob["mdb"])
        # ---- tiles
        d, se, t, nu = welch(tx.mean[..., k], tx.se[..., k], tx.nu[..., k], tr.mean[..., k], tr.se[..., k], tr.nu[..., k])
        to = tost(d, se, nu, tr.mean[..., k], rimg, spec.delta_tile, alpha=spec.alpha_tost,
                  dark_frac=spec.dark_frac, zero_tol=spec.zero_tol)
        p = p_two_sided(t, nu)
        a1 = sidak_alpha(au, max(nt, 1))
        sidak_rej = np.where(np.isnan(p), False, p < a1)
        bh = bh_reject(p, spec.fdr_q)
        t_mdb = mdb(to["se_rel"], spec.alpha_tost, spec.power)
        finite_nu = nu[np.isfinite(nu)]
        nu_med = float(np.median(finite_nu)) if finite_nu.size else np.inf
        target = sizing_target(spec.delta_tile, nt, nu_med, spec.alpha_tost)
        # sizing ratio in margin units: SE_Δ·(t+z)/margin ≤ 1 means sized (dark tiles use their abs margin)
        with np.errstate(divide="ignore", invalid="ignore"):
            size_ratio = np.where(to["margin"] > 0, se * spec.delta_tile / np.where(to["margin"] > 0, to["margin"], 1) / target,
                                  np.where(se == 0, 0.0, np.inf))
        c2 = chi2_red(t, nu)
        mt = mean_t(t, nu, au)
        ka = ks_ad(t, nu, au)
        fails = int(np.sum(~to["passed"]))
        worst = np.unravel_index(int(np.nanargmax(np.where(np.isfinite(to["ratio"]), to["ratio"], 1e308))), to["ratio"].shape)
        tiles = dict(m=nt, tile=spec.tile, tost_failed=fails, dark=int(to["dark"].sum()),
                     zero_var=int(to["zero_var"].sum()), max_ratio=_fl(to["ratio"][worst]),
                     worst_tile=[int(worst[0]), int(worst[1])], worst_rel=_fl(to["rel"][worst]),
                     max_abs_rel=_fl(np.nanmax(np.abs(np.where(np.isfinite(to["rel"]), to["rel"], np.nan)))) if np.any(np.isfinite(to["rel"])) else None,
                     sidak_alpha=a1, sidak_rejected=int(sidak_rej.sum()), max_abs_t=_fl(np.nanmax(np.abs(np.where(np.isfinite(t), t, np.nan)))) if np.any(np.isfinite(t)) else None,
                     bh_rejected=int(bh.sum()), chi2_red=c2, mean_t=mt, ks_ad=ka,
                     mdb_max=_fl(np.max(t_mdb)), mdb_median=_fl(np.median(t_mdb)),
                     sizing_target=target, sizing_ratio_max=_fl(np.max(size_ratio)),
                     replicate_multiplier_needed=_fl(max(1.0, float(np.max(size_ratio)) ** 2)))
        add("tost_tiles", ch, fails == 0, value=tiles["max_ratio"], threshold=1.0, failed_tiles=fails, m=nt,
            delta=spec.delta_tile, worst_tile=tiles["worst_tile"], worst_rel=tiles["worst_rel"])
        add("sidak_tiles", ch, tiles["sidak_rejected"] == 0, value=tiles["max_abs_t"],
            threshold=_fl(np.nanmax(sidak_threshold(au, max(nt, 1), finite_nu))) if finite_nu.size else None,
            rejected=tiles["sidak_rejected"], alpha=au)
        add("chi2_red", ch, c2["passed"], value=c2.get("stat"), threshold=c2.get("bound"), applicable=c2["applicable"])
        add("mean_t", ch, mt["passed"], value=mt.get("p"), threshold=au, z=mt.get("z"), applicable=mt["applicable"])
        add("ks_ad", ch, ka["passed"], value=None if not ka["applicable"] else min(ka["p_ks"], ka["p_ad"]),
            threshold=au / 2, applicable=ka["applicable"])
        arrays[ch] = dict(tile_t=t, tile_nu=nu, tile_p=p, tile_rel=to["rel"], tile_pass=to["passed"],
                          tile_dark=to["dark"], tile_sidak=sidak_rej, tile_bh=bh, tile_mdb=t_mdb,
                          tile_ratio=to["ratio"])
        # ---- masks
        masks = []
        if mx is not None:
            d, se, t, nu = welch(mx.mean[:, k], mx.se[:, k], mx.nu[:, k], mr.mean[:, k], mr.se[:, k], mr.nu[:, k])
            to = tost(d, se, nu, mr.mean[:, k], rimg, spec.delta_mask, alpha=spec.alpha_tost,
                      dark_frac=spec.dark_frac, zero_tol=spec.zero_tol)
            mm = mdb(to["se_rel"], spec.alpha_tost, spec.power)
            for i, name in enumerate(ours.mask_names):
                masks.append(dict(name=name, rel=_fl(to["rel"][i]), se_rel=_fl(to["se_rel"][i]), t=_fl(t[i]),
                                  nu=_fl(nu[i]), ratio=_fl(to["ratio"][i]), mdb=_fl(mm[i]),
                                  dark=bool(to["dark"][i]), passed=bool(to["passed"][i])))
            add("tost_masks", ch, bool(np.all(to["passed"])), value=_fl(np.max(to["ratio"])) if len(masks) else None,
                threshold=1.0, failed=[m["name"] for m in masks if not m["passed"]], delta=spec.delta_mask)
        per_ch[ch] = dict(global_=glob, tiles=tiles, masks=masks)

    gating_fail = [c for c in checks if c["gating"] and not c["passed"]]
    summary = dict(spec=spec.to_json(), alpha_u=au, n_ours=ours.n, n_ref=ref.n, channels=per_ch,
                   failed_checks=[f"{c['name']}[{c['channel']}]" for c in gating_fail])
    return GateResult(passed=not gating_fail, checks=checks, summary=summary, arrays=arrays)


def significant(result: GateResult, alpha_global: float | None = None) -> bool:
    """Did any REJECTION-type test (or the global Welch t at α_tost two-sided) reject H0: Δ = 0?"""
    a = alpha_global if alpha_global is not None else result.summary["spec"]["alpha_tost"]
    for c in result.checks:
        if c["name"] in REJECTION_CHECKS and not c["passed"]:
            return True
    for ch in result.summary["channels"].values():
        p = ch["global_"]["p"]
        if p is not None and p < a:
            return True
    return False


# --------------------------------------------------------------------------------------------
# Convergence and error metrics
# --------------------------------------------------------------------------------------------

def _norm_scale(ref: np.ndarray) -> float:
    m = float(np.mean(ref))
    return 1.0 / m if m > 0 else 1.0


def rel_mse(x, r, eps: float = 0.01, normalize: bool = True) -> float:
    """relMSE = mean (x − r)² / (r² + ε), after scaling both so mean(r) = 1."""
    x, r = np.asarray(x, dtype=np.float64), np.asarray(r, dtype=np.float64)
    s = _norm_scale(r) if normalize else 1.0
    x, r = x * s, r * s
    return float(np.mean((x - r) ** 2 / (r * r + eps)))


def rel_mse_corr(x, r, se_r, eps: float = 0.01, normalize: bool = True) -> float:
    """Noise-corrected relMSE: mean [(x − r)² − SE_R²] / (r² + ε), same normalisation."""
    x, r, se_r = (np.asarray(a, dtype=np.float64) for a in (x, r, se_r))
    s = _norm_scale(r) if normalize else 1.0
    x, r, se_r = x * s, r * s, se_r * s
    return float(np.mean(((x - r) ** 2 - se_r ** 2) / (r * r + eps)))


def mse_corr(x, r, se_r, normalize: bool = True) -> float:
    """Noise-corrected MSE (mean(r) = 1 units): mean (x − r)² − SE_R²."""
    x, r, se_r = (np.asarray(a, dtype=np.float64) for a in (x, r, se_r))
    s = _norm_scale(r) if normalize else 1.0
    return float(np.mean((x * s - r * s) ** 2 - (se_r * s) ** 2))


def rmsrb(x, r, se_x, se_r, floor_frac: float = 0.05) -> float:
    """RMS relative bias, noise-corrected, over pixels with r above floor_frac·mean(r)."""
    x, r, se_x, se_r = (np.asarray(a, dtype=np.float64) for a in (x, r, se_x, se_r))
    keep = r > floor_frac * max(float(np.mean(r)), 0.0)
    if not np.any(keep):
        return 0.0
    v = ((x - r) ** 2 - se_x ** 2 - se_r ** 2)[keep] / r[keep] ** 2
    return float(math.sqrt(max(0.0, float(np.mean(v)))))


def grey(img) -> np.ndarray:
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 3 and a.shape[-1] >= 3:
        return a[..., :3] @ LUMA_709
    return a[..., 0] if a.ndim == 3 else a


def mape(img, gt) -> float:
    """GRIS §9.3 fn. 13: mean |I − Ĩ_gt| / (0.01·mean(Ĩ_gt) + Ĩ_gt), on greyscale (Rec.709 Y)."""
    i, g = grey(img), grey(gt)
    return float(np.mean(np.abs(i - g) / (0.01 * np.mean(g) + g)))


def bnr(delta, se_d) -> np.ndarray:
    """Bias-to-noise ratio |Δ|/SE_Δ (0/0 → 0)."""
    delta, se_d = np.asarray(delta, dtype=np.float64), np.asarray(se_d, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(se_d > 0, np.abs(delta) / np.where(se_d > 0, se_d, 1), np.where(delta == 0, 0.0, np.inf))


def slope_fit(n, y) -> dict:
    """OLS fit log y = a + s·log n over points with y > 0."""
    n, y = np.asarray(n, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = (y > 0) & (n > 0)
    if ok.sum() < 3:
        return dict(slope=None, intercept=None, stderr=None, points=int(ok.sum()))
    r = st.linregress(np.log(n[ok]), np.log(y[ok]))
    return dict(slope=float(r.slope), intercept=float(r.intercept), stderr=float(r.stderr), points=int(ok.sum()))


def hdr_flip(ref_rgb, test_rgb) -> tuple[float, np.ndarray]:
    """HDR-FLIP (flip_evaluator 1.7) of two converged RGB means; returns (mean, HxW error map)."""
    import flip_evaluator as flip

    r = np.ascontiguousarray(np.clip(np.asarray(ref_rgb, dtype=np.float32), 0, None))
    t = np.ascontiguousarray(np.clip(np.asarray(test_rgb, dtype=np.float32), 0, None))
    if r.shape[-1] == 1:
        r, t = np.repeat(r, 3, -1), np.repeat(t, 3, -1)
    emap, mean, _ = flip.evaluate(r, t, "HDR", applyMagma=False)
    emap = np.asarray(emap, dtype=np.float64)
    if emap.ndim == 3:
        emap = emap[..., 0]
    return float(mean), emap


# --------------------------------------------------------------------------------------------
# Suite-level decisions
# --------------------------------------------------------------------------------------------

def confirmatory_decision(first_passed: bool, rerun_passed: bool | None = None,
                          first_seeds: Iterable | None = None, rerun_seeds: Iterable | None = None) -> dict:
    """Suite FWER rule: a failed unit is re-run once on DISJOINT seeds and fails only if the re-run fails.

    status: pass | rerun_required | pass_on_rerun | fail | invalid_rerun
    """
    if first_passed:
        return dict(status="pass", passed=True, reason="first run passed")
    if rerun_passed is None:
        return dict(status="rerun_required", passed=False,
                    reason="first run failed; re-run once on a disjoint seed range")
    if first_seeds is None or rerun_seeds is None:
        return dict(status="invalid_rerun", passed=False, reason="seed sets unknown; disjointness cannot be verified")
    a, b = set(map(str, first_seeds)), set(map(str, rerun_seeds))
    if not a or not b or a & b:
        return dict(status="invalid_rerun", passed=False,
                    reason=f"re-run seeds overlap the first run ({len(a & b)} shared) or are empty")
    if rerun_passed:
        return dict(status="pass_on_rerun", passed=True, reason="first run failed, disjoint confirmatory re-run passed")
    return dict(status="fail", passed=False, reason="first run and disjoint confirmatory re-run both failed")


# --------------------------------------------------------------------------------------------
# Calibration (Gate 1): A/A re-splits and planted biases
# --------------------------------------------------------------------------------------------

def _halves(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    perm = rng.permutation(n)
    h = n // 2
    return np.sort(perm[:h]), np.sort(perm[h:2 * h])


def _binom_excess_p(k: int, n: int, p0: float) -> float:
    """One-sided P(X ≥ k) for X ~ Binomial(n, p0)."""
    return float(st.binom.sf(k - 1, n, p0)) if n > 0 else 1.0


def aa_split(reps: Replicates, spec: GateSpec, n_splits: int = 20, rng: np.random.Generator | int | None = 0,
             nominal: Sequence[float] = (0.05, 0.01), excess_p: float = 1e-3) -> dict:
    """A/A calibration by ≥ 20 disjoint random re-splits of one replicate set into halves.

    Each split runs the full gate (half A as "ours", half B as "ref") with min_replicates relaxed to
    the half size. Reported rates are pooled over splits × channels (× tiles for per-tile rates):
    - per-tile uncorrected false-positive rate at each nominal α (p_tile < α);
    - failure rate of each rejection-type check vs its nominal α_u;
    - TOST pass rate (a power diagnostic: an A/A TOST fails only when under-powered).
    Verdict `ok`: no rate is significantly above nominal (one-sided binomial p < excess_p). Splits of
    one set are not independent, so this binomial test is approximate (documented in README).
    """
    if n_splits < 20:
        raise ValueError("A/A calibration needs >= 20 re-splits (plan §7.3)")
    rng = np.random.default_rng(rng)
    sp = spec.with_(min_replicates=2)
    tile_p: list[np.ndarray] = []
    rej = {c: 0 for c in REJECTION_CHECKS}
    rej_n = {c: 0 for c in REJECTION_CHECKS}
    tost_pass = 0
    gate_pass = 0
    rej_any = 0
    for _ in range(n_splits):
        a, b = _halves(reps.n, rng)
        res = evaluate_gate(reps.subset(a), reps.subset(b), sp)
        gate_pass += res.passed
        any_r = False
        for c in res.checks:
            if c["name"] in rej and c.get("applicable", True) is not False:
                rej_n[c["name"]] += 1
                if not c["passed"]:
                    rej[c["name"]] += 1
                    any_r = True
        rej_any += any_r
        tost_pass += all(c["passed"] for c in res.checks if c["name"] in EQUIVALENCE_CHECKS)
        for ch in sp.channels:
            tile_p.append(res.arrays[ch]["tile_p"].ravel())
    p = np.concatenate(tile_p)
    p = p[~np.isnan(p)]
    per_tile = {}
    ok = True
    for a in nominal:
        k = int(np.sum(p < a))
        pv = _binom_excess_p(k, p.size, a)
        per_tile[str(a)] = dict(rate=k / max(p.size, 1), nominal=a, count=k, n=int(p.size), p_excess=pv)
        ok &= pv >= excess_p
    checks = {}
    for c in REJECTION_CHECKS:
        pv = _binom_excess_p(rej[c], rej_n[c], sp.alpha_u)
        checks[c] = dict(rate=rej[c] / max(rej_n[c], 1), nominal=sp.alpha_u, count=rej[c], n=rej_n[c], p_excess=pv)
        ok &= pv >= excess_p
    return dict(n_splits=n_splits, half_size=reps.n // 2, alpha_u=sp.alpha_u, per_tile=per_tile,
                rejection_checks=checks, any_rejection_rate=rej_any / n_splits,
                tost_pass_rate=tost_pass / n_splits, gate_pass_rate=gate_pass / n_splits, ok=bool(ok))


def plant(kind: str, stack: np.ndarray, *, factor: float, region: Sequence[int] | None = None,
          channel: int | None = None) -> np.ndarray:
    """Planted-bias copy of a replicate stack (N, H, W, C) or single image (H, W, C).

    kind 'scale'   : every pixel × factor (Stage A light ×1.0075 in a one-light scene, Stage B W ×1.003)
         'region'  : region = (x0, y0, size) or (x0, y0, w, h) × factor (Stage A one 32² region +3%)
         'channel' : one colour channel × factor (a chromatic bug)
    """
    a = np.array(stack, dtype=np.float64, copy=True)
    if kind in ("scale", "global"):
        a *= factor
    elif kind == "region":
        if region is None:
            raise ValueError("region plant needs region=(x0, y0, size) or (x0, y0, w, h)")
        x0, y0, *wh = (int(v) for v in region)
        w, h = (wh[0], wh[0]) if len(wh) == 1 else wh
        a[..., y0:y0 + h, x0:x0 + w, :] *= factor
    elif kind == "channel":
        if channel is None:
            raise ValueError("channel plant needs channel=index")
        a[..., channel] *= factor
    else:
        raise ValueError(f"unknown plant kind {kind!r}")
    return a


def plant_detection(stack: np.ndarray, spec: GateSpec, plant_fn: Callable[[np.ndarray], np.ndarray],
                    n_repeats: int = 10, rng: np.random.Generator | int | None = 0,
                    masks: np.ndarray | None = None, mask_names: Sequence[str] = (),
                    required: int | None = None) -> dict:
    """δ-scale positive control: in each repeat, split the replicates into disjoint halves A/B,
    compare plant(A) vs B (must FAIL) and A vs B (control, must PASS).

    detected   ⇔ the planted gate fails in ≥ 9/10 repeats (`required`, default ceil(0.9·n_repeats));
    powered    ⇔ the unplanted control passes in ≥ the same count (an under-powered TOST "detects"
                 every plant, so detection alone proves nothing: plan §7.3 / review V8);
    calibrated ⇔ detected and powered.
    `significant_rate` counts repeats where a rejection-type test or the global Welch t rejects Δ = 0.
    """
    rng = np.random.default_rng(rng)
    required = required if required is not None else math.ceil(0.9 * n_repeats)
    sp = spec.with_(min_replicates=2)
    tiles = sorted({16, 32, 64, sp.tile})
    base = aggregate_stack(stack, sp.channels, tiles, masks, mask_names)
    planted = aggregate_stack(plant_fn(np.asarray(stack)), sp.channels, tiles, masks, mask_names)
    fails = ctrl_pass = sig = 0
    mdbs = []
    for _ in range(n_repeats):
        a, b = _halves(base.n, rng)
        r_plant = evaluate_gate(planted.subset(a), base.subset(b), sp)
        r_ctrl = evaluate_gate(base.subset(a), base.subset(b), sp)
        fails += not r_plant.passed
        ctrl_pass += r_ctrl.passed
        sig += significant(r_plant)
        mdbs.append({ch: v["global_"]["mdb"] for ch, v in r_ctrl.summary["channels"].items()})
    mdb_y = [m[sp.channels[0]] for m in mdbs]
    return dict(n_repeats=n_repeats, required=required, gate_fail_count=fails, control_pass_count=ctrl_pass,
                significant_count=sig, gate_fail_rate=fails / n_repeats, control_pass_rate=ctrl_pass / n_repeats,
                significant_rate=sig / n_repeats, mdb_global_median=float(np.median(mdb_y)),
                detected=fails >= required, powered=ctrl_pass >= required,
                calibrated=bool(fails >= required and ctrl_pass >= required))


def rendered_plant_detection(ref: Replicates, planted: Replicates, spec: GateSpec, n_repeats: int = 10,
                             rng: np.random.Generator | int | None = 0, required: int | None = None) -> dict:
    """Rendered (real) plant: the planted replicates come from a separate render of the planted scene on seeds
    disjoint from `ref` (the caller checks disjointness). In each repeat `ref` is split into disjoint halves A/B
    and a half-size subset P of `planted` is drawn; P vs B must FAIL (detected) and A vs B must PASS (powered),
    both in ≥ `required` (default ⌈0.9·n_repeats⌉) repeats — the same rule as plant_detection.
    Also reported: which gating checks failed on the planted side (equivalence vs rejection-type), the planted
    Δ (P vs B) and the control MDBs (global, tile max/median) per channel."""
    rng = np.random.default_rng(rng)
    required = required if required is not None else math.ceil(0.9 * n_repeats)
    sp = spec.with_(min_replicates=2)
    h = min(ref.n // 2, planted.n)
    fails = ctrl_pass = sig = equiv_fail = 0
    failed_names: dict[str, int] = {}
    deltas: dict[str, list[float]] = {ch: [] for ch in sp.channels}
    mdb_g: dict[str, list[float]] = {ch: [] for ch in sp.channels}
    mdb_tmax: dict[str, list[float]] = {ch: [] for ch in sp.channels}
    mdb_tmed: dict[str, list[float]] = {ch: [] for ch in sp.channels}
    for _ in range(n_repeats):
        a, b = _halves(ref.n, rng)
        p = np.sort(rng.permutation(planted.n)[:h])
        r_plant = evaluate_gate(planted.subset(p), ref.subset(b[:h]), sp)
        r_ctrl = evaluate_gate(ref.subset(a[:h]), ref.subset(b[:h]), sp)
        fails += not r_plant.passed
        ctrl_pass += r_ctrl.passed
        sig += significant(r_plant)
        equiv_fail += any(c["name"] in EQUIVALENCE_CHECKS and c["gating"] and not c["passed"] for c in r_plant.checks)
        for n in r_plant.summary["failed_checks"]:
            failed_names[n] = failed_names.get(n, 0) + 1
        for ch in sp.channels:
            deltas[ch].append(r_plant.summary["channels"][ch]["global_"]["rel"])
            cc = r_ctrl.summary["channels"][ch]
            mdb_g[ch].append(cc["global_"]["mdb"])
            mdb_tmax[ch].append(cc["tiles"]["mdb_max"])
            mdb_tmed[ch].append(cc["tiles"]["mdb_median"])
    med = lambda v: float(np.median([x for x in v if x is not None])) if any(x is not None for x in v) else None  # noqa: E731
    per_ch = {ch: dict(planted_global_rel_median=med(deltas[ch]), mdb_global_median=med(mdb_g[ch]),
                       mdb_tile_max_median=med(mdb_tmax[ch]), mdb_tile_median_median=med(mdb_tmed[ch]))
              for ch in sp.channels}
    return dict(n_repeats=n_repeats, required=required, half_size=h, n_ref=ref.n, n_planted=planted.n,
                gate_fail_count=fails, equivalence_fail_count=equiv_fail, control_pass_count=ctrl_pass,
                significant_count=sig, gate_fail_rate=fails / n_repeats, control_pass_rate=ctrl_pass / n_repeats,
                failed_checks_histogram=dict(sorted(failed_names.items(), key=lambda kv: -kv[1])),
                channels=per_ch, mdb_global_median=per_ch[sp.channels[0]]["mdb_global_median"],
                detected=fails >= required, powered=ctrl_pass >= required,
                calibrated=bool(fails >= required and ctrl_pass >= required))
