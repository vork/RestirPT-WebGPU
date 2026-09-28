"""Static PNG figures for compare.py reports (matplotlib, Agg backend).

Colour roles follow the dataviz reference palette: diverging maps are blue <-> red around a neutral
grey midpoint, magnitude maps are one-hue (blue) light -> dark, line series use categorical slots
1-2 (blue, orange) in fixed order, text stays in ink colours.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy import stats as st  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#e34948"
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a32626"]).with_extremes(bad="#bdbcb8")
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]).with_extremes(bad="#bdbcb8")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
})


def _save(fig, path: Path) -> str:
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return Path(path).name


def _img_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def tonemap(rgb: np.ndarray, exposure: float) -> np.ndarray:
    """Linear -> display: exposure scale, clip, sRGB OETF (identical for both images)."""
    x = np.clip(np.asarray(rgb, dtype=np.float64) * exposure, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def exposure_for(ref_rgb: np.ndarray) -> float:
    y = np.asarray(ref_rgb, dtype=np.float64)
    y = y[..., :3] @ np.array([0.2126, 0.7152, 0.0722]) if y.shape[-1] >= 3 else y[..., 0]
    p = float(np.percentile(y, 99.5))
    return 1.0 / p if p > 0 else 1.0


def side_by_side(ref_rgb, ours_rgb, path: Path, labels=("reference", "ours")) -> str:
    e = exposure_for(ref_rgb)
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.6))
    for ax, img, lab in zip(axs, (ref_rgb, ours_rgb), labels):
        im = img if img.shape[-1] >= 3 else np.repeat(img, 3, -1)
        ax.imshow(tonemap(im[..., :3], e), interpolation="nearest")
        ax.set_title(lab)
        _img_axes(ax)
    fig.suptitle(f"Converged means, identical exposure (x{e:.3g}, sRGB)", color=INK2, fontsize=9)
    return _save(fig, path)


def rel_diff(rel: np.ndarray, delta: float, path: Path, title: str) -> str:
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(np.clip(rel, -delta, delta), cmap=DIVERGING, vmin=-delta, vmax=delta, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("(ours - ref) / ref, clipped to +/-delta")
    cb.outline.set_visible(False)
    ax.set_title(title)
    _img_axes(ax)
    return _save(fig, path)


def t_map(abs_t: np.ndarray, sidak_t: float, bh_mask: np.ndarray | None, path: Path, title: str) -> str:
    n = 2 if bh_mask is not None else 1
    fig, axs = plt.subplots(1, n, figsize=(5.6 * n, 4.8))
    axs = np.atleast_1d(axs)
    a = np.clip(abs_t, 0, 6)
    im = axs[0].imshow(a, cmap=SEQUENTIAL, vmin=0, vmax=6, interpolation="nearest")
    full = np.nan_to_num(np.asarray(abs_t, dtype=float), posinf=1e6)
    if np.isfinite(sidak_t) and full.max() >= sidak_t:  # contour on the unclipped |t|
        axs[0].contour(full, levels=[sidak_t], colors=[ORANGE], linewidths=1.0)
    axs[0].set_title(f"{title}: |t| per pixel (clip 6), Sidak |t*|={sidak_t:.2f} contour")
    cb = fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    _img_axes(axs[0])
    if bh_mask is not None:
        axs[1].imshow(bh_mask.astype(float), cmap=SEQUENTIAL, vmin=0, vmax=1, interpolation="nearest")
        axs[1].set_title(f"BH-FDR q<=0.01 pixels ({int(bh_mask.sum())})")
        _img_axes(axs[1])
    return _save(fig, path)


def tile_t(t: np.ndarray, passed: np.ndarray, sidak: np.ndarray, tile: int, path: Path, title: str) -> str:
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    lim = 6.0  # same clip as the |t| map; larger |t| saturate
    tt = np.clip(np.where(np.isnan(t), 0.0, t), -lim, lim)
    im = ax.imshow(tt, cmap=DIVERGING, vmin=-lim, vmax=lim, interpolation="nearest")
    ys, xs = np.nonzero(~passed)
    ax.scatter(xs, ys, marker="x", s=36, c=INK, linewidths=1.2, label="TOST fail")
    ys, xs = np.nonzero(sidak)
    ax.scatter(xs, ys, marker="s", s=80, facecolors="none", edgecolors=ORANGE, linewidths=1.5, label="Sidak reject")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("tile Welch t")
    cb.outline.set_visible(False)
    ax.set_title(f"{title}: {tile}x{tile} tile t (clip +/-6)")
    _img_axes(ax)
    if (~passed).any() or sidak.any():
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=8)
    return _save(fig, path)


def flip_map(emap: np.ndarray, mean: float, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(emap, cmap=SEQUENTIAL, vmin=0, vmax=1, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    ax.set_title(f"HDR-FLIP error map (mean {mean:.4f})")
    _img_axes(ax)
    return _save(fig, path)


def z_hist_qq(t: np.ndarray, nu: np.ndarray, path: Path, title: str) -> str:
    ok = np.isfinite(t) & np.isfinite(nu)
    t, nu = t[ok], nu[ok]
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4))
    if t.size == 0:
        for ax in axs:
            ax.text(0.5, 0.5, "no finite t", ha="center", transform=ax.transAxes)
        return _save(fig, path)
    nm = float(np.median(nu))
    lim = 6.0
    n_out = int(np.sum(np.abs(t) > lim))
    axs[0].hist(np.clip(t, -lim, lim), bins=max(10, min(60, t.size // 6)), range=(-lim, lim), density=True,
                color=BLUE, edgecolor=SURFACE, linewidth=0.8, label="tile t" + (f" ({n_out} beyond +/-6, piled at edge)" if n_out else ""))
    xs = np.linspace(-lim, lim, 400)
    axs[0].plot(xs, st.t.pdf(xs, nm), color=ORANGE, lw=2, label=f"t(nu={nm:.1f})")
    axs[0].set_title(f"{title}: tile t histogram")
    axs[0].legend(frameon=False)
    axs[0].grid(axis="y", color=GRID, lw=0.6)
    # QQ against each tile's own t(nu_A) via the probability-integral transform
    u = np.sort(st.t.cdf(t, nu))
    q_emp = st.t.ppf(np.clip(u, 1e-12, 1 - 1e-12), nm)
    q_th = st.t.ppf((np.arange(1, t.size + 1) - 0.5) / t.size, nm)
    axs[1].plot([q_th[0], q_th[-1]], [q_th[0], q_th[-1]], color=INK2, lw=1, ls="--")
    axs[1].scatter(q_th, q_emp, s=10, color=BLUE)
    axs[1].set_xlabel(f"theoretical quantile, t({nm:.1f})")
    axs[1].set_ylabel("observed quantile")
    axs[1].set_title("QQ vs t(nu)")
    axs[1].grid(color=GRID, lw=0.6)
    return _save(fig, path)


def curve(ns, relmse, nmse, bnr, slope: dict, path: Path) -> str:
    ns = np.asarray(ns, dtype=float)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    r = np.asarray(relmse, dtype=float)
    pos = r > 0
    axs[0].loglog(ns[pos], r[pos], "o-", color=BLUE, lw=2, ms=6, label="relMSE_corr")
    if slope.get("slope") is not None:
        fit = np.exp(slope["intercept"]) * ns ** slope["slope"]
        axs[0].loglog(ns, fit, color=ORANGE, lw=1.5, ls="--", label=f"fit slope {slope['slope']:.3f}")
    axs[0].set_xlabel("N (batches)")
    axs[0].set_title("relMSE_corr(N)")
    axs[0].legend(frameon=False)
    axs[1].semilogx(ns, nmse, "o-", color=BLUE, lw=2, ms=6)
    axs[1].set_xlabel("N (batches)")
    axs[1].set_title("N * MSE_corr(N)  (flat iff unbiased)")
    axs[2].semilogx(ns, bnr, "o-", color=BLUE, lw=2, ms=6)
    axs[2].axhline(1.0, color=INK2, lw=1, ls="--")
    axs[2].set_xlabel("N (batches)")
    axs[2].set_title("BNR(N) = |mean delta| / SE (global Y)")
    for ax in axs:
        ax.grid(color=GRID, lw=0.6)
    return _save(fig, path)
