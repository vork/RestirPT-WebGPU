"""Shared synthetic-data helpers for the compare.py / stats.py tests (known ground truth)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

PARETO_A = 2.5          # finite variance, infinite third moment
PARETO_W = 0.05         # mixture weight of the Pareto component
_P_MEAN = (1 - PARETO_W) + PARETO_W * PARETO_A / (PARETO_A - 1)


def heavy_noise(rng: np.random.Generator, shape, scale: float = 1.0, sigma: float = 0.6) -> np.ndarray:
    """Mean-1, non-negative, heavy-tailed multiplicative noise: lognormal × (Pareto mixture).

    x = 1 + scale·(ξ − 1) with E[ξ] = 1 exactly, so the estimator stays unbiased for scale ≤ 1.
    """
    ln = rng.lognormal(-0.5 * sigma ** 2, sigma, shape)
    par = np.where(rng.random(shape) < PARETO_W, 1.0 + rng.pareto(PARETO_A, shape), 1.0) / _P_MEAN
    xi = ln * par
    return 1.0 + scale * (xi - 1.0)


def base_image(h: int = 128, w: int = 128) -> np.ndarray:
    """Asymmetric RGB ground truth with a dark block (absolute-margin tiles) and a black block
    (exact zero, zero-variance tiles). Asymmetric so any row/column flip changes it."""
    yy, xx = np.mgrid[0:h, 0:w]
    r = 0.3 + 1.5 * xx / w + 0.4 * (yy / h) ** 2
    g = 0.5 + 0.8 * yy / h
    b = 0.2 + 0.6 * (xx * yy) / (w * h)
    img = np.stack([r, g, b], -1)
    img[:32, :32] = 0.0                    # black: zero variance in both engines
    img[h - 32:, :32] *= 0.01              # dark: R̄_tile < 0.05·R̄_image
    return img


def replicate_stack(rng, n: int, base: np.ndarray, scale: float, sigma: float = 0.6) -> np.ndarray:
    return base[None] * heavy_noise(rng, (n,) + base.shape, scale, sigma)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
