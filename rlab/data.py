"""Ready-made 2D datasets. Each returns (X [N, 2], labels [N]) with class
indices 0..K-1. For binary data class 1 is y=+1 and class 0 is y=-1.

You can always pass your own arrays to Experiment instead.
"""
import numpy as np


def separable_blobs(n=40, angle_deg=30.0, separation=2.0, spread=0.6, min_margin=0.15, seed=0):
    """Two Gaussian blobs on either side of a line through the origin."""
    rng = np.random.default_rng(seed)
    a = np.deg2rad(angle_deg)
    v = np.array([np.cos(a), np.sin(a)])
    X, y = [], []
    while len(X) < n:
        lab = len(X) % 2
        s = 1 if lab == 1 else -1
        x = s * separation * v + spread * rng.standard_normal(2)
        if s * x @ v >= min_margin:
            X.append(x)
            y.append(lab)
    return np.array(X), np.array(y)


def blobs_offset(n=40, center=(2.0, 1.5), gap=1.2, spread=0.4, seed=0):
    """Separable blobs whose separator does not pass through the origin (needs a bias)."""
    rng = np.random.default_rng(seed)
    c = np.asarray(center, float)
    X = np.concatenate([c + [-gap, 0] + spread * rng.standard_normal((n // 2, 2)),
                        c + [gap, 0] + spread * rng.standard_normal((n - n // 2, 2))])
    return X, np.array([0] * (n // 2) + [1] * (n - n // 2))


def overlapping_blobs(n=60, separation=0.8, spread=1.0, seed=0):
    """Two overlapping Gaussians (not separable)."""
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 2
    X = spread * rng.standard_normal((n, 2))
    X[:, 0] += np.where(y == 1, separation, -separation)
    return X, y


def xor(n=80, spread=0.35, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]], float)
    idx = np.arange(n) % 4
    return centers[idx] + spread * rng.standard_normal((n, 2)), (idx < 2).astype(int)


def moons(n=100, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    n1 = n // 2
    t1, t2 = np.pi * rng.random(n1), np.pi * rng.random(n - n1)
    X = np.concatenate([np.c_[np.cos(t1), np.sin(t1)], np.c_[1 - np.cos(t2), 0.5 - np.sin(t2)]])
    X = (X - [0.5, 0.25]) * 1.6 + noise * rng.standard_normal((n, 2))
    return X, np.array([0] * n1 + [1] * (n - n1))


def circles(n=100, noise=0.08, ratio=0.45, seed=0):
    rng = np.random.default_rng(seed)
    t = 2 * np.pi * rng.random(n)
    y = np.arange(n) % 2
    r = np.where(y == 1, ratio, 1.0) * 1.8
    return np.c_[r * np.cos(t), r * np.sin(t)] + noise * rng.standard_normal((n, 2)), y


def multiclass_blobs(n=60, K=3, radius=2.0, spread=0.45, seed=0):
    """K Gaussian groups arranged on a circle."""
    rng = np.random.default_rng(seed)
    y = np.arange(n) % K
    ang = 2 * np.pi * y / K + 0.3
    return radius * np.c_[np.cos(ang), np.sin(ang)] + spread * rng.standard_normal((n, 2)), y
