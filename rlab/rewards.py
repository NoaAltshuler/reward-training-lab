"""Ready-made reward functions.

A reward is either an array R [N, K] or a function (X, labels, K) -> R, where
R[n, k] is the reward for choosing class k on example n. For binary data the
thesis quantities are
    r_+(x_n) = R[n, label_n],  r_-(x_n) = R[n, other],  Delta(x_n) = r_+ - r_-.
The factories below return such functions.
"""
import numpy as np


def from_gaps(labels, K, r_plus, r_minus=0.0):
    """Correct class gets r_plus[n], every wrong class gets r_minus[n]."""
    N = len(labels)
    R = np.broadcast_to(np.asarray(r_minus, float).reshape(-1, 1), (N, K)).copy()
    R[np.arange(N), labels] = np.broadcast_to(r_plus, N)
    return R


def constant_gap(r_plus=1.0, r_minus=0.0):
    """Equal gaps (draft 1)."""
    return lambda X, labels, K: from_gaps(labels, K, r_plus, r_minus)


def random_gaps(low=0.2, high=2.0, shift=0.0, seed=1):
    """Unequal positive gaps (draft 2): Delta_n ~ U[low, high], r_- ~ U[-shift, shift]."""
    def f(X, labels, K):
        rng = np.random.default_rng(seed)
        gaps = rng.uniform(low, high, len(labels))
        r_minus = rng.uniform(-shift, shift, len(labels))
        return from_gaps(labels, K, r_minus + gaps, r_minus)
    return f


def gap_per_class(gaps=(1.0, 3.0), r_minus=0.0):
    """Gap gaps[k] for examples of class k."""
    return lambda X, labels, K: from_gaps(labels, K, r_minus + np.asarray(gaps, float)[labels], r_minus)


def gap_fn(delta):
    """Gap given by any function of the input: Delta_n = delta(x_n)."""
    return lambda X, labels, K: from_gaps(labels, K, np.asarray([delta(x) for x in X], float), 0.0)


def noisy_gaps(gap=1.0, flip_fraction=0.1, seed=2):
    """Some gaps flipped negative (violates Assumption 1)."""
    def f(X, labels, K):
        rng = np.random.default_rng(seed)
        g = np.full(len(labels), gap)
        g[rng.random(len(labels)) < flip_fraction] *= -1
        return from_gaps(labels, K, g, 0.0)
    return f
