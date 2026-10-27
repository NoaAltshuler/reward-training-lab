"""Losses / objectives. A loss is any function

    loss(out, labels, R, K) -> scalar tensor

with `out` the network output on the batch ([B] if K == 2, else [B, K]),
`labels` a long tensor [B] and `R` the reward matrix [B, K].

`expected_reward` is the thesis objective: minimizing
    L(theta) = sum_n Delta(x_n)/N * sigma(-u_n) = C - V(theta)
is exactly gradient ascent on the expected reward V.
"""
import torch
import torch.nn.functional as F


def logits_of(out, K):
    """Binary score f -> logits [-f/2, f/2], so softmax gives pi(+1|x) = sigmoid(f)."""
    return torch.stack([-out / 2, out / 2], dim=1) if K == 2 else out


def margins(out, labels, K):
    """u_n = y_n f(x_n) for binary; z_y - max_{k != y} z_k for multiclass."""
    if K == 2:
        return (2 * labels - 1).to(out.dtype) * out
    z_y = out.gather(1, labels[:, None])[:, 0]
    other = out.masked_fill(F.one_hot(labels, K).bool(), float("-inf"))
    return z_y - other.max(dim=1).values


def gaps(R, labels):
    """Delta_n = R[n, y_n] - max_{k != y_n} R[n, k]."""
    r_plus = R.gather(1, labels[:, None])[:, 0]
    other = R.masked_fill(F.one_hot(labels, R.shape[1]).bool(), float("-inf"))
    return r_plus - other.max(dim=1).values


def expected_reward(out, labels, R, K):
    """Exact policy gradient: C - V(theta), V = (1/N) sum_n sum_k pi(k|x_n) R[n, k]."""
    pi = logits_of(out, K).softmax(dim=1)
    V = (pi * R).sum(dim=1).mean()
    C = R.gather(1, labels[:, None]).mean()
    return C - V


expected_reward.tail = "gap"   # -l_n'(u) ~ (Delta_n / N) e^{-u}; used by the theta-tilde diagnostic


def reinforce(samples=1, baseline="none", seed=0):
    """Sampled policy gradient: a ~ pi(.|x_n), surrogate -(R[n,a] - b) log pi(a|x_n).
    baseline: "none" | "batch_mean" | "expected"."""
    gen = torch.Generator().manual_seed(seed)

    def f(out, labels, R, K):
        logp = logits_of(out, K).log_softmax(dim=1)
        pi = logp.exp().detach()
        a = torch.multinomial(pi, samples, replacement=True, generator=gen)
        r = R.gather(1, a)
        if baseline == "batch_mean":
            r = r - r.mean()
        elif baseline == "expected":
            r = r - (pi * R).sum(dim=1, keepdim=True)
        return -(r * logp.gather(1, a)).mean()
    return f


def cross_entropy(weight_by_gap=False):
    """Logistic / softmax cross-entropy, log(1 + e^{-u}) in the binary case."""
    def f(out, labels, R, K):
        ce = F.cross_entropy(logits_of(out, K), labels, reduction="none")
        return ((gaps(R, labels) if weight_by_gap else 1.0) * ce).mean()
    f.tail = "gap" if weight_by_gap else "one"
    return f


def exponential(weight_by_gap=True):
    """Exponential loss e^{-u_n} (Soudry et al.)."""
    def f(out, labels, R, K):
        return ((gaps(R, labels) if weight_by_gap else 1.0) * torch.exp(-margins(out, labels, K))).mean()
    f.tail = "gap" if weight_by_gap else "one"
    return f


def hinge(weight_by_gap=False):
    """max(0, 1 - u_n): gradient vanishes once all margins reach 1."""
    def f(out, labels, R, K):
        return ((gaps(R, labels) if weight_by_gap else 1.0) * F.relu(1 - margins(out, labels, K))).mean()
    return f
