"""Update rules. An update rule is any callable

    update(params, grads, state, t) -> None

that modifies `params` (list of tensors) in place given their gradients,
a dict `state` it may use freely, and the step index t. Set an `lr`
attribute on it to have the theory diagnostics use that step size as eta.
"""
import torch


def _with_lr(f, lr):
    f.lr = lr
    return f


def gd(lr=0.1):
    """theta_{t+1} = theta_t - eta * grad (the thesis' exact update)."""
    def f(params, grads, state, t):
        for p, g in zip(params, grads):
            p.sub_(lr * g)
    return _with_lr(f, lr)


def normalized_gd(lr=0.1, eps=1e-30):
    """theta_{t+1} = theta_t - eta * grad / ||grad|| (Nacson et al.: faster directional convergence)."""
    def f(params, grads, state, t):
        norm = torch.sqrt(sum((g ** 2).sum() for g in grads)) + eps
        for p, g in zip(params, grads):
            p.sub_(lr * g / norm)
    return _with_lr(f, lr)


def momentum(lr=0.1, beta=0.9):
    def f(params, grads, state, t):
        bufs = state.setdefault("v", [torch.zeros_like(p) for p in params])
        for p, g, v in zip(params, grads, bufs):
            v.mul_(beta).add_(g)
            p.sub_(lr * v)
    return _with_lr(f, lr)


def torch_optimizer(cls, **kwargs):
    """Wrap any torch.optim optimizer, e.g. torch_optimizer(torch.optim.Adam, lr=1e-2)."""
    def f(params, grads, state, t):
        if "opt" not in state:
            state["opt"] = cls(params, **kwargs)
        for p, g in zip(params, grads):
            p.grad = g
        state["opt"].step()
    return _with_lr(f, kwargs.get("lr"))
