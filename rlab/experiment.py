"""Experiment: data + reward + network + loss + update rule, trained while
recording the weights, gradients and per-sample pushes at log-spaced steps.

    exp = Experiment(X, labels, reward=rewards.random_gaps(), model="linear",
                     loss=losses.expected_reward, update=updates.gd(0.5))
    exp.train(steps=100_000, live=True)     # opens the viewer while training
"""
import sys
import time

import numpy as np
import torch
from torch import nn

from rlab import losses as L
from rlab import models, theory, updates

LOSSES = {"expected_reward": L.expected_reward, "reinforce": L.reinforce(), "cross_entropy": L.cross_entropy(),
          "exponential": L.exponential(), "hinge": L.hinge()}


def _round(a, sig=6):
    return [float(f"{v:.{sig}g}") for v in np.asarray(a, float).ravel()]


def _flat(tensors):
    return torch.cat([t.detach().reshape(-1) for t in tensors])


class Experiment:
    def __init__(self, X, labels, reward, model="linear", loss=L.expected_reward, update=None, batch_size=None,
                 K=None, name=None, seed=0, grid_res=50, bounds=None, max_param_record=5000):
        """
        X, labels   inputs [N, 2] and class indices [N] (class 1 = y=+1 when binary)
        reward      R array [N, K] or function (X, labels, K) -> R
        model       "linear" | "linear_bias" | "mlp" | any nn.Module (see rlab.models)
        loss        name in LOSSES or function (out, labels, R, K) -> scalar
        update      update rule (rlab.updates), default gd(lr=0.1)
        batch_size  None = full batch (exact gradient), else minibatch size
        """
        torch.manual_seed(seed)
        self.X = np.asarray(X, float)
        self.labels = np.asarray(labels, int)
        self.N, self.d = self.X.shape
        self.K = int(K or self.labels.max() + 1)
        self.R = np.asarray(reward(self.X, self.labels, self.K) if callable(reward) else reward, float)
        assert self.R.shape == (self.N, self.K), f"reward must be [N, K] = {(self.N, self.K)}, got {self.R.shape}"

        if isinstance(model, str):
            model = {"linear": lambda: models.Linear(self.d, self.K),
                     "linear_bias": lambda: models.Linear(self.d, self.K, bias=True),
                     "mlp": lambda: models.MLP(self.d, self.K)}[model]()
        self.model = model.double()
        self.loss = LOSSES[loss] if isinstance(loss, str) else loss
        self.update = update or updates.gd(0.1)
        self.batch_size = batch_size
        self.name = name or f"{type(self.model).__name__} · {getattr(self.loss, '__qualname__', 'loss').split('.')[0]}"
        self.seed = seed
        self.max_param_record = max_param_record

        self.Xt = torch.tensor(self.X)
        self.yt = torch.tensor(self.labels)
        self.Rt = torch.tensor(self.R)
        self.gaps = L.gaps(self.Rt, self.yt).numpy()

        lo, hi = self.X.min(0), self.X.max(0)
        pad = 0.25 * (hi - lo).max() + 0.3
        if bounds is None:
            c, h = (lo + hi) / 2, (hi - lo).max() / 2 + pad
            bounds = [c[0] - h, c[0] + h, c[1] - h, c[1] + h]
            # keep the origin in view for models without bias: the separator passes through it
            bounds = [min(bounds[0], -0.5), max(bounds[1], 0.5), min(bounds[2], -0.5), max(bounds[3], 0.5)]
        self.bounds = [float(b) for b in bounds]
        gx = np.linspace(self.bounds[0], self.bounds[1], grid_res)
        gy = np.linspace(self.bounds[2], self.bounds[3], grid_res)
        self.grid_res = grid_res
        self.grid = torch.tensor(np.array([[x, y] for y in gy for x in gx]))

        self.is_linear = isinstance(self.model, models.Linear)
        self.theory = None
        if self.is_linear:
            Phi = self.model.features(self.Xt).numpy()
            self.theory = theory.analyse(Phi, self.labels, self.K, self.gaps, getattr(self.update, "lr", None),
                                         getattr(self.loss, "tail", None))
        self.snapshots = []
        self.status = "ready"
        self.steps = 0
        self.progress = 0

    # ------------------------------------------------------------------ training
    def train(self, steps=10_000, snapshots=300, log_spacing=True, live=False, port=8765, wait=None, verbose=True):
        """Run the update rule for `steps` steps, recording ~`snapshots` frames.
        live=True opens the viewer in the browser and updates it while training."""
        if log_spacing:
            marks = np.unique(np.round(np.geomspace(1, steps, snapshots)).astype(int))
        else:
            marks = np.unique(np.linspace(0, steps, snapshots).astype(int))
        marks = set(int(m) for m in marks) | {0, steps}
        self.steps, self.status, self.snapshots = steps, "running", []
        if live:
            from rlab import viewer
            viewer.serve(self, port=port)

        params = [p for p in self.model.parameters() if p.requires_grad]
        state, rng = {}, np.random.default_rng(self.seed)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)    # tiny tensors: thread overhead dominates
        t0 = time.time()
        for t in range(steps + 1):
            snap = self._record(t, params) if t in marks else None
            if t == steps:
                self.snapshots.append(snap)
                break
            if self.batch_size:
                idx = torch.tensor(rng.choice(self.N, self.batch_size, replace=False))
                Xb, yb, Rb = self.Xt[idx], self.yt[idx], self.Rt[idx]
            else:
                Xb, yb, Rb = self.Xt, self.yt, self.Rt
            lossv = self.loss(self.model(Xb), yb, Rb, self.K)
            grads = torch.autograd.grad(lossv, params)
            before = _flat(params) if snap is not None else None
            with torch.no_grad():
                self.update(params, grads, state, t)
            if snap is not None:
                snap["update"] = _round(_flat(params) - before)
                self.snapshots.append(snap)   # append only once complete: the live viewer reads this list
            self.progress = t + 1
            if verbose and (t + 1) % max(1, steps // 10) == 0:
                print(f"[{self.name}] step {t + 1}/{steps}  loss {lossv.item():.4g}  ({time.time() - t0:.1f}s)",
                      file=sys.stderr)
        torch.set_num_threads(threads)
        self.status, self.progress = "done", steps

        if live:
            from rlab import viewer
            interactive = hasattr(sys, "ps1") or "ipykernel" in sys.modules
            if wait if wait is not None else not interactive:
                viewer.block()
        return self

    def _record(self, t, params):
        Xt, yt, Rt, K = self.Xt, self.yt, self.Rt, self.K
        out = self.model(Xt)
        lossv = self.loss(out, yt, Rt, K)
        dout, *grads = torch.autograd.grad(lossv, [out] + params, allow_unused=True)
        grads = [torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)]
        with torch.no_grad():
            pi = L.logits_of(out, K).softmax(1)
            V = (pi * Rt).sum(1).mean().item()
            C = Rt.gather(1, yt[:, None]).mean().item()
            u = L.margins(out, yt, K)
            acc = (pi.argmax(1) == yt).double().mean().item()
            flat = _flat(params)
            snap = {"t": t, "loss": lossv.item(), "V": V, "regret": C - V, "acc": acc,
                    "min_margin": u.min().item(), "param_norm": flat.norm().item(),
                    "grad_norm": _flat(grads).norm().item(), "margins": _round(u)}
            if K == 2:
                snap["push"] = _round(-(2 * yt - 1) * dout)          # >0: sample pushes its margin up
            else:
                snap["push"] = _round(dout.norm(dim=1))

            named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
            if flat.numel() <= self.max_param_record:
                snap["params"] = {n: _round(p) for n, p in named}
                snap["neg_grad"] = {n: _round(-g) for (n, _), g in zip(named, grads)}
            snap["layer_norms"] = {n: p.norm().item() for n, p in named}

            g = self.model(self.grid)
            if K == 2:
                snap["grid"] = _round(torch.sigmoid(g), 4)
            else:
                pg = g.softmax(1)
                snap["grid_class"] = pg.argmax(1).tolist()
                snap["grid"] = _round(pg.max(1).values, 3)

            if self.is_linear:
                phi = self.model.features(Xt)
                contrib = -dout[:, None] * phi if K == 2 else -(dout[:, :, None] * phi[:, None, :]).reshape(len(phi), -1)
                snap["contrib"] = [_round(c) for c in contrib]
                snap.update(self._theory_metrics(t, flat.numpy(), u.numpy(), lossv.item()))
            else:
                snap["neurons"] = self._neurons()
        return snap

    def _theory_metrics(self, t, theta, u, lossv):
        th = self.theory or {}
        m = {"theta": _round(theta)}
        n = np.linalg.norm(theta)
        if n > 0:
            m["norm_margin"] = float(u.min() / n)
        if "theta_hat" in th:
            hat = np.ravel(th["theta_hat"])
            if n > 0:
                cos = np.clip(theta @ hat / (n * np.linalg.norm(hat)), -1, 1)
                m["angle_deg"] = float(np.degrees(np.arccos(cos)))
                m["one_minus_cos"] = float(1 - cos)
            if t >= 1 and "theta_tilde" in th:
                m["residual"] = float(np.linalg.norm(theta - hat * np.log(t) - np.asarray(th["theta_tilde"])))
        if t >= 1:
            m["t_loss"] = t * lossv
        return m

    def _neurons(self):
        """First hidden layer as 2D directions: (w_j, b_j, ||outgoing weights of j||)."""
        lin = [m for m in self.model.modules() if isinstance(m, nn.Linear)]
        if len(lin) < 2 or lin[0].in_features != 2:
            return None
        W, b = lin[0].weight, lin[0].bias
        a = lin[1].weight.norm(dim=0)
        return {"w": [_round(w) for w in W], "b": None if b is None else _round(b), "a": _round(a)}

    # ------------------------------------------------------------------ export
    def to_dict(self):
        return {"name": self.name, "status": self.status, "steps": self.steps, "progress": self.progress,
                "K": self.K, "N": self.N, "X": self.X.tolist(), "labels": self.labels.tolist(),
                "R": self.R.tolist(), "gaps": self.gaps.tolist(), "bounds": self.bounds, "grid_res": self.grid_res,
                "linear": self.is_linear, "bias": bool(getattr(self.model, "bias", False)),
                "lr": getattr(self.update, "lr", None), "batch_size": self.batch_size,
                "theory": self.theory, "snapshots": list(self.snapshots)}

    def save_html(self, path):
        from rlab import viewer
        return viewer.save_html(path, self)
