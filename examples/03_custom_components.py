"""Everything is pluggable: bring your own data, reward, network, loss and update.

    python examples/03_custom_components.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn

import rlab
from rlab.losses import logits_of

# 1. data: any arrays X [N, 2], labels [N]
rng = np.random.default_rng(0)
X = rng.uniform(-2, 2, (80, 2))
labels = (X[:, 0] ** 2 + X[:, 1] ** 2 < 1.6).astype(int)


# 2. reward: R[n, k] = reward for answering class k on example n
def reward(X, labels, K):
    R = np.zeros((len(X), K))
    R[np.arange(len(X)), labels] = 1.0 + np.linalg.norm(X, axis=1)   # far points matter more
    return R


# 3. network: any nn.Module, X [N, 2] -> scores [N] (binary) or logits [N, K]
class TwoLayer(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.hidden = nn.Linear(2, width)
        self.out = nn.Linear(width, 1, bias=False)

    def forward(self, X):
        return self.out(torch.relu(self.hidden(X)))[:, 0]


# 4. loss: (out, labels, R, K) -> scalar. Here: expected reward with an entropy bonus.
def entropy_regularized(out, labels, R, K, beta=0.01):
    logp = logits_of(out, K).log_softmax(1)
    pi = logp.exp()
    V = (pi * R).sum(1).mean()
    H = -(pi * logp).sum(1).mean()
    return -(V + beta * H)


# 5. update rule: (params, grads, state, t) -> None, in place
def sign_gd(params, grads, state, t, lr=0.01):
    for p, g in zip(params, grads):
        p.sub_(lr * g.sign())


sign_gd.lr = 0.01

exp = rlab.Experiment(X, labels, reward=reward, model=TwoLayer(), loss=entropy_regularized, update=sign_gd,
                      name="custom: 2-layer ReLU, sign-GD")
exp.train(steps=20_000, live=True)
