"""Ready-made networks. Any torch.nn.Module works as long as it maps
X [N, d] to
  scores [N]     when K == 2  (policy pi(y=+1|x) = sigmoid(f(x)))
  logits [N, K]  when K > 2   (policy pi(k|x)    = softmax(z(x))_k)

Linear models expose `features(X)` and a single parameter `theta`
(shape [D] for binary, [K, D] for multiclass). The theory diagnostics
(max-margin direction, theta-tilde, residual) are computed for these.
"""
import torch
from torch import nn


class Linear(nn.Module):
    """f(x) = <phi(x), theta>, phi(x) = x (thesis model) or (x, 1) with bias=True."""

    def __init__(self, d_in=2, K=2, bias=False, init="zeros", init_scale=0.1):
        super().__init__()
        self.bias = bias
        D = d_in + int(bias)
        shape = (D,) if K == 2 else (K, D)
        self.theta = nn.Parameter(torch.zeros(shape) if init == "zeros" else init_scale * torch.randn(shape))

    def features(self, X):
        return torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], dim=1) if self.bias else X

    def forward(self, X):
        return self.features(X) @ (self.theta if self.theta.dim() == 1 else self.theta.T)


ACTS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "leaky_relu": nn.LeakyReLU, "softplus": nn.Softplus}


class MLP(nn.Module):
    """Fully connected network. With bias=False and ReLU it is positively homogeneous."""

    def __init__(self, d_in=2, K=2, width=32, depth=2, activation="relu", bias=True, init_scale=1.0):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width, bias=bias), ACTS[activation]()]
            d = width
        layers.append(nn.Linear(d, 1 if K == 2 else K, bias=bias))
        self.net = nn.Sequential(*layers)
        self.K = K
        with torch.no_grad():
            for p in self.parameters():
                p.mul_(init_scale)

    def forward(self, X):
        out = self.net(X)
        return out[:, 0] if self.K == 2 else out
