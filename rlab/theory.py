"""Max-margin quantities from the thesis, for linear models.

Binary (theta in R^D, signed features z_n = y_n phi(x_n)):
    theta_hat = argmin 1/2 ||theta||^2  s.t.  <z_n, theta> >= 1
    S         = support set,  theta_hat = sum_{n in S} alpha_n z_n
    theta_til : <z_n, theta_til> = log(eta * c_n / alpha_n),  n in S
                (c_n = Delta_n / N for the expected-reward loss)
    prediction: theta_t ~ theta_hat log t + theta_til
Multiclass (W in R^{K x D}):
    W_hat = argmin 1/2 ||W||^2  s.t.  <w_{y_n} - w_k, phi_n> >= 1  for k != y_n
"""
import numpy as np
from scipy.optimize import linprog, minimize


def is_separable(Z):
    """Is there theta with Z theta >= 1 (Z rows are signed features)?"""
    res = linprog(np.zeros(Z.shape[1]), A_ub=-Z, b_ub=-np.ones(len(Z)), bounds=[(None, None)] * Z.shape[1],
                  method="highs")
    return res.status == 0


def hard_margin_binary(Z, tol=1e-7):
    """Returns (theta_hat, S, alpha) or None if not separable."""
    if not is_separable(Z):
        return None
    D = Z.shape[1]
    res = minimize(lambda t: 0.5 * t @ t, np.zeros(D), jac=lambda t: t, method="SLSQP",
                   constraints=[{"type": "ineq", "fun": lambda t: Z @ t - 1, "jac": lambda t: Z}],
                   options={"ftol": 1e-14, "maxiter": 1000})
    theta = res.x
    # Refine on the active set: solve Z_S Z_S^T alpha = 1 exactly.
    for slack in (1e-6, 1e-5, 1e-4, 1e-3):
        S = np.where(Z @ theta <= 1 + slack)[0]
        ZS = Z[S]
        alpha, *_ = np.linalg.lstsq(ZS @ ZS.T, np.ones(len(S)), rcond=None)
        cand = ZS.T @ alpha
        if np.all(alpha > -tol) and np.all(Z @ cand >= 1 - 1e-6):
            return cand, S, alpha
    return theta, np.where(Z @ theta <= 1 + 1e-4)[0], None


def theta_tilde(Z, S, alpha, eta, c):
    """Least-norm solution of <z_n, theta_til> = log(eta c_n / alpha_n), n in S."""
    if alpha is None or eta is None or np.any(alpha <= 0):
        return None
    rhs = np.log(eta * c[S] / alpha)
    sol, *_ = np.linalg.lstsq(Z[S], rhs, rcond=None)
    return sol


def hard_margin_multiclass(Phi, labels, K):
    N, D = Phi.shape
    rows = []
    for n in range(N):
        for k in range(K):
            if k != labels[n]:
                a = np.zeros((K, D))
                a[labels[n]] = Phi[n]
                a[k] = -Phi[n]
                rows.append(a.ravel())
    A = np.array(rows)
    if not is_separable(A):
        return None
    res = minimize(lambda w: 0.5 * w @ w, np.zeros(K * D), jac=lambda w: w, method="SLSQP",
                   constraints=[{"type": "ineq", "fun": lambda w: A @ w - 1, "jac": lambda w: A}],
                   options={"ftol": 1e-14, "maxiter": 2000})
    return res.x.reshape(K, D)


def analyse(Phi, labels, K, gaps, eta, tail):
    """Static theory for a linear model. Returns a JSON-able dict."""
    N = len(labels)
    out = {"N": N, "gap_min": float(gaps.min()), "gap_max": float(gaps.max()),
           "positive_gaps": bool(np.all(gaps > 0))}
    if K == 2:
        Z = (2 * labels - 1)[:, None] * Phi
        smax = np.linalg.svd(Z, compute_uv=False)[0]
        if gaps.max() > 0:
            out["eta_bound"] = float(8 * N / (gaps.max() * smax ** 2))
        hm = hard_margin_binary(Z)
        out["separable"] = hm is not None
        if hm is None:
            return out
        th, S, alpha = hm
        out.update(theta_hat=th.tolist(), support=S.tolist(), margin=float(1 / np.linalg.norm(th)),
                   alpha=None if alpha is None else alpha.tolist())
        c = {"gap": gaps / N, "one": np.full(N, 1.0 / N)}.get(tail)
        if c is not None and np.all(c > 0):
            tt = theta_tilde(Z, S, alpha, eta, c)
            if tt is not None:
                out["theta_tilde"] = tt.tolist()
    else:
        W = hard_margin_multiclass(Phi, labels, K)
        out["separable"] = W is not None
        if W is not None:
            out["theta_hat"] = W.tolist()
            out["margin"] = float(1 / np.linalg.norm(W))
    return out
