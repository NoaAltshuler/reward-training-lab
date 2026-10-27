# Reward Training Lab

**See where the gradient takes your weights.** A small PyTorch lab for visualizing how policy-gradient training on
rewards moves a model's weights. It was built to study the **implicit bias of reward-trained models**: on separable data,
exact policy gradient converges in direction to the max-margin separator, even when examples have unequal reward gaps.

You define the experiment in Python: data, labels, reward function, network, loss and update rule. A live browser
viewer then shows the weights moving during training, with the theory overlaid: the max-margin direction θ̂, the
predicted trajectory θ̂ log t + θ̃, and the residual that the theorem says stays bounded.

## The setting

For data (xₙ, yₙ) with yₙ ∈ {−1, +1}, consider the policy π_θ(y|x) = σ(y⟨x,θ⟩). Choosing the right answer gives reward
r₊(xₙ) and the wrong one gives r₋(xₙ), with gap Δ(xₙ) = r₊ − r₋ > 0. Gradient ascent on the expected reward is
gradient descent on

    L(θ) = Σₙ (Δ(xₙ)/N) · σ(−yₙ⟨xₙ,θ⟩)

With separable data and a small enough step size, θ_t = θ̂ log t + θ̃ + r(t), where r(t) stays bounded. So
θ_t/‖θ_t‖ → θ̂/‖θ̂‖ at a rate of O(1/log t). The gaps change the offset θ̃ but not the limit direction. The lab lets you
watch each part of this statement happen. It also lets you break the assumptions (negative gaps, non-separable data,
large steps, other losses, non-linear networks) and see what changes.

## Quick start

```bash
git clone <this repo> && cd traning-lab
pip install -r requirements.txt          # torch, numpy, scipy
python examples/01_reward_gaps.py        # equal vs unequal reward gaps; opens the viewer in your browser
```

| Example | What it shows |
|---|---|
| `examples/01_reward_gaps.py` | Linear policy, equal vs unequal gaps, side by side |
| `examples/02_groups.py` | K = 3 groups with a softmax policy and per-class gaps |
| `examples/03_custom_components.py` | Your own reward, network, loss and update rule |

## Defining an experiment

```python
import rlab
from rlab import data, rewards, losses, updates

X, labels = data.separable_blobs(n=30)          # or your own arrays: X [N, 2], labels [N] in 0..K-1

exp = rlab.Experiment(
    X, labels,
    reward=rewards.random_gaps(0.2, 2.0),       # R [N, K] array or f(X, labels, K) -> R
    model="linear",                             # "linear" | "linear_bias" | "mlp" | any nn.Module
    loss=losses.expected_reward,                # f(out, labels, R, K) -> scalar
    update=updates.gd(lr=0.5),                  # f(params, grads, state, t) -> None (in place)
    batch_size=None,                            # None = exact full-batch gradient
)
exp.train(steps=200_000, live=True)             # opens the viewer and updates it while training
exp.save_html("runs/my_run.html")               # self-contained file you can share
```

To compare runs in one viewer, use `rlab.show(exp_a, exp_b)` or `rlab.save_html(path, exp_a, exp_b)`.

Every component is a plain Python object you can replace:

| Piece | Contract | Built-ins |
|---|---|---|
| data | `X [N,2]`, `labels [N]` (binary: class 1 = y=+1) | `rlab.data`: separable_blobs, blobs_offset, overlapping_blobs, xor, moons, circles, multiclass_blobs |
| reward | `R[n,k]` = reward for choosing class k on example n | `rlab.rewards`: constant_gap, random_gaps, gap_per_class, gap_fn, noisy_gaps |
| network | `X → scores [N]` (K=2) or `logits [N,K]` | `rlab.models`: Linear(bias=…), MLP(width, depth, activation, bias) |
| loss | `(out, labels, R, K) → scalar` | `rlab.losses`: expected_reward, reinforce, cross_entropy, exponential, hinge |
| update | `(params, grads, state, t) → None` | `rlab.updates`: gd, normalized_gd, momentum, torch_optimizer(cls, …) |

For a custom loss, setting `my_loss.tail = "gap"` (or `"one"`) tells the lab that −ℓₙ′(u) ≈ cₙe^{−u}. The θ̃ and
residual diagnostics are then computed for it.

## The viewer

**Input space.** This panel shows:
- the policy heatmap
- the current separator (solid line) and the max-margin separator with its margin band (dashed)
- point size proportional to the gap Δₙ, with support vectors ringed
- a green halo for each sample's gradient push |∂L/∂f(xₙ)|; late in training only the support vectors keep it
- green arrows for each sample's pull on θ, which add up to −∇L

For groups it shows class regions instead. For MLPs it also draws the first-layer neurons.

**Weight space.** This panel shows the trajectory θ_t, the θ̂ direction, and the predicted curve θ̂ log t + θ̃. The
distance between the current iterate and its prediction is exactly r(t). It also draws the −∇L arrow, which ends up
parallel to θ̂ rather than to θ_t. A toggle shows θ/‖θ‖ on the unit circle instead. For multiclass models there is one
path per class vector, and for MLPs one path per neuron.

**Training dynamics.** These charts use a log-t axis. Click a chart to jump to that time.

| Chart | Theory |
|---|---|
| Loss | Θ(1/t), slope −1 on log–log |
| Angle ∠(θ_t, θ̂) | O(1/log t), very slow |
| ‖θ_t‖ | parallel to ‖θ̂‖ log t |
| Residual ‖θ_t − θ̂ log t − θ̃‖ | O(1) |
| t · L(θ_t) | flat, → ‖θ̂‖²/η at leading order |
| Normalized margin | rises to 1/‖θ̂‖ |
| Weights θᵢ(t) | slope θ̂ᵢ, intercept θ̃ᵢ |
| Per-sample push | support vectors ~ 1/t; others decay faster |
| Margins uₙ(t) | support vectors: parallel lines, slope 1 in log t |

Also plotted: expected reward V(θ_t), accuracy and ‖∇L‖. Accuracy reaches 1 early, so the implicit-bias dynamics all
happen after the task is "solved".

Controls: space plays or pauses, ←/→ steps a frame, and hovering a point shows its rewards, gap, margin and push.

## Theory computed for linear models

`rlab/theory.py` computes:
- separability (an LP)
- the hard-margin solution θ̂, its support set S and the KKT coefficients α, refined on the active set
- θ̃ as the least-norm solution of ⟨yₙxₙ, θ̃⟩ = log(η cₙ / αₙ) for n ∈ S
- the step-size bound η < 8N / (Δ_max σ²_max(X))
- the multiclass max-margin Ŵ when there are K > 2 groups

θ̃ assumes plain gradient descent. Snapshots are log-spaced in t, and training runs in float64 so the e^{−u} tails stay
accurate.

## Project layout

```
rlab/
  experiment.py   Experiment: training loop and snapshot recording
  theory.py       max-margin θ̂, support vectors, θ̃, step-size bound
  data.py         2D datasets
  rewards.py      reward functions
  models.py       Linear, MLP
  losses.py       expected_reward (policy gradient), reinforce, cross-entropy, exponential, hinge
  updates.py      gd, normalized_gd, momentum, torch optimizers
  viewer.py       local live server and HTML export
  web/            the viewer (plain HTML/JS, no build step)
examples/
```

## References

- D. Soudry, E. Hoffer, M. S. Nacson, S. Gunasekar, N. Srebro. *The Implicit Bias of Gradient Descent on Separable
  Data.* JMLR 19 (2018).
