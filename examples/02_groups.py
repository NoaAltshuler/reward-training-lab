"""Groups: a K-class softmax policy pi(k|x) = softmax(W phi(x))_k trained by
exact policy gradient. The viewer shows each class vector w_k moving, and the
angle to the multiclass max-margin solution W_hat.

    python examples/02_groups.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rlab
from rlab import data, rewards, updates

X, labels = data.multiclass_blobs(n=45, K=3, seed=3)
exp = rlab.Experiment(X, labels, reward=rewards.gap_per_class([1.0, 2.0, 0.5]), model="linear_bias",
                      loss="expected_reward", update=updates.gd(lr=1.0), name="3 groups, per-class gaps")
exp.train(steps=50_000, live=True, wait=False)
exp.save_html("runs/02_groups.html")
rlab.show()
