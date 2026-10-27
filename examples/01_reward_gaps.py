"""Draft 1 vs draft 2: exact policy gradient on a linear policy with equal and
unequal positive reward gaps. Both should converge in direction to the same
max-margin theta_hat; the gaps only change the offset theta_tilde.

    python examples/01_reward_gaps.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rlab
from rlab import data, losses, rewards, updates

X, labels = data.separable_blobs(n=30, angle_deg=30, seed=0)

equal = rlab.Experiment(X, labels, reward=rewards.constant_gap(1.0, 0.0), model="linear",
                        loss=losses.expected_reward, update=updates.gd(lr=0.5), name="equal gaps Δ=1")
unequal = rlab.Experiment(X, labels, reward=rewards.random_gaps(0.2, 2.0, seed=1), model="linear",
                          loss=losses.expected_reward, update=updates.gd(lr=0.5), name="unequal gaps Δ~U[0.2,2]")

print("eta bound 8N/(Δmax σmax²):", unequal.theory["eta_bound"])
rlab.show(equal, unequal, wait=False)          # open the viewer first, then watch both train
equal.train(steps=200_000)
unequal.train(steps=200_000)
rlab.save_html("runs/01_reward_gaps.html", equal, unequal)
rlab.show()
