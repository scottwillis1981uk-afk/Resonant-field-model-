"""
yadel/core.py — Stage 1: Core Engine  (v0.2 canonical)

Canonical model equation:
    u(t) = (g0 + detune) * (x(t-1) + k_a*x(t-da) - k_b*x(t-db)) + sigma*noise(t)
    x(t) = tanh(u(t))

Canonical parameters:
    k_a   = 0.55
    k_b   = 0.48
    g0    = 0.80
    sigma = 0.002

    Topology A:  da=5,  db=11
    Topology B:  da=7,  db=13

Stability landscape (detuning sweep from −0.25 → +0.50):
  det < −0.07   → stable at x*=0          (origin-adjacent stable zone)
  −0.07 to 0.31 → limit-cycle / collapsed  (includes det=0 — the origin)
  det > +0.31   → stable at x*≈0.65       (disconnected pocket, away from origin)

Topology B transitions at det≈+0.27 (different envelope to A).

The non-monotone structure means classical bisection starting from det=0 sees
only collapse and never discovers the stable pocket — that is YADEL's insight.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Canonical parameters (v0.2)
# ---------------------------------------------------------------------------
K_A   = 0.55
K_B   = 0.48
G0    = 0.80
SIGMA = 0.002

# Topology delay indices
DELAYS = {
    "A": {"da": 5,  "db": 11},
    "B": {"da": 7,  "db": 13},
}


# ---------------------------------------------------------------------------
# Default simulation / classification parameters
# ---------------------------------------------------------------------------
DEFAULTS = {
    # Canonical model
    "k_a":   K_A,
    "k_b":   K_B,
    "g0":    G0,
    "sigma": SIGMA,
    # Simulation length
    "n_samples": 3000,
    "warmup":    400,    # transient samples before variance measurement
    # Classification thresholds (tuned to canonical variance ranges)
    "collapse_threshold": 0.05,    # mean running-var ≥ this → collapsed
    "fragile_threshold":  0.0008,  # mean running-var ≥ this → fragile (below = stable)
    "window":             100,     # sliding-window length for variance estimate
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Noise generator
# ---------------------------------------------------------------------------
def make_noise(n_samples: int, sigma: float,
               rng: np.random.Generator) -> np.ndarray:
    """
    Continuous Gaussian noise with amplitude sigma.
    (Canonical spec: sigma*noise(t), noise ~ N(0,1))
    """
    return sigma * rng.standard_normal(n_samples)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
def simulate(detuning: float, noise: np.ndarray, params: dict,
             topology: str = "A") -> np.ndarray:
    """
    Run one instance of the canonical delayed oscillator.

    Parameters
    ----------
    detuning : float
        Swept parameter.  Collapsed at det≈0; stable pocket at det>0.31 (Topo A).
    noise    : 1-D array, length n_samples
        Pre-generated noise (sigma already applied; shared across topologies).
    params   : dict
        Simulation parameters (see DEFAULTS).
    topology : "A" or "B"
        Selects the delay indices da, db.

    Returns
    -------
    x : 1-D array, length n_samples
    """
    k_a = params["k_a"]
    k_b = params["k_b"]
    g0  = params["g0"]
    n   = params["n_samples"]
    da  = DELAYS[topology]["da"]
    db  = DELAYS[topology]["db"]

    buf = db + 2   # buffer must cover the longest delay
    rng_init = np.random.default_rng(params.get("seed", 42))

    x = np.zeros(n)
    x[:buf] = 0.01 * rng_init.standard_normal(buf)   # small non-zero IC

    for t in range(buf, n):
        u    = (g0 + detuning) * (x[t-1] + k_a*x[t-da] - k_b*x[t-db]) + noise[t]
        x[t] = np.tanh(u)

    return x


# ---------------------------------------------------------------------------
# Running variance and classification
# ---------------------------------------------------------------------------
def running_variance(x: np.ndarray, window: int) -> np.ndarray:
    """Causal sliding-window variance."""
    n   = len(x)
    out = np.zeros(n)
    for i in range(window, n):
        out[i] = np.var(x[i - window: i])
    return out


def mean_running_variance(x: np.ndarray, params: dict) -> float:
    """Scalar stability score: mean running variance over post-warmup window."""
    rv = running_variance(x, params["window"])
    return float(np.mean(rv[params["warmup"]:]))


def classify(x: np.ndarray, params: dict) -> str:
    """
    Classify a trajectory as 'stable', 'fragile', or 'collapsed'.

    Thresholds are calibrated to the canonical v0.2 parameter set:
      stable    : mean_rv < 0.0008   (x converged to fixed point)
      fragile   : 0.0008 ≤ mean_rv < 0.05
      collapsed : mean_rv ≥ 0.05    (sustained limit cycle)
    """
    mean_rv  = mean_running_variance(x, params)
    col_thr  = params["collapse_threshold"]
    frag_thr = params["fragile_threshold"]

    if mean_rv >= col_thr:
        return "collapsed"
    elif mean_rv >= frag_thr:
        return "fragile"
    else:
        return "stable"
