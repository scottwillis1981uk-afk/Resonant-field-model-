"""
yadel/core.py — Stage 1: Core Engine

Simulates a delayed nonlinear oscillator using an Ikeda-type recurrence:

    Topology A:  x[n] = A * sin(omega * x[n-tau] + detuning) + noise[n]
    Topology B:  x[n] = A * sin(omega * x[n-tau] + 0.5*x[n-2*tau] + detuning) + noise[n]

Key behaviour:
  • detuning ≈ 0           → unstable / collapsed (origin is not a stable operating point)
  • detuning ≈ 0.2–1.0     → stable pocket (away from the origin)
  • detuning > ~1.3        → transition into fragile / collapsed regime
  Topology B has a shifted stability envelope due to the extra delay term.

This non-monotone stability landscape means classical bisection from the
origin would miss the stable pocket entirely — that is YADEL's core insight.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Default simulation parameters
# ---------------------------------------------------------------------------
DEFAULTS = {
    "A":                 1.50,   # loop gain / nonlinearity amplitude
    "omega":             1.00,   # angular frequency inside nonlinearity
    "tau":               3,      # primary delay (integer samples)
    "n_samples":         2000,   # total simulation length
    "warmup":            300,    # transient samples discarded before analysis
    "noise_amplitude":   0.08,   # burst noise amplitude
    "noise_burst_prob":  0.10,   # probability of burst per sample
    "collapse_threshold": 0.25,  # running-variance → collapsed
    "fragile_threshold":  0.02,  # running-variance → fragile (below = stable)
    "window":            80,     # sliding window length for variance estimate
    "seed":              42,
}


def make_noise(n_samples: int, amplitude: float, burst_prob: float,
               rng: np.random.Generator) -> np.ndarray:
    """Burst noise: amplitude * N(0,1) with probability burst_prob per sample."""
    mask = rng.random(n_samples) < burst_prob
    return amplitude * rng.standard_normal(n_samples) * mask


def simulate(detuning: float, noise: np.ndarray, params: dict,
             topology: str = "A") -> np.ndarray:
    """
    Run one instance of the delayed nonlinear oscillator.

    Parameters
    ----------
    detuning  : float
        Phase offset that is swept to traverse the stability landscape.
    noise     : 1-D array, length n_samples
        Pre-generated burst noise (same array shared across both topologies).
    params    : dict
        Simulation parameters (see DEFAULTS).
    topology  : "A" or "B"
        A — single delay:  x[n] = A*sin(omega*x[n-tau] + detuning) + noise[n]
        B — dual delay:    x[n] = A*sin(omega*x[n-tau] + 0.5*x[n-2*tau] + detuning) + noise[n]
            Topology B shifts the stability envelope by mixing two delayed states.

    Returns
    -------
    x : 1-D array, length n_samples
    """
    A     = params["A"]
    omega = params["omega"]
    tau   = params["tau"]
    n     = params["n_samples"]
    rng_i = np.random.default_rng(params.get("seed", 42))

    x      = np.zeros(n)
    buf_sz = 2 * tau
    x[:buf_sz] = 0.05 * rng_i.standard_normal(buf_sz)

    start = buf_sz

    for i in range(start, n):
        if topology == "A":
            arg = omega * x[i - tau] + detuning
        else:
            # Topology B mixes two delay taps — shifts the resonance pocket
            arg = omega * x[i - tau] + 0.5 * x[i - 2 * tau] + detuning
        x[i] = A * np.sin(arg) + noise[i]

    return x


def running_variance(x: np.ndarray, window: int) -> np.ndarray:
    """Sliding-window variance (causal, no lookahead)."""
    n   = len(x)
    out = np.zeros(n)
    for i in range(window, n):
        out[i] = np.var(x[i - window: i])
    return out


def classify(x: np.ndarray, params: dict) -> str:
    """
    Classify a trajectory as 'stable', 'fragile', or 'collapsed'.

    Uses the mean running variance over the post-warmup portion of x.
    """
    warmup   = params["warmup"]
    window   = params["window"]
    col_thr  = params["collapse_threshold"]
    frag_thr = params["fragile_threshold"]

    rv      = running_variance(x, window)
    mean_rv = float(np.mean(rv[warmup:]))

    if mean_rv >= col_thr:
        return "collapsed"
    elif mean_rv >= frag_thr:
        return "fragile"
    else:
        return "stable"


def mean_running_variance(x: np.ndarray, params: dict) -> float:
    """Return the scalar stability score (mean post-warmup running variance)."""
    rv = running_variance(x, params["window"])
    return float(np.mean(rv[params["warmup"]:]))
