"""
yadel/sweep.py — Stage 2: Sweep and Envelope Mapper

Sweeps the detuning parameter from det_min to det_max using n_points steps.
All operating points share the same burst-noise realisation so that differences
in the envelope are purely due to dynamics, not noise variation.

Returns a SweepResult dataclass with:
  - detuning values
  - per-point stability score (mean running variance)
  - per-point classification ('stable' / 'fragile' / 'collapsed')
  - the full state trajectories (optional, controlled by keep_traces)

Also saves an envelope plot to the output directory.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless — never open a display window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .core import DEFAULTS, make_noise, simulate, classify, mean_running_variance


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class SweepResult:
    detuning:   np.ndarray          # shape (n_points,)
    scores:     np.ndarray          # mean running variance per point
    labels:     List[str]           # 'stable' / 'fragile' / 'collapsed'
    topology:   str
    params:     dict
    noise:      np.ndarray          # the shared noise realisation
    traces:     List[np.ndarray] = field(default_factory=list)   # optional


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_LABEL_COLOUR = {"stable": "#2ecc71", "fragile": "#f39c12", "collapsed": "#e74c3c"}


# ---------------------------------------------------------------------------
# Main sweep function
# ---------------------------------------------------------------------------
def run_sweep(
    det_min:     float = 0.0,
    det_max:     float = 3.0,
    n_points:    int   = 60,
    topology:    str   = "A",
    params:      dict  = None,
    noise:       np.ndarray = None,   # supply externally to share with topology B
    keep_traces: bool  = False,
) -> SweepResult:
    """
    Sweep detuning, classify each point, return SweepResult.

    Parameters
    ----------
    det_min, det_max : float
        Detuning range.
    n_points : int
        Number of operating points.
    topology : "A" or "B"
        Which oscillator topology to simulate.
    params : dict | None
        Override DEFAULTS.
    noise : array | None
        Pre-generated noise.  If None, a fresh one is created from params seed.
    keep_traces : bool
        If True, store all state trajectories in SweepResult.traces.
    """
    if params is None:
        params = dict(DEFAULTS)

    if noise is None:
        rng   = np.random.default_rng(params.get("seed", 42))
        noise = make_noise(params["n_samples"], params["noise_amplitude"],
                           params["noise_burst_prob"], rng)

    det_values = np.linspace(det_min, det_max, n_points)
    scores     = np.zeros(n_points)
    labels     = []
    traces     = []

    for idx, det in enumerate(det_values):
        x      = simulate(det, noise, params, topology=topology)
        scores[idx] = mean_running_variance(x, params)
        labels.append(classify(x, params))
        if keep_traces:
            traces.append(x)

    return SweepResult(
        detuning=det_values,
        scores=scores,
        labels=labels,
        topology=topology,
        params=params,
        noise=noise,
        traces=traces,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_envelope(result: SweepResult, out_path: Path) -> Path:
    """
    Save an envelope plot (running-variance vs detuning, coloured by label).

    Returns the saved file Path.
    """
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))

    det = result.detuning
    sc  = result.scores

    # Scatter coloured by classification
    colours = [_LABEL_COLOUR[lb] for lb in result.labels]
    ax.scatter(det, sc, c=colours, s=40, zorder=3, edgecolors="k", linewidths=0.4)
    ax.plot(det, sc, color="#555555", linewidth=0.8, zorder=2, alpha=0.6)

    # Threshold lines
    ax.axhline(result.params["fragile_threshold"],  color="#f39c12", linestyle="--",
               linewidth=1.0, label=f"fragile thr = {result.params['fragile_threshold']}")
    ax.axhline(result.params["collapse_threshold"], color="#e74c3c", linestyle="--",
               linewidth=1.0, label=f"collapse thr = {result.params['collapse_threshold']}")

    # Legend patches
    patches = [mpatches.Patch(color=c, label=lb)
               for lb, c in _LABEL_COLOUR.items()]
    ax.legend(handles=patches, loc="upper right", fontsize=8)

    ax.set_xlabel("Detuning", fontsize=11)
    ax.set_ylabel("Mean running variance", fontsize=11)
    ax.set_title(f"YADEL — Stability Envelope  (Topology {result.topology})", fontsize=12)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    fname = out_path / f"envelope_topo{result.topology}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname
