"""
yadel/comparator.py — Stage 5: Topology Comparator

Runs Topology A and Topology B under an *identical* noise realisation so that
any difference in the envelope is purely due to topology, not noise variation.

Topology A:  x[n] = A*sin(omega*x[n-tau] + detuning) + noise[n]
Topology B:  x[n] = A*sin(omega*x[n-tau] + 0.5*x[n-2*tau] + detuning) + noise[n]

The extra half-weighted second delay in B shifts the effective resonance
pocket to a lower detuning region and narrows the stable band — producing
visibly different envelopes.

The overlay plot shows:
  1. Both score envelopes on the same axes (colour-coded)
  2. Classification scatter (stable/fragile/collapsed) for each topology
  3. Measurable divergence metric: area between the two score curves
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from .core import DEFAULTS, make_noise
from .sweep import run_sweep, SweepResult


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class ComparisonResult:
    sweep_A:         SweepResult
    sweep_B:         SweepResult
    divergence_auc:  float    # area between the two score curves (L1 distance)


# ---------------------------------------------------------------------------
# Main comparison runner
# ---------------------------------------------------------------------------
def compare_topologies(
    det_min:  float = 0.0,
    det_max:  float = 3.0,
    n_points: int   = 60,
    params:   dict  = None,
) -> ComparisonResult:
    """
    Run both topologies under identical shared noise.

    Returns
    -------
    ComparisonResult with both SweepResult objects and the divergence metric.
    """
    if params is None:
        params = dict(DEFAULTS)

    # Single shared noise realisation
    rng   = np.random.default_rng(params.get("seed", 42))
    noise = make_noise(params["n_samples"], params["noise_amplitude"],
                       params["noise_burst_prob"], rng)

    sweep_A = run_sweep(det_min, det_max, n_points, topology="A",
                        params=params, noise=noise)
    sweep_B = run_sweep(det_min, det_max, n_points, topology="B",
                        params=params, noise=noise)

    # L1 divergence between score curves (area between)
    divergence = float(np.trapezoid(np.abs(sweep_A.scores - sweep_B.scores),
                                    sweep_A.detuning))

    return ComparisonResult(sweep_A=sweep_A, sweep_B=sweep_B,
                            divergence_auc=divergence)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
_TOPO_COLOURS = {"A": "#2980b9", "B": "#c0392b"}
_LABEL_MARKER  = {"stable": "o", "fragile": "^", "collapsed": "x"}


def plot_comparison(comp: ComparisonResult, out_path: Path) -> Path:
    """
    Save an overlay comparison plot.

    Two panels:
      Top  — score envelopes of A and B on the same axes
      Bottom — classification agreement / disagreement map
    """
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})

    det = comp.sweep_A.detuning

    # --- Panel 1: score envelopes ---
    for sweep in (comp.sweep_A, comp.sweep_B):
        c    = _TOPO_COLOURS[sweep.topology]
        ax1.plot(det, sweep.scores, color=c, linewidth=1.8,
                 label=f"Topology {sweep.topology}", alpha=0.85)
        ax1.scatter(det, sweep.scores, c=c, s=22, zorder=3,
                    edgecolors="k", linewidths=0.2, alpha=0.7)

    # Shade divergence area
    ax1.fill_between(det, comp.sweep_A.scores, comp.sweep_B.scores,
                     alpha=0.12, color="#7f8c8d",
                     label=f"Divergence AUC = {comp.divergence_auc:.4f}")

    ax1.axhline(comp.sweep_A.params["collapse_threshold"], linestyle="--",
                color="#e74c3c", linewidth=0.9, alpha=0.7, label="Collapse thr")
    ax1.axhline(comp.sweep_A.params["fragile_threshold"],  linestyle="--",
                color="#f39c12", linewidth=0.9, alpha=0.7, label="Fragile thr")
    ax1.set_ylabel("Mean running variance", fontsize=10)
    ax1.set_title("YADEL — Topology A vs B  (identical noise realisation)", fontsize=11)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_ylim(bottom=0)

    # --- Panel 2: agreement map ---
    n = len(det)
    agreement = np.zeros(n)
    for i, (la, lb) in enumerate(zip(comp.sweep_A.labels, comp.sweep_B.labels)):
        if la == lb:
            agreement[i] =  1.0   # agree
        else:
            agreement[i] = -1.0   # disagree

    colours_agree = ["#2ecc71" if a > 0 else "#e74c3c" for a in agreement]
    ax2.bar(det, np.ones(n), width=(det[1] - det[0]) * 0.9,
            color=colours_agree, align="center", alpha=0.7)
    ax2.set_yticks([])
    ax2.set_ylabel("Agreement", fontsize=9)
    ax2.set_xlabel("Detuning", fontsize=11)

    agree_pct = 100 * np.mean(np.array(agreement) > 0)
    ax2.set_title(f"Classification agreement: {agree_pct:.0f}%", fontsize=9,
                  loc="left", pad=2)

    # Legend patches
    ax2.legend(handles=[
        mpatches.Patch(color="#2ecc71", alpha=0.7, label="Agree"),
        mpatches.Patch(color="#e74c3c", alpha=0.7, label="Disagree"),
    ], fontsize=8, loc="upper right")

    fig.tight_layout()
    fname = out_path / "comparison_AvsB.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname
