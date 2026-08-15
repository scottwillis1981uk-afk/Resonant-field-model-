"""
yadel/changepoint.py — Stage 4: Changepoint Detector

Finds the structural boundary that precedes collapse.  In a delayed
oscillator the system does not collapse abruptly: it first "stiffens" —
the variance signal shows a rapid upward inflection that predicts the
upcoming collapse boundary.

Two complementary detection methods are provided:

1. Gradient method
   Compute the first-difference (gradient) of the score array.  A
   changepoint is the index with the largest positive gradient that lies
   *before* the first collapsed point.  This is where the score begins
   its steepest rise.

2. CUSUM (cumulative-sum) method
   Compute the CUSUM of the deviation of scores from a running baseline.
   A changepoint is where the CUSUM first exceeds a detection threshold.
   CUSUM is more sensitive to sustained drift than the gradient method.

Both methods return a ChangePoint dataclass.  The plot overlays both
detections on the envelope for comparison.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .sweep import SweepResult, _LABEL_COLOUR


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class ChangePoint:
    method:         str          # 'gradient' or 'cusum'
    index:          int          # index in the sweep
    detuning:       float        # detuning value at changepoint
    score:          float        # variance score at changepoint
    # For CUSUM only:
    cusum_threshold: Optional[float] = None
    cusum_series:    Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Detection methods
# ---------------------------------------------------------------------------
def detect_gradient(sweep: SweepResult) -> ChangePoint:
    """
    Gradient changepoint: index of the steepest upward score transition that
    leads INTO the collapsed zone.

    For the canonical v0.2 landscape (left-stable → collapsed → right-pocket),
    the changepoint of interest is the exit from the left stable region.  We
    find the global argmax of the positive first-difference of scores, which
    naturally lands on the sharpest stable→collapsed transition edge.
    """
    scores = sweep.scores
    det    = sweep.detuning

    grad = np.diff(scores)
    if len(grad) == 0:
        idx = 0
    else:
        idx = int(np.argmax(grad))   # index just before the steepest rise

    return ChangePoint(
        method   = "gradient",
        index    = idx,
        detuning = float(det[idx]),
        score    = float(scores[idx]),
    )


def detect_cusum(sweep: SweepResult, slack_factor: float = 0.5,
                 threshold_sigma: float = 15.0) -> ChangePoint:
    """
    CUSUM changepoint: first sustained upward drift in the score series.

    Parameters
    ----------
    slack_factor : float
        The CUSUM slack = slack_factor * std(scores_in_stable_region).
        Lower → more sensitive.
    threshold_sigma : float
        Detection threshold = threshold_sigma * std_stable.
    """
    scores = sweep.scores
    det    = sweep.detuning
    labels = sweep.labels
    n      = len(scores)

    # Baseline from the flat (early) portion of the FIRST contiguous stable run.
    # Two reasons for this restriction:
    #  1. The left stable zone has a gradual upward drift; using all stable
    #     points biases mu_base high, delaying detection.
    #  2. A second disconnected stable pocket (right side) has a very different
    #     score level and must not contaminate the baseline.
    # We take the first contiguous stable run, then use only its first half to
    # stay in the flat portion — this gives ~6-step early warning before the
    # stable→collapsed transition.
    first_run = []
    for i, lb in enumerate(labels):
        if lb == "stable":
            first_run.append(i)
        elif first_run:
            break   # end of first contiguous stable block

    if len(first_run) < 4:
        early = list(range(min(4, n)))
    else:
        early = first_run[: max(len(first_run) // 2, 2)]

    mu_base  = float(np.mean(scores[early]))
    std_base = float(np.std(scores[early])) + 1e-12   # guard against zero std

    slack     = slack_factor * std_base
    threshold = threshold_sigma * std_base

    # CUSUM accumulator (upward only)
    cusum = np.zeros(n)
    for i in range(1, n):
        cusum[i] = max(0.0, cusum[i - 1] + (scores[i] - mu_base) - slack)

    # First exceedance
    exceed_idxs = np.where(cusum > threshold)[0]
    if len(exceed_idxs) == 0:
        idx = n - 1   # no detection — report the last point
    else:
        idx = int(exceed_idxs[0])

    return ChangePoint(
        method           = "cusum",
        index            = idx,
        detuning         = float(det[idx]),
        score            = float(scores[idx]),
        cusum_threshold  = threshold,
        cusum_series     = cusum,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_changepoints(sweep: SweepResult,
                      cp_grad: ChangePoint,
                      cp_cusum: ChangePoint,
                      out_path: Path) -> Path:
    """Save a two-panel plot: envelope (with changepoints) + CUSUM series."""
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    det  = sweep.detuning
    sc   = sweep.scores
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # --- Panel 1: envelope with changepoint markers ---
    colours = [_LABEL_COLOUR[lb] for lb in sweep.labels]
    ax1.plot(det, sc, color="#555555", linewidth=1.0, alpha=0.5)
    ax1.scatter(det, sc, c=colours, s=35, zorder=3, edgecolors="k", linewidths=0.3)

    ax1.axvline(cp_grad.detuning, color="#8e44ad", linestyle="-.",
                linewidth=1.5, label=f"Gradient CP  det={cp_grad.detuning:.3f}")
    ax1.axvline(cp_cusum.detuning, color="#2980b9", linestyle=":",
                linewidth=1.8, label=f"CUSUM CP     det={cp_cusum.detuning:.3f}")

    ax1.axhline(sweep.params["collapse_threshold"], color="#e74c3c",
                linestyle="--", linewidth=0.9, alpha=0.7)
    ax1.set_ylabel("Mean running variance", fontsize=10)
    ax1.set_title(
        f"YADEL — Changepoint Detector  (Topology {sweep.topology})", fontsize=11)
    ax1.legend(fontsize=8, loc="upper left")

    # --- Panel 2: CUSUM series ---
    if cp_cusum.cusum_series is not None:
        ax2.plot(det, cp_cusum.cusum_series, color="#2980b9", linewidth=1.2,
                 label="CUSUM")
        if cp_cusum.cusum_threshold is not None:
            ax2.axhline(cp_cusum.cusum_threshold, color="#e74c3c", linestyle="--",
                        linewidth=1.0, label=f"threshold={cp_cusum.cusum_threshold:.4f}")
        ax2.axvline(cp_cusum.detuning, color="#2980b9", linestyle=":",
                    linewidth=1.5, alpha=0.7)
        ax2.set_ylabel("CUSUM", fontsize=10)
        ax2.legend(fontsize=8)

    ax2.set_xlabel("Detuning", fontsize=11)

    fig.tight_layout()
    fname = out_path / f"changepoint_topo{sweep.topology}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname
