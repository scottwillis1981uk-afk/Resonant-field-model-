"""
yadel/pockets.py — Stage 3: Pocket Detector

A "pocket" is a contiguous run of stable operating points that is
disconnected from the origin (det=0).  This captures the core YADEL insight:
stable pockets can exist entirely away from the nominal design point.

Algorithm
---------
1. Walk the sorted detuning sweep left-to-right.
2. Merge consecutive stable (or fragile) points into candidate runs.
3. Classify each run:
   - origin-adjacent  : run starts at the first grid point (touches det=0)
   - disconnected     : run starts at an interior point (true pocket)
4. Flag non-monotone envelope regions: any local maximum in the score
   array that is surrounded by lower values on both sides.

Returns a PocketReport dataclass.  Also saves an annotated pocket plot.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from .sweep import SweepResult, _LABEL_COLOUR


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class Pocket:
    start_idx:     int
    end_idx:       int           # inclusive
    det_start:     float
    det_end:       float
    is_disconnected: bool        # True if not touching the first grid point
    mean_score:    float
    n_points:      int


@dataclass
class PocketReport:
    pockets:            List[Pocket]
    n_disconnected:     int
    non_monotone_idxs:  List[int]   # indices of local maxima in the score array
    sweep:              SweepResult


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------
_STABLE_LABELS = {"stable", "fragile"}


def detect_pockets(sweep: SweepResult, min_run_length: int = 2) -> PocketReport:
    """
    Identify stable pockets in a SweepResult.

    Parameters
    ----------
    sweep : SweepResult
    min_run_length : int
        Minimum consecutive stable/fragile points to count as a pocket.

    Returns
    -------
    PocketReport
    """
    labels  = sweep.labels
    scores  = sweep.scores
    det     = sweep.detuning
    n       = len(labels)

    # --- Build runs of stable/fragile labels ---
    pockets = []
    i = 0
    while i < n:
        if labels[i] in _STABLE_LABELS:
            j = i
            while j < n and labels[j] in _STABLE_LABELS:
                j += 1
            run_len = j - i
            if run_len >= min_run_length:
                pockets.append(Pocket(
                    start_idx      = i,
                    end_idx        = j - 1,
                    det_start      = float(det[i]),
                    det_end        = float(det[j - 1]),
                    is_disconnected= (i > 0),   # not touching first grid point
                    mean_score     = float(np.mean(scores[i:j])),
                    n_points       = run_len,
                ))
            i = j
        else:
            i += 1

    # --- Detect non-monotone regions (local maxima in score) ---
    non_mono = []
    for k in range(1, n - 1):
        if scores[k] > scores[k - 1] and scores[k] > scores[k + 1]:
            non_mono.append(k)

    n_disconnected = sum(1 for p in pockets if p.is_disconnected)

    return PocketReport(
        pockets=pockets,
        n_disconnected=n_disconnected,
        non_monotone_idxs=non_mono,
        sweep=sweep,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_pockets(report: PocketReport, out_path: Path) -> Path:
    """Save annotated pocket plot."""
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    sweep = report.sweep
    det   = sweep.detuning
    sc    = sweep.scores

    fig, ax = plt.subplots(figsize=(11, 4.5))

    # Base envelope line
    colours = [_LABEL_COLOUR[lb] for lb in sweep.labels]
    ax.plot(det, sc, color="#555555", linewidth=1.0, zorder=2, alpha=0.5)
    ax.scatter(det, sc, c=colours, s=35, zorder=3, edgecolors="k", linewidths=0.3)

    # Shade pocket regions
    for pk in report.pockets:
        colour = "#1a73e8" if pk.is_disconnected else "#34a853"
        ax.axvspan(pk.det_start, pk.det_end, alpha=0.18, color=colour, zorder=1,
                   label=("Disconnected pocket" if pk.is_disconnected
                          else "Origin-adjacent stable region"))

    # Mark non-monotone local maxima
    for idx in report.non_monotone_idxs:
        ax.annotate("⚠ non-mono",
                    xy=(det[idx], sc[idx]),
                    xytext=(det[idx] + 0.05, sc[idx] + 0.03),
                    fontsize=7, color="#c0392b",
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8))

    # Threshold lines
    ax.axhline(sweep.params["fragile_threshold"],  linestyle="--",
               color="#f39c12", linewidth=0.9, alpha=0.7)
    ax.axhline(sweep.params["collapse_threshold"], linestyle="--",
               color="#e74c3c", linewidth=0.9, alpha=0.7)

    # Custom legend (deduplicate)
    seen = set()
    legend_handles = []
    for lb, c in _LABEL_COLOUR.items():
        legend_handles.append(mpatches.Patch(color=c, label=lb))
    legend_handles.append(mpatches.Patch(color="#1a73e8", alpha=0.4,
                                          label="Disconnected pocket"))
    legend_handles.append(mpatches.Patch(color="#34a853", alpha=0.4,
                                          label="Origin-adjacent stable"))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7, ncol=2)

    n_disc  = report.n_disconnected
    n_total = len(report.pockets)
    ax.set_title(
        f"YADEL — Pocket Detector  (Topology {sweep.topology}) | "
        f"{n_total} pocket(s), {n_disc} disconnected",
        fontsize=11,
    )
    ax.set_xlabel("Detuning", fontsize=11)
    ax.set_ylabel("Mean running variance", fontsize=11)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    fname = out_path / f"pockets_topo{sweep.topology}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname
