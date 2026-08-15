"""
yadel/reporter.py — Stage 6: Report Generator

Orchestrates a full YADEL run:
  1. Sweep both topologies under identical noise
  2. Detect pockets in each topology
  3. Detect changepoints in each topology
  4. Compare topologies
  5. Save all plots
  6. Write a timestamped JSON report

The JSON report is structured for programmatic reuse:
  {
    "run_id":       "<timestamp>",
    "params":       { ... },
    "topology_A":   { "n_stable":..., "n_fragile":..., "n_collapsed":...,
                      "pockets": [...], "changepoint_gradient":...,
                      "changepoint_cusum":... },
    "topology_B":   { ... },
    "comparison":   { "divergence_auc":..., "agreement_pct":... },
    "plots":        { "envelope_A":..., ... },
    "four_checks":  { "collapse_reliable":..., "pockets_detected":...,
                      "changepoint_found":..., "topology_divergence":... }
  }
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path

from .core import DEFAULTS, make_noise
from .sweep import run_sweep, plot_envelope
from .pockets import detect_pockets, plot_pockets
from .changepoint import detect_gradient, detect_cusum, plot_changepoints
from .comparator import compare_topologies, plot_comparison


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pocket_summary(pockets) -> list:
    return [
        {
            "det_start":       round(p.det_start, 4),
            "det_end":         round(p.det_end, 4),
            "n_points":        p.n_points,
            "is_disconnected": p.is_disconnected,
            "mean_score":      round(p.mean_score, 6),
        }
        for p in pockets
    ]


def _label_counts(labels) -> dict:
    return {lb: labels.count(lb) for lb in ("stable", "fragile", "collapsed")}


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_full(
    det_min:  float = 0.0,
    det_max:  float = 3.0,
    n_points: int   = 80,
    params:   dict  = None,
    out_dir:  Path  = None,
) -> dict:
    """
    Execute a complete YADEL analysis run.

    Parameters
    ----------
    det_min, det_max : float
    n_points : int
    params   : dict | None   — override DEFAULTS
    out_dir  : Path | None   — directory for outputs; defaults to ./yadel_output/<run_id>

    Returns
    -------
    report : dict   — the structured report (also written to JSON)
    """
    if params is None:
        params = dict(DEFAULTS)

    run_id  = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    if out_dir is None:
        out_dir = Path("yadel_output") / run_id
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Shared noise — both topologies use the same realisation           #
    # ------------------------------------------------------------------ #
    rng   = np.random.default_rng(params.get("seed", 42))
    noise = make_noise(params["n_samples"], params["sigma"], rng)

    # ------------------------------------------------------------------ #
    # 2. Sweep each topology                                               #
    # ------------------------------------------------------------------ #
    sweep_A = run_sweep(det_min, det_max, n_points, topology="A",
                        params=params, noise=noise, keep_traces=False)
    sweep_B = run_sweep(det_min, det_max, n_points, topology="B",
                        params=params, noise=noise, keep_traces=False)

    # ------------------------------------------------------------------ #
    # 3. Pocket detection                                                  #
    # ------------------------------------------------------------------ #
    pkt_A = detect_pockets(sweep_A)
    pkt_B = detect_pockets(sweep_B)

    # ------------------------------------------------------------------ #
    # 4. Changepoint detection                                             #
    # ------------------------------------------------------------------ #
    cp_grad_A  = detect_gradient(sweep_A)
    cp_cusum_A = detect_cusum(sweep_A)
    cp_grad_B  = detect_gradient(sweep_B)
    cp_cusum_B = detect_cusum(sweep_B)

    # ------------------------------------------------------------------ #
    # 5. Topology comparison                                               #
    # ------------------------------------------------------------------ #
    from .comparator import ComparisonResult
    comp = ComparisonResult(sweep_A=sweep_A, sweep_B=sweep_B,
                            divergence_auc=float(
                                np.trapezoid(np.abs(sweep_A.scores - sweep_B.scores),
                                             sweep_A.detuning)
                            ))
    agree_pct = 100 * np.mean(
        np.array([int(la == lb) for la, lb in
                  zip(sweep_A.labels, sweep_B.labels)], dtype=float)
    )

    # ------------------------------------------------------------------ #
    # 6. Save plots                                                        #
    # ------------------------------------------------------------------ #
    plots = {
        "envelope_A":   str(plot_envelope(sweep_A, out_dir)),
        "envelope_B":   str(plot_envelope(sweep_B, out_dir)),
        "pockets_A":    str(plot_pockets(pkt_A, out_dir)),
        "pockets_B":    str(plot_pockets(pkt_B, out_dir)),
        "changepoint_A":str(plot_changepoints(sweep_A, cp_grad_A, cp_cusum_A, out_dir)),
        "changepoint_B":str(plot_changepoints(sweep_B, cp_grad_B, cp_cusum_B, out_dir)),
        "comparison":   str(plot_comparison(comp, out_dir)),
    }

    # ------------------------------------------------------------------ #
    # 7. Four required checks                                              #
    # ------------------------------------------------------------------ #
    four_checks = {
        "collapse_reliable": "collapsed" in sweep_A.labels and "collapsed" in sweep_B.labels,
        "pockets_detected":  pkt_A.n_disconnected >= 1 or pkt_B.n_disconnected >= 1,
        "changepoint_found": cp_cusum_A.index < len(sweep_A.labels) - 1,
        # Threshold calibrated to canonical parameter set:
        # B's pocket entry is ~4 det-steps earlier than A's, producing AUC ~0.03.
        "topology_divergence": comp.divergence_auc > 0.02,
    }
    all_pass = all(four_checks.values())

    # ------------------------------------------------------------------ #
    # 8. Assemble report dict                                              #
    # ------------------------------------------------------------------ #
    report = {
        "run_id":  run_id,
        "params":  {k: (int(v) if isinstance(v, np.integer) else
                         float(v) if isinstance(v, (np.floating, float)) else v)
                    for k, v in params.items()},
        "sweep_range": {"det_min": det_min, "det_max": det_max, "n_points": n_points},
        "topology_A": {
            **_label_counts(sweep_A.labels),
            "pockets":              _pocket_summary(pkt_A.pockets),
            "n_disconnected_pockets": pkt_A.n_disconnected,
            "changepoint_gradient": {"detuning": round(cp_grad_A.detuning, 4),
                                      "score":   round(cp_grad_A.score, 6)},
            "changepoint_cusum":    {"detuning": round(cp_cusum_A.detuning, 4),
                                      "score":   round(cp_cusum_A.score, 6),
                                      "threshold": round(cp_cusum_A.cusum_threshold, 6)
                                                  if cp_cusum_A.cusum_threshold else None},
        },
        "topology_B": {
            **_label_counts(sweep_B.labels),
            "pockets":              _pocket_summary(pkt_B.pockets),
            "n_disconnected_pockets": pkt_B.n_disconnected,
            "changepoint_gradient": {"detuning": round(cp_grad_B.detuning, 4),
                                      "score":   round(cp_grad_B.score, 6)},
            "changepoint_cusum":    {"detuning": round(cp_cusum_B.detuning, 4),
                                      "score":   round(cp_cusum_B.score, 6),
                                      "threshold": round(cp_cusum_B.cusum_threshold, 6)
                                                  if cp_cusum_B.cusum_threshold else None},
        },
        "comparison": {
            "divergence_auc": round(comp.divergence_auc, 4),
            "agreement_pct":  round(float(agree_pct), 1),
        },
        "plots":       plots,
        "four_checks": four_checks,
        "all_checks_pass": all_pass,
    }

    # ------------------------------------------------------------------ #
    # 9. Write JSON                                                        #
    # ------------------------------------------------------------------ #
    json_path = out_dir / f"report_{run_id}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    report["_json_path"] = str(json_path)
    return report
