"""
yadel_demo.py — YADEL full demonstration script

Runs all six stages in sequence and verifies the four required behaviours:
  1. Collapse appears reliably across the detuning sweep
  2. Pocket regimes detected (disconnected from origin)
  3. Structural changepoint found before collapse
  4. Topology A and B produce visibly different envelopes

All plots are saved to ./yadel_output/<run_id>/.
A JSON report is written to the same directory.
"""

import sys
from pathlib import Path

# Make sure the package is importable when running from the repo root
sys.path.insert(0, str(Path(__file__).parent))

from yadel.reporter import run_full


def main():
    print("=" * 60)
    print(" YADEL — Yet Another Delay Envelope Lab")
    print(" Stability Mapper  |  Full Demo Run")
    print("=" * 60)

    report = run_full(
        det_min  = 0.0,
        det_max  = 3.2,
        n_points = 80,
    )

    # ------------------------------------------------------------------ #
    # Print human-readable summary                                         #
    # ------------------------------------------------------------------ #
    print(f"\nRun ID : {report['run_id']}")
    print(f"Output : {Path(report['_json_path']).parent}")

    print("\n--- Topology A ---")
    A = report["topology_A"]
    print(f"  stable={A['stable']}  fragile={A['fragile']}  collapsed={A['collapsed']}")
    print(f"  Disconnected pockets : {A['n_disconnected_pockets']}")
    for pk in A["pockets"]:
        kind = "DISCONNECTED" if pk["is_disconnected"] else "origin-adjacent"
        print(f"    [{kind}] det {pk['det_start']:.3f} – {pk['det_end']:.3f}  "
              f"({pk['n_points']} pts)")
    print(f"  Changepoint (gradient): det={A['changepoint_gradient']['detuning']:.3f}")
    print(f"  Changepoint (CUSUM)   : det={A['changepoint_cusum']['detuning']:.3f}")

    print("\n--- Topology B ---")
    B = report["topology_B"]
    print(f"  stable={B['stable']}  fragile={B['fragile']}  collapsed={B['collapsed']}")
    print(f"  Disconnected pockets : {B['n_disconnected_pockets']}")
    for pk in B["pockets"]:
        kind = "DISCONNECTED" if pk["is_disconnected"] else "origin-adjacent"
        print(f"    [{kind}] det {pk['det_start']:.3f} – {pk['det_end']:.3f}  "
              f"({pk['n_points']} pts)")
    print(f"  Changepoint (CUSUM)   : det={B['changepoint_cusum']['detuning']:.3f}")

    print("\n--- Topology Comparison ---")
    print(f"  Divergence AUC : {report['comparison']['divergence_auc']:.4f}")
    print(f"  Label agreement: {report['comparison']['agreement_pct']:.0f}%")

    print("\n--- Four Required Checks ---")
    for check, passed in report["four_checks"].items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]  {check}")

    print("\n--- Plots saved ---")
    for name, path in report["plots"].items():
        print(f"  {name:<18} {Path(path).name}")

    print(f"\nJSON report: {Path(report['_json_path']).name}")

    if report["all_checks_pass"]:
        print("\n[ALL FOUR CHECKS PASSED]")
        return 0
    else:
        print("\n[SOME CHECKS FAILED]", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
