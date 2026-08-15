"""Stage 4 smoke test — changepoint detector (v0.2 canonical)."""
import sys, os, tempfile
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from yadel.sweep import run_sweep
from yadel.changepoint import detect_gradient, detect_cusum, plot_changepoints

def test_stage4():
    result   = run_sweep(det_min=-0.25, det_max=0.50, n_points=60, topology="A")
    cp_grad  = detect_gradient(result)
    cp_cusum = detect_cusum(result)

    print(f"  Gradient CP: detuning={cp_grad.detuning:.4f}  index={cp_grad.index}")
    print(f"  CUSUM    CP: detuning={cp_cusum.detuning:.4f}  index={cp_cusum.index}")

    labels = result.labels

    # First collapsed index (left-stable exits here)
    first_collapse_idx = next((i for i, lb in enumerate(labels) if lb == "collapsed"),
                               len(labels) - 1)
    # Last collapsed index (right pocket starts after here)
    last_collapse_idx  = max((i for i, lb in enumerate(labels) if lb == "collapsed"),
                              default=len(labels) - 1)

    print(f"  First collapse idx={first_collapse_idx}  "
          f"last collapse idx={last_collapse_idx}")

    # Both CPs must fire at or before the last collapsed point
    assert cp_grad.index  <= last_collapse_idx, \
        f"Gradient CP (idx={cp_grad.index}) beyond last collapse (idx={last_collapse_idx})"
    assert cp_cusum.index <= last_collapse_idx, \
        f"CUSUM CP (idx={cp_cusum.index}) beyond last collapse (idx={last_collapse_idx})"

    # CUSUM should fire before or at the first collapse (critical validation)
    steps_early = first_collapse_idx - cp_cusum.index
    print(f"  CUSUM fires {steps_early} steps before first collapse "
          f"(target: ~6)")
    assert steps_early >= 0, \
        f"CUSUM fired after first collapse (steps_early={steps_early})"

    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = plot_changepoints(result, cp_grad, cp_cusum, Path(tmpdir))
        assert fpath.exists()
        print(f"  Plot saved → {fpath.name}")

    print("Stage 4 PASSED.")

if __name__ == "__main__":
    test_stage4()
