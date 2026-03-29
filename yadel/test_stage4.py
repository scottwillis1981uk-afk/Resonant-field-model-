"""Stage 4 smoke test — changepoint detector."""
import sys, os, tempfile
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from yadel.sweep import run_sweep
from yadel.changepoint import detect_gradient, detect_cusum, plot_changepoints

def test_stage4():
    result   = run_sweep(det_min=0.0, det_max=3.0, n_points=60, topology="A")
    cp_grad  = detect_gradient(result)
    cp_cusum = detect_cusum(result)

    print(f"  Gradient CP: detuning={cp_grad.detuning:.3f}  score={cp_grad.score:.4f}")
    print(f"  CUSUM    CP: detuning={cp_cusum.detuning:.3f}  score={cp_cusum.score:.4f}")

    # Both CPs should precede or coincide with the last-collapsed transition
    labels = result.labels
    last_stable_idx  = max((i for i, lb in enumerate(labels)
                            if lb in ("stable", "fragile")), default=0)
    last_collapse_idx = max((i for i, lb in enumerate(labels)
                             if lb == "collapsed"), default=len(labels)-1)
    assert cp_grad.index  <= last_collapse_idx, "Gradient CP is beyond all collapses"
    assert cp_cusum.index <= last_collapse_idx, "CUSUM CP is beyond all collapses"
    print(f"  Last stable idx={last_stable_idx}  last collapse idx={last_collapse_idx}")

    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = plot_changepoints(result, cp_grad, cp_cusum, Path(tmpdir))
        assert fpath.exists()
        print(f"  Plot saved → {fpath.name}")

    print("Stage 4 PASSED.")

if __name__ == "__main__":
    test_stage4()
