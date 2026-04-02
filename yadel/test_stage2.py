"""Stage 2 smoke test — envelope sweep and plot."""
import sys, os, tempfile
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from yadel.sweep import run_sweep, plot_envelope

def test_stage2():
    result = run_sweep(det_min=-0.25, det_max=0.50, n_points=40, topology="A")

    assert len(result.labels) == 40
    assert "stable"    in result.labels, "No stable points found in sweep"
    assert "collapsed" in result.labels, "No collapsed points found in sweep"
    print(f"  Label counts: { {lb: result.labels.count(lb) for lb in ['stable','fragile','collapsed']} }")

    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = plot_envelope(result, Path(tmpdir))
        assert fpath.exists(), "Envelope plot not saved"
        print(f"  Plot saved → {fpath.name}")

    print("Stage 2 PASSED.")

if __name__ == "__main__":
    test_stage2()
