"""Stage 5 smoke test — topology comparator."""
import sys, os, tempfile
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from yadel.comparator import compare_topologies, plot_comparison

def test_stage5():
    comp = compare_topologies(det_min=-0.25, det_max=0.50, n_points=60)

    print(f"  Divergence AUC: {comp.divergence_auc:.4f}")
    print(f"  Topology A labels: { {lb: comp.sweep_A.labels.count(lb) for lb in ['stable','fragile','collapsed']} }")
    print(f"  Topology B labels: { {lb: comp.sweep_B.labels.count(lb) for lb in ['stable','fragile','collapsed']} }")

    # Envelopes must differ
    assert comp.divergence_auc > 0.0, "Topologies produced identical envelopes"

    # Both topologies must have stable AND collapsed regions
    for sweep in (comp.sweep_A, comp.sweep_B):
        assert "stable"    in sweep.labels, f"Topology {sweep.topology} has no stable points"
        assert "collapsed" in sweep.labels, f"Topology {sweep.topology} has no collapsed points"

    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = plot_comparison(comp, Path(tmpdir))
        assert fpath.exists()
        print(f"  Plot saved → {fpath.name}")

    print("Stage 5 PASSED.")

if __name__ == "__main__":
    test_stage5()
