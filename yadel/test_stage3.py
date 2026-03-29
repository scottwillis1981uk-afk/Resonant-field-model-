"""Stage 3 smoke test — pocket detector."""
import sys, os, tempfile
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from yadel.sweep import run_sweep
from yadel.pockets import detect_pockets, plot_pockets

def test_stage3():
    result = run_sweep(det_min=0.0, det_max=3.0, n_points=60, topology="A")
    report = detect_pockets(result)

    print(f"  Pockets found: {len(report.pockets)}")
    for pk in report.pockets:
        kind = "DISCONNECTED" if pk.is_disconnected else "origin-adjacent"
        print(f"    [{kind}] det={pk.det_start:.3f}–{pk.det_end:.3f}  n={pk.n_points}")
    print(f"  Disconnected pockets: {report.n_disconnected}")
    print(f"  Non-monotone indices: {report.non_monotone_idxs[:5]}")

    assert report.n_disconnected >= 1, "Expected at least one disconnected pocket"

    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = plot_pockets(report, Path(tmpdir))
        assert fpath.exists()
        print(f"  Plot saved → {fpath.name}")

    print("Stage 3 PASSED.")

if __name__ == "__main__":
    test_stage3()
