"""Stage 6 smoke test — full report generation."""
import sys, os, tempfile, json
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from yadel.reporter import run_full

def test_stage6():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        report = run_full(det_min=-0.25, det_max=0.50, n_points=60, out_dir=out)

        print(f"  Run ID: {report['run_id']}")
        print(f"  Topology A: {report['topology_A']['stable']} stable / "
              f"{report['topology_A']['collapsed']} collapsed / "
              f"{report['topology_A']['n_disconnected_pockets']} disconnected pockets")
        print(f"  Topology B: {report['topology_B']['stable']} stable / "
              f"{report['topology_B']['collapsed']} collapsed")
        print(f"  Divergence AUC: {report['comparison']['divergence_auc']}")
        print(f"  Four checks: {report['four_checks']}")
        print(f"  All checks pass: {report['all_checks_pass']}")

        # Verify JSON was written (check while tmpdir still exists)
        assert Path(report["_json_path"]).exists(), "JSON report not written"
        # Verify it is valid JSON
        with open(report["_json_path"]) as f:
            parsed = json.load(f)
        assert parsed["run_id"] == report["run_id"]

        # Verify all four checks
        assert report["all_checks_pass"], \
            f"Not all four checks passed: {report['four_checks']}"

    print("Stage 6 PASSED.")

if __name__ == "__main__":
    test_stage6()
