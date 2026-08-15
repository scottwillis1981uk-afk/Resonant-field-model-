"""Stage 1 smoke test — verify canonical v0.2 model."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from yadel.core import DEFAULTS, make_noise, simulate, classify, K_A, K_B, G0, SIGMA, DELAYS

def test_stage1():
    # ---- Verify canonical parameter values ----
    assert K_A == 0.55,  f"k_a should be 0.55, got {K_A}"
    assert K_B == 0.48,  f"k_b should be 0.48, got {K_B}"
    assert G0  == 0.80,  f"g0 should be 0.80, got {G0}"
    assert SIGMA == 0.002, f"sigma should be 0.002, got {SIGMA}"
    assert DELAYS["A"]["da"] == 5  and DELAYS["A"]["db"] == 11
    assert DELAYS["B"]["da"] == 7  and DELAYS["B"]["db"] == 13
    print("  Canonical parameters verified: k_a=0.55 k_b=0.48 g0=0.80 sigma=0.002")
    print("  Topology A: da=5 db=11  |  Topology B: da=7 db=13")

    params = dict(DEFAULTS)
    rng    = np.random.default_rng(params["seed"])
    noise  = make_noise(params["n_samples"], params["sigma"], rng)

    # det=0.0 is in the collapsed zone (origin is unstable)
    x = simulate(detuning=0.0, noise=noise, params=params)
    c = classify(x, params)
    print(f"  detuning=0.00 → class={c}")
    assert c == "collapsed", f"Expected collapsed at origin, got {c}"

    # det=0.40 is inside the stable disconnected pocket (Topo A pocket starts ~0.31)
    x = simulate(detuning=0.40, noise=noise, params=params)
    c = classify(x, params)
    print(f"  detuning=0.40 → class={c}")
    assert c == "stable", f"Expected stable in pocket, got {c}"

    # det=-0.20 is in the left origin-adjacent stable zone
    x = simulate(detuning=-0.20, noise=noise, params=params)
    c = classify(x, params)
    print(f"  detuning=-0.20 → class={c}")
    assert c == "stable", f"Expected stable in left zone, got {c}"

    print("Stage 1 PASSED.")

if __name__ == "__main__":
    test_stage1()
