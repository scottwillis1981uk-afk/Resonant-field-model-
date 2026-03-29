"""Stage 1 smoke test — verify core simulation and classify work."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from yadel.core import DEFAULTS, make_noise, simulate, classify

def test_stage1():
    params = dict(DEFAULTS)
    rng    = np.random.default_rng(params["seed"])
    noise  = make_noise(params["n_samples"], params["noise_amplitude"],
                        params["noise_burst_prob"], rng)

    # detuning=0 is in the chaotic/collapsed region for the Ikeda map
    x_col = simulate(detuning=0.0, noise=noise, params=params)
    c_col  = classify(x_col, params)
    print(f"  detuning=0.00 → class={c_col}")
    assert c_col == "collapsed", f"Expected collapsed at origin, got {c_col}"

    # detuning=0.4 is inside the stable pocket
    x_stab = simulate(detuning=0.4, noise=noise, params=params)
    c_stab  = classify(x_stab, params)
    print(f"  detuning=0.40 → class={c_stab}")
    assert c_stab == "stable", f"Expected stable in pocket, got {c_stab}"

    # detuning=2.0 is back in the collapsed regime
    x_col2 = simulate(detuning=2.0, noise=noise, params=params)
    c_col2  = classify(x_col2, params)
    print(f"  detuning=2.00 → class={c_col2}")
    assert c_col2 == "collapsed", f"Expected collapsed, got {c_col2}"

    print("Stage 1 PASSED.")

if __name__ == "__main__":
    test_stage1()
