# Resonant Field Model (RFM)

First public disclosure: @HadrianMusic — December 3 2025, 00:27 GMT

Reality is one complex scalar field ϕ on ℝ × 𝕋³.  
No fundamental particles or forces.

The equation:  
∂²ϕ/∂t² − c²∇²ϕ + αϕ + β|ϕ|²ϕ + γ ϕ(t−τ) + η ∂ϕ/∂t = 0

Everything emerges: mass (knots at γτ≈1), gravity (∇α bend), charge (chirality), consciousness (nested phase-lock), 1/f, fine-structure, dark sector.

Quantitative hits:  
- Knot stability R²=0.998  
- Gravity ≤1.8% GR weak field  
- 1/f exact over 23 decades  
- DEAP EEG conscious cluster γτ≈1.00±0.05  
- α convergence to 1/137.04

<$50k experiments pending (plasma sidebands, metamaterial deflection).

Code, sims, Lagrangian in this repo.  
Run it yourself.

One field. One equation. The rest is music. 🎛️

## YADEL stability mapper

YADEL (Yet Another Delay Envelope Lab) is the reproducible numerical harness
for mapping the delayed nonlinear model's stability envelope. It currently:

- sweeps detuning for the canonical `(5, 11)` and `(7, 13)` delay topologies;
- classifies stable, fragile, and collapsed regimes;
- detects disconnected stability pockets and structural changepoints;
- compares topology envelopes; and
- writes plots and a machine-readable JSON report under `yadel_output/`.

The canonical YADEL v0.2 parameters are defined in `yadel/core.py`:
`k_a=0.55`, `k_b=0.48`, `g0=0.80`, and `sigma=0.002`.

### Set up

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Validate

The stage checks are executable smoke-test scripts:

```powershell
$env:PYTHONIOENCODING='utf-8'
Get-ChildItem yadel\test_stage*.py | Sort-Object Name | ForEach-Object {
    .\.venv\Scripts\python.exe $_.FullName
}
```

All six scripts must exit successfully. Running `unittest discover` is not a
substitute: these are plain test functions and are not `unittest.TestCase`
classes.

### Run the full demonstration

```powershell
.\.venv\Scripts\python.exe yadel_demo.py
```

Generated files are intentionally ignored by Git. The source, canonical
parameters, tests, and dependency manifest remain version controlled.
