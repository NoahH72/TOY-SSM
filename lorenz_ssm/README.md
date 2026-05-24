# Lorenz linear SSM (learning toy)

CPU-friendly 1-step-ahead prediction on Lorenz-63. Swap `lorenz_ssm/data/lorenz.py` for plasma time series later; keep `LinearSSM` and the training loop.

## Run

```bash
cd "/Users/noahsmac/toy ssm"
pip install torch scipy matplotlib numpy
python3 -m lorenz_ssm.train
```

Plots land in `lorenz_ssm/outputs/`.

## Layout

| File | Role |
|------|------|
| `data/lorenz.py` | `scipy.integrate.solve_ivp` trajectory + normalize |
| `models/linear_ssm.py` | Learnable A, B, C, D |
| `train.py` | Training loop + matplotlib figures |

## Next step (your roadmap)

Input-dependent Δ, B, C (Mamba-style selective scan) on top of this baseline — see `toy_ssm/` and `mamba-main/` in the parent repo.
