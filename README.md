# Toy SSM (Mamba-style) + Official Repo

This workspace has two layers:

| Layer | Path | Purpose |
|-------|------|---------|
| **Official Mamba** | [`mamba-main/`](mamba-main/) | Production code from [state-spaces/mamba](https://github.com/state-spaces/mamba): CUDA kernels, Mamba-2/3, training |
| **Toy SSM** | [`toy_ssm/`](toy_ssm/) | Small, readable PyTorch for learning — same math, no custom CUDA |

## Learning map (toy → official)

Read the toy code first, then open the matching official file.

| Concept | Toy | Official (`mamba-main/`) |
|---------|-----|--------------------------|
| Selective scan (Algorithm 2) | [`toy_ssm/selective_scan.py`](toy_ssm/selective_scan.py) | [`mamba_ssm/ops/selective_scan_interface.py`](mamba-main/mamba_ssm/ops/selective_scan_interface.py) — see `selective_scan_ref` |
| Mamba block | [`toy_ssm/mamba.py`](toy_ssm/mamba.py) | [`mamba_ssm/modules/mamba_simple.py`](mamba-main/mamba_ssm/modules/mamba_simple.py) |
| Stacked LM | [`toy_ssm/model.py`](toy_ssm/model.py) | [`mamba_ssm/models/mixer_seq_simple.py`](mamba-main/mamba_ssm/models/mixer_seq_simple.py) |
| Mamba-2 / SSD (chunk scan) | — | [`mamba_ssm/modules/ssd_minimal.py`](mamba-main/mamba_ssm/modules/ssd_minimal.py) |
| Fast CUDA scan | — | [`csrc/selective_scan/`](mamba-main/csrc/selective_scan/) |

## What “like Mamba” means here

Each channel runs a **selective** state-space model:

1. **Input-dependent** step size Δ, and matrices B, C (from `x_proj`)
2. **Input-independent** diagonal A (S4D init) and skip D
3. **Discretization** (zero-order hold):  
   `h_t = exp(Δ·A) ⊙ h_{t-1} + (Δ·B) ⊙ x_t`, then `y_t = C^T h_t + D·x_t`
4. **Local mixing**: depthwise causal Conv1d before the SSM
5. **Gating**: `y * silu(z)` where `z` comes from the second half of `in_proj`

The toy implementation uses an explicit time loop in `selective_scan` so you can set breakpoints and inspect `h` at each step. The official repo replaces that with fused CUDA / Triton for speed.

## HW2D plasma stages (Lorenz → scalars → full fields)

| Stage | Notebook | What it does |
|-------|----------|----------------|
| 1 | `lorenz_ssm/stage1_lorenz_linear_ssm.ipynb` | Linear SSM on Lorenz attractor |
| 2 | `hw2d_ssm/stage2_hw2d_linear_ssm.ipynb` | Same `LinearSSM` on HW2D energy/enstrophy scalars |
| 3 | `hw2d_ssm/stage3_cnn_ssm.ipynb` | CNN encoder + Linear SSM + decoder on full 2D fields |

Simulation HDF5 is **not** in git (~500 MB). After clone, run the simulate cell in Stage 2 or 3 once (`run_hw2d` skips if the file already exists).

```bash
git clone https://github.com/NoahH72/TOY-SSM.git
cd TOY-SSM
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# GPU machine: install CUDA PyTorch from https://pytorch.org first
jupyter notebook hw2d_ssm/stage3_cnn_ssm.ipynb
```

## Quick start (toy Mamba)

```bash
cd TOY-SSM
pip install torch einops
python examples/demo.py
```

Compare toy vs official reference scan (uses vendored `mamba-main/` on `PYTHONPATH`):

```bash
python examples/compare_with_official.py
```

## Installing the real package (optional)

From the vendored tree (needs Linux + NVIDIA GPU + CUDA for full build):

```bash
cd mamba-main
pip install -e . --no-build-isolation
pip install causal-conv1d>=1.4.0 --no-build-isolation  # optional speedup
```

Then use it like upstream:

```python
from mamba_ssm import Mamba
```

See [`mamba-main/README.md`](mamba-main/README.md) and [`mamba-main/usage.md`](mamba-main/usage.md).

## Suggested reading order

1. `toy_ssm/selective_scan.py` — the recurrence
2. `toy_ssm/mamba.py` — how Δ, B, C are produced and wired in
3. `mamba-main/mamba_ssm/ops/selective_scan_interface.py` — `selective_scan_ref` (same loop, more edge cases)
4. `mamba-main/mamba_ssm/modules/mamba_simple.py` — production block + fast path
5. Paper: [Mamba (2312.00752)](https://arxiv.org/abs/2312.00752); for Mamba-2 see `ssd_minimal.py` + [2405.21060](https://arxiv.org/abs/2405.21060)
