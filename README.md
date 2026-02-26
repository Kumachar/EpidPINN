# Metapopulation PINN Example

This repository contains a worked example of a Physics-Informed Neural Network (PINN) for a metapopulation SIR(+D) / SEIR(+D) epidemic model with unknown inter-patch movement.  
The full workflow (simulation → training → evaluation/plots) lives in the notebook:

- `metapop_model_PINN_example.ipynb`

An accompanying Conda environment file is provided:

- `EpidPINN.yml`

---

## What the notebook does

### 1) Simulate metapopulation epidemic data (CTMC / Gillespie-style)
The notebook implements a continuous-time Markov chain simulator with deaths:

- Within-patch dynamics:
  - SIR+D: `S → I → R`, and `I → D`
  - SEIR+D: `S → E → I → R`, and `I → D`
- Between-patch movement:
  - Movement applies to living compartments (S/E/I/R).
  - Dead (D) do not move.
- Movement can be:
  - `None` (no movement),
  - a constant `(P,P)` rate matrix,
  - a piecewise-constant `(K,P,P)` tensor over time bins,
  - or a callable `m(t) -> (P,P)`.

The simulator outputs daily sampled time series per patch and writes them to text files:

- `Infectious_p{k}.txt`
- `Recovered_p{k}.txt`
- `Death_p{k}.txt`
- `Incidence_p{k}.txt` (new infections per day)
- `Mortality_p{k}.txt` (new deaths per day)

Optionally, it also writes totals across patches: `*_total.txt`.

Default output directory in the notebook:
- `PINN/Data/`

### 2) Train a fractional metapopulation PINN (TensorFlow 1.x style)
The core model is implemented as a TensorFlow class:

- `PhysicsInformedNN_Metapop`

Key modeled quantities:
- Patch-level epidemic states: S(t), I(t), R(t), D(t) per patch.
- Per-patch recovery and death rates:
  - `gamma_p` (recovery), `mu_p` (death), constrained positive via `softplus`.
- Per-patch transmission:
  - `beta_p(t)` (time-varying, NN head with P outputs) and/or a bounded constant β vector.
- Unknown movement matrix:
  - `M(t)` parameterized as piecewise-constant over `B_bins` time bins.
  - Nonnegative, with diagonal masked to zero.
  - Optional soft constraints/regularizers: symmetry prior, row-sum caps, max-edge caps.

Fractional-order dynamics:
- The notebook constructs Jacobi polynomial features and uses a residual grid to build a fractional-derivative PINN objective (as in fractional PINN formulations).

Training loop:
- Adam warmup + optional L-BFGS refinement (controlled by `LBFGS` in the training cell).
- Predictions and learned parameters are exported to a timestamped folder:
  - `PINN/ResultsMetapop-<MM-DD>/`

### 3) Visualize fit and learned movement
The notebook includes plotting helpers (defined in-notebook) to produce:
- Patch-level overlays: true vs predicted I/R/D
- Totals overlays across patches
- Heatmaps of movement matrices (true vs inferred)
- β(t) plots (or constant β bar charts), and γ/μ summaries

---

## Repository layout (as used in the notebook)

```
.
├── metapop_model_PINN_example.ipynb
├── EpidPINN.yml
└── PINN/
    ├── Data/                # generated simulation series (*.txt)
    └── ResultsMetapop-*/    # learned states/parameters + inferred movement
```

---

## Environment setup

The Conda environment is defined in `EpidPINN.yml` (environment name: `EpidSim`). It includes both Conda and Pip-installed packages.

### Create the Conda environment
```bash
conda env create -f EpidPINN.yml
conda activate EpidSim
```

### Key dependencies (from `EpidPINN.yml`)
| Package | Version | Source |
|---|---:|---|
| `python` | `3.7.12` | `conda` |
| `numpy` | `1.18.5` | `conda` |
| `scipy` | `1.4.1` | `conda` |
| `pandas` | `1.1.5` | `conda` |
| `matplotlib` | `3.3.4` | `conda` |
| `notebook` | `6.5.7` | `conda` |
| `tensorflow` | `1.15.5` | `pip` |
| `tensorboard` | `1.15.0` | `pip` |
| `h5py` | `2.10.0` | `pip` |
| `protobuf` | `3.19.6` | `pip` |


### Full dependency list
The YAML contains a large set of Jupyter/runtime packages. For a complete, exact list, see `EpidPINN.yml`:
- Conda dependencies: 153 entries
- Pip dependencies: 19 entries (includes TensorFlow 1.15.5)

> Note: The notebook uses TF1-style graph/session APIs (e.g., `tf.Session`, `tf.reset_default_graph`). The provided YAML already pins a compatible TensorFlow (`tensorflow==1.15.5`) under the `pip:` section.


---

## Quickstart (run the notebook end-to-end)

1. Open Jupyter
   ```bash
   jupyter notebook
   ```

2. Run `metapop_model_PINN_example.ipynb` top-to-bottom
   - Cell group 1: defines the CTMC simulator and file writers
   - Cell group 2: sets P, initial conditions, and movement; generates data into `PINN/Data/`
   - Cell group 3+: defines the PINN class, loads data, trains, exports outputs
   - Final cells: plotting and model diagnostics

---

## Key configuration points (edit in the notebook)

### Simulation
- `P`: number of patches
- `init`: initial state array
  - SIR: shape `(P,4)` as `[S, I, R, D]`
  - SEIR: shape `(P,5)` as `[S, E, I, R, D]`
- `movement`: movement rate specification (None / matrix / bins / callable)
- `T_days`: simulation duration in days
- `out_dir`, `out_prefix`: where files are saved (defaults: `PINN/Data`, prefix like `sim_`)

### Training
- `data_dir`, `prefix`: where to read the per-patch `.txt` files
- `sf`: scaling factor for numerical conditioning (the notebook scales I/R/D/Mortality and N)
- `N_vec`: per-patch population sizes (scaled consistently with `sf`)
- Network widths:
  - `layers`: state network outputting `4*P` states
  - `layers_Beta`: β network outputting `P` values
- Movement bins:
  - `B_bins`: number of piecewise-constant movement matrices (1 = constant over time)
- Regularization / constraints (passed into `PhysicsInformedNN_Metapop`):
  - `lambda_M`, `lambda_beta_smooth`, `lambda_M_sym`, `lambda_M_row`, `M_row_cap`, `M_max`, `lambda_kappa`, etc.
- Optimization:
  - `nIter` (Adam iterations), `warmup_iters`, `adam_lr`
  - `use_lbfgs` switch

---

## Outputs

After training, the notebook writes (at minimum):

- `S.txt, I.txt, R.txt, D.txt`  — predicted trajectories (shape `(T,P)`)
- `Beta.txt`                    — predicted β per patch (shape `(T,P)` in export)
- `Kappa1.txt ... Kappa4.txt`    — fractional parameters (shape `(T,1)` in export)
- `M_bin{b}.txt`                — inferred movement matrix per bin (shape `(P,P)`)

Saved under:
- `PINN/ResultsMetapop-<MM-DD>/`
