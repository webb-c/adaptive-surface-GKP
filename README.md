# Adaptive Surface-GKP Code Simulator

Python implementation for Monte Carlo simulation of the surface-GKP code with adaptive stabilizer measurement scheduling and minimum-weight perfect matching (MWPM) decoding.

## Project Structure

- `main.py`: Entry point that parses arguments, runs simulations, and outputs results.
- `simulator.py`: Core simulation classes implementing different measurement strategies.
  - `SurfaceGKPDefaultTrackingSimulator`: Standard measurement strategy with all stabilizers measured every round.
  - `SurfaceGKPInfoMeasureWeightTrackingSimulator`: Adaptive strategy that selects optimal measurement policies per round.
  - `SurfaceGKPInfoMeasureWeightTrackingMeasureSimulator`: Adaptive strategy with configurable forced measurement/skip rounds.
- `simulator_base.py`: Base classes and shared functionality for simulators.
- `decoder.py`: Minimum-weight perfect matching (MWPM) decoder implementation.
- `graph.py`: Graph structures and utilities for decoding algorithms.
- `noise.py`: Noise models for GKP codes and circuit operations.
- `utils.py`: Utility functions for simulation, argument parsing, and result saving.

## Dependencies

- Python 3+
- NumPy
- NetworkX (for graph operations)
- Matplotlib (for visualization, if needed)

```bash
pip install numpy networkx matplotlib
```

## Usage

Run the Monte Carlo simulation via the command line:

```bash
python main.py [options]
```

### Examples

**1. Standard surface-GKP code:**

```bash
python main.py --distance 3 --n_round 3 --mode default_tracking --n_iter 1000
```

**2. Adaptive measurement with distance-5 code:**

```bash
python main.py --distance 5 --n_round 5 --mode adaptive --n_iter 1000
```

**3. Adaptive with forced measurements on specific rounds:**

```bash
python main.py --mode adaptive_meas --target_rounds "first,last" --skip_rounds "middle"
```

### Arguments

| Argument | Short | Type | Default | Description |
|---|---|---|---|---|
| `--distance` | `-d` | `odd int ≥ 1` | `3` | Surface code distance |
| `--n_round` | `-r` | `int` | `3` | Number of noisy surface stabilizer measurement rounds |
| `--n_iter` | `-n` | `int ≥ 1` | `1` | Number of Monte Carlo iterations |
| `--sigma` | `-s` | `float` | `1.0` | Circuit-level noise standard deviation |
| `--sigma_GKP` | `-g` | `float` | `1.0` | GKP state preparation noise standard deviation |
| `--sigma_idle` | `-i` | `float` | `= sigma` | Idle noise standard deviation |
| `--mode` | `-m` | `str` | `default_tracking` | Measurement strategy (see modes below) |
| `--with_info` | `-w` | `bool` | `True` | Use syndrome information in MWPM weights |
| `--clean` | `-c` | `bool` | `False` | Noiseless GKP inner-code measurements |
| `--workers` | — | `int` | `0` | Parallel workers (0=auto, 1=serial) |
| `--verbose` | — | `bool` | `False` | Print detailed per-round information |
| `--log` | — | `bool` | `True` | Save results to CSV |
| `--csv_path` | — | `str` | `results/results.csv` | Output CSV file path |

### Measurement Modes

| Mode | Description |
|---|---|
| `default_tracking` | Measures all stabilizers every round with noise-tracking MWPM weights |
| `adaptive` | Selects optimal measurement policy per round (Z-only, X-only, both, or neither) |
| `adaptive_meas` | Adaptive with configurable forced measurement/skip rounds |
| `adaptive_round` | Adaptive with forced full measurement every `d`-th round |
| `skip` | Skips all stabilizer measurements except the final round |
| `skip_round` | Measures only at distance-multiples and final round |


## BibTex
```

```