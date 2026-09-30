# Analysis Scripts

This directory contains scripts for statistical analysis and visualization of existing three-body simulation data. They read HDF5 initial conditions or trajectory results and the completion table containing `time_unbound`; they do not train predictive models. By default, each base system consists of 32 perturbed trajectories. The scripts select complete, valid systems before examining how simulation outcomes relate to initial conditions.

## What Each Script Does

| Script | Main question | Main outputs |
| --- | --- | --- |
| [`audit_system_level_statistics.py`](audit_system_level_statistics.py) | What are the overall data distributions? | Stable fraction per system, failure counts by time horizon, survival curve, correlations with initial conditions, excluded records, summary tables, and figures. |
| [`plot_ma_boundary.py`](plot_ma_boundary.py) | How do simulated outcomes relate to the theoretical Mardling–Aarseth (MA) stability boundary? | A scatter plot of MA stability margin versus mutual inclination, colored by the observed stable fraction among each system's 32 trajectories. |
| [`analyze_local_stable_fraction_disagreement.py`](analyze_local_stable_fraction_disagreement.py) | Do systems with similar initial conditions have similar stable fractions? | Nearest-neighbor disagreement in standardized `orbital15` space, trends near the boundary, a per-system CSV, and diagnostic figures. |

### System-Level Statistical Audit

`audit_system_level_statistics.py` identifies systems for which all 32 perturbed trajectories have valid completion records. For each system, it computes the stable fraction `p`, the number of failures by selected times, and `4p(1-p)`, which is largest when stable and unstable outcomes are mixed within a system. It also examines failure-time distributions, a Kaplan–Meier survival curve, and correlations between stable fraction and initial-condition quantities such as eccentricity, mutual inclination, mass ratio, and MA stability margin.

By default, it reads `sampling_2` and writes to `results/system_level_audit_sampling_2/`. Representative files include `system_level_statistics.csv`, `survival_curve.csv`, `coarse_ic_correlations.csv`, `excluded_systems.csv`, `summary.md`, and PNG/PDF figures. The `--fail-horizons` values are absolute `time_unbound` times, not trajectory prefix lengths.

### Empirical Stability Boundary Plot

`plot_ma_boundary.py` plots one point for each complete system. The horizontal axis is the logarithmic MA stability margin, the vertical axis is mutual inclination, and the color shows the fraction of its 32 trajectories that remain stable at `--time-horizon`. The zero-margin line marks the theoretical boundary and helps show where simulated outcomes agree or disagree with the criterion.

By default, it reads `sampling_3` and saves `figures/ma_margin_vs_mutual_inclination.png` and `.pdf`. If no separate initial-condition file exists, it uses the first time step of variation 0 from the result shards. It does not average angular variables across the 32 trajectories.

### Stable-Fraction Disagreement Among Nearby Systems

`analyze_local_stable_fraction_disagreement.py` builds a 15-dimensional initial `orbital15` feature vector for each system, standardizes the features across systems, and finds nearest neighbors by Euclidean distance. Its main statistic is the mean absolute difference between a system's stable fraction and those of its `k` nearest neighbors:

```text
L_i(k) = mean_{j among the k nearest neighbors of system i} |p_i - p_j|
```

The script compares this disagreement with distance from the MA boundary and checks whether neighbor distance could explain the observed disagreement. By default, it analyzes `sampling_3`, writes `local_stable_fraction_disagreement.csv` and a trend table to `results/analysis/`, and writes the main plot, a disagreement histogram, and a neighbor-distance diagnostic to `figures/`. `L_i(k)` measures local outcome disagreement; it is not a Lyapunov exponent, a measure of chaos strength, or a fractal-dimension estimate.

## Running the Scripts

Run these commands from the repository root. The selected data directories must contain the relevant HDF5 files and `completed_systems_sampling_*.txt`:

```bash
python analysis/audit_system_level_statistics.py --dataset-dir dataset_v3 --sampling-id 2
python analysis/plot_ma_boundary.py --dataset-dir dataset_sampling_3_prefix1024 --sampling-id 3
python analysis/analyze_local_stable_fraction_disagreement.py --dataset-dir dataset_sampling_3_prefix1024 --sampling-id 3
```

The scripts can run independently. To explore the results, start with the overall audit, inspect the theoretical boundary plot, and then use the nearest-neighbor analysis to examine whether systems with similar initial conditions have different outcomes.
