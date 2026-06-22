# Predictability in Chaotic Three-Body Dynamics with Machine Learning

## Positioning

This paper should not be framed as "a bigger model forecasts a harder time series."
The stronger ICLR-style framing is:

- In chaotic three-body dynamics, what is still predictable from short observational prefixes?
- Which targets remain learnable after direct state forecasting becomes inaccurate?
- Does explicitly modeling perturbation ensembles matter more than increasing backbone complexity?

The repo already supports a compelling target hierarchy:

1. `Task E`: direct trajectory/state forecasting
2. `Task B`: final stable vs unstable outcome classification from early prefixes
3. `Task C`: multi-horizon survival curve prediction
4. `Task D`: instability/divergence timescale regression

This creates a natural scientific story:
chaos makes exact long-horizon trajectory forecasting hard, but does not eliminate all useful predictability. Coarser and more physically aligned targets remain substantially more predictable.

## Candidate Title

- What Remains Predictable in Chaotic Three-Body Dynamics?
- Beyond Trajectory Forecasting: Predictability Hierarchies in Chaotic Three-Body Systems
- Learning the Predictability Frontier of Chaotic Three-Body Dynamics

## One-Paragraph Thesis

We study machine learning prediction on hierarchical three-body systems under chaos. Rather than treating the problem purely as direct long-horizon trajectory forecasting, we compare a hierarchy of targets with increasing physical coarseness: future state prediction, final instability classification, multi-horizon survival prediction, and instability-timescale regression. Our central hypothesis is that while sensitive dependence on initial conditions rapidly degrades direct state forecasts, short trajectory prefixes still contain robust signal about macroscopic future behavior. Across several datasets with grouped perturbations, we find that target choice matters more than raw model complexity: ensemble-aware models that explicitly reason over perturbation families deliver strong gains on survival prediction, while generic forecasting baselines struggle more on direct state rollout. These results suggest that the right question for machine learning in chaotic dynamics is not only "can we forecast the trajectory?" but "what physical observables remain predictable, and at what resolution?"

## Problem Setup

### Physical Setting

- System: hierarchical three-body dynamics
- Data: multiple perturbations per underlying base system
- Regime: chaotic or near-chaotic behavior with eventual instability/unbinding for a subset of trajectories
- Observation: early prefix of trajectory or initial-condition descriptors
- Goal: predict a future property of the system

### Datasets

Three local dataset generations exist in the repo:

- `dataset/`: 8 perturbations per system, 1001 time steps, legacy notebooks
- `dataset_v2/`: 8 perturbations per system, 2001 time steps, cleaner forecasting/survival setup
- `dataset_v3/`: 32 perturbations per system, 2001 time steps, main scripted benchmark

Important split rule:

- Splits must be at the system/case level, not per trajectory
- All perturbations from the same base system stay in the same split

This should be highlighted early because it is both scientifically correct and methodologically important.

### Core Representations

- `orbital15`: orbital-element-based representation with angle encoding
- `trajectory/raw_state18`: positions and velocities for all 3 bodies
- optional richer engineered geometry features in some tasks

Working hypothesis:

- physically structured orbital features improve stability-related targets
- raw state forecasting is harder and more brittle under long-horizon chaos

## Main Claims

These claims are ambitious but still grounded in the current repo contents.

### Claim 1

Direct long-horizon forecasting is the least stable prediction target in chaotic three-body systems.

Evidence source:

- `Task E` direct forecasting baselines
- comparison of naive, Transformer, and pretrained time-series/foundation models

### Claim 2

Short trajectory prefixes retain strong predictive signal for coarse future outcomes, even when exact future state prediction is difficult.

Evidence source:

- `Task B` early-prefix instability classification with AUC around `0.97`
- `Task C` survival prediction with substantial improvement over mean baselines
- `Task D` useful rank correlation on divergence-time regression

### Claim 3

Target design matters as much as, or more than, model scale.

Evidence source:

- outcome/survival targets outperform direct forecasting in reliability
- monotone survival parameterization gives physically valid outputs
- finite-time phase-space divergence rates and threshold-crossing times are more learnable than exact phase-space rollout

### Claim 4

Modeling grouped perturbation structure explicitly is more valuable than only increasing temporal backbone sophistication.

Evidence source:

- `Task C` hierarchical/dual-attention models outperform direct case-level summaries
- explicit set-level reasoning improves prediction over flat modeling

## Suggested Contributions

A good ICLR contribution list would be:

1. A predictability-hierarchy benchmark for chaotic three-body dynamics, spanning direct forecasting, classification, survival prediction, and divergence-time regression.
2. An empirical finding that coarse physically meaningful targets remain learnable from short prefixes even when direct state forecasting degrades.
3. A demonstration that perturbation-ensemble structure is a key inductive bias for chaotic-system prediction.
4. A reproducible benchmark suite over grouped three-body datasets with case-level splits and multiple feature representations.

If results become stronger after reruns, a fifth contribution could be:

5. A systematic comparison between domain-specific models and pretrained general-purpose time-series foundation models on chaotic dynamics.

## Recommended Paper Narrative

The narrative should move from the strongest conceptual question to the most concrete experiments:

1. Chaos limits exact long-range forecasting.
2. But scientific usefulness does not require exact trajectory recovery.
3. Different future targets encode different levels of coarse-graining.
4. ML can recover meaningful predictability at the level of outcomes, survival probabilities, and instability timescales.
5. The key inductive bias is not only temporal sequence modeling, but also perturbation-set structure.

This is a cleaner story than leading with "we benchmarked several models."

## Experimental Axes

### Axis A: Prediction Target

Compare:

- direct future-state forecasting
- final instability classification
- multi-horizon survival prediction
- divergence-time regression

This should be the primary organizing axis of the paper.

### Axis B: Input Information

Compare:

- initial conditions only
- early trajectory prefixes
- grouped perturbation sets vs single trajectory only

This helps answer how much predictive signal is present in different observation budgets.

### Axis C: Representation

Compare:

- `orbital15`
- raw positions/velocities
- optional geometry-engineered features

### Axis D: Architecture / Inductive Bias

Compare:

- shallow baselines
- CNN/TCN/GRU/Transformer temporal encoders
- set-aware or hierarchical models
- pretrained foundation forecasters

## Proposed Experiment Section Structure

### Experiment 1: Direct Forecasting Benchmark

Question:
How rapidly does predictive quality degrade for exact future-state prediction?

Use:

- `Task E`
- naive baseline
- Transformer baseline
- Chronos / TimesFM / PatchTST / Panda where available

Metrics:

- normalized MSE
- raw MAE/MSE
- horizon-wise error growth

Desired takeaway:

- direct trajectory forecasting deteriorates quickly
- stronger generic forecasters do not fundamentally solve chaos

### Experiment 2: Prefix-Based Final Outcome Prediction

Question:
Can short prefixes predict eventual instability?

Use:

- `Task B`
- prefix lengths `T = 100, 200, 500`
- SVM/CNN/TCN/attention

Metrics:

- AUROC
- F1
- accuracy
- calibration if possible

Desired takeaway:

- even short prefixes contain strong outcome signal
- more temporal context improves outcome prediction

### Experiment 3: Multi-Horizon Survival Prediction

Question:
Can models estimate the probability of remaining bound across future horizons?

Use:

- `Task C`
- direct case-level baseline
- trajectory-context model
- dual-attention model

Metrics:

- MAE / RMSE on survival curves
- monotonicity satisfaction rate
- area-under-survival-curve error
- early/mid/late horizon breakdown

Desired takeaway:

- survival prediction is both learnable and scientifically interpretable
- set-aware modeling matters
- physically constrained outputs improve validity

### Experiment 4: Instability Timescale Regression

Question:
Can models estimate when instability manifests, not just whether it does?

Use:

- `Task D`
- random forest
- MLP
- CNN

Metrics:

- MAE on `log_tau`
- RMSE on `log_tau`
- Spearman rank correlation

Desired takeaway:

- timescale prediction is noisy but informative
- useful ranking of "more stable vs less stable" systems is possible

### Experiment 5: Predictability Hierarchy Summary

Question:
Which targets remain predictable at which level of granularity?

Construct a summary figure/table that compares all tasks in one place.

Possible forms:

- normalized performance relative to baseline across tasks
- performance vs target granularity
- performance vs horizon length

This should become the centerpiece figure of the paper.

## Figures To Build

### Figure 1

Problem setup diagram:

- base system
- multiple perturbations
- early observed prefix
- four prediction targets

### Figure 2

Predictability hierarchy figure:

- direct forecast error grows quickly
- outcome/survival/timescale targets remain more predictable

### Figure 3

Task B results:

- AUROC or F1 vs prefix length
- compare architectures

### Figure 4

Task C survival curves:

- true vs predicted survival curves
- baseline vs set-aware model
- monotonicity-constrained outputs

### Figure 5

Task D scatter:

- true vs predicted `log_tau`
- ideally one panel per model

### Figure 6

Ablation on perturbation structure:

- single-trajectory summary vs ensemble-aware model

## Tables To Build

### Table 1

Dataset summary:

- dataset version
- number of systems
- perturbations per system
- time steps
- main tasks

### Table 2

Main cross-task comparison:

- task
- target
- representation
- best model
- main metric
- baseline gap

### Table 3

Task C ablation:

- flat case model
- hierarchical model
- dual attention
- with/without monotone constraint if tested

### Table 4

Foundation-model forecasting comparison:

- naive
- Transformer
- Chronos
- TimesFM
- PatchTST
- Panda

## Ablation Plan

This section matters a lot for ICLR.

### Ablation 1: Prefix Length

Test whether predictability increases with observed context:

- `T=100`
- `T=200`
- `T=500`

Applies strongly to `Task B`, and optionally to `Task C` and `Task D`.

### Ablation 2: Representation

Compare:

- orbital features
- raw state
- geometry features

Main question:
is the gain coming from architecture or from physically structured representation?

### Ablation 3: Set Structure

Compare:

- pooled single-trajectory or case-average encoding
- explicit perturbation-set encoder

This is one of the most important ablations in the repo.

### Ablation 4: Physical Constraint

For `Task C`, compare:

- unconstrained survival outputs
- monotone parameterization

Main question:
does the physical constraint improve validity, stability, and downstream accuracy?

### Ablation 5: Target Choice

This is the paper's conceptual ablation.
Using the same prefix budget, compare difficulty across:

- direct forecasting
- final outcome classification
- survival curve prediction
- timescale regression

## Potential Weaknesses

These are the points reviewers are most likely to question.

### Weakness 1: Novelty Risk

If framed as "we applied standard ML models to 3-body data," novelty may feel limited.

Mitigation:

- lead with the predictability-hierarchy question
- emphasize target choice and coarse-grained predictability under chaos
- foreground ensemble structure and physical constraints

### Weakness 2: Benchmark Fragmentation

Right now the repo contains several tasks with different datasets, targets, and metrics.
Without a unifying comparison, the paper may read like four loosely connected projects.

Mitigation:

- organize everything under one benchmark philosophy
- add one unified summary figure
- standardize the framing of observed prefix, target horizon, and split rule

### Weakness 3: Insufficient Theory

An ICLR reviewer may ask for stronger analytical grounding than pure benchmarking.

Mitigation:

- discuss chaos as a limit on fine-grained forecasting but not necessarily on coarse observables
- relate target hierarchy to information loss / coarse-graining
- possibly add Lyapunov-time-inspired discussion for why different targets have different predictability horizons

### Weakness 4: Dataset Scope

The study is currently on one physical problem family.
Reviewers may ask whether conclusions generalize beyond this dataset.

Mitigation:

- position the work as a careful case study of chaotic dynamical predictability
- avoid overclaiming universality
- if possible, add multiple dataset variants as robustness checks

### Weakness 5: Foundation Model Comparison May Be Noisy

Pretrained time-series models may underperform for reasons unrelated to chaos:

- channel mismatch
- univariate assumptions
- domain gap
- local checkpoint availability

Mitigation:

- present foundation results as informative stress tests, not the sole basis of the paper
- be explicit about representation compatibility and adaptation burden

## What Is Already Strong Enough

Based on current repo notes:

- `Task B` is already strong enough to support a serious result section
- `Task C` is probably the most paper-ready component
- `Task D` gives a valuable continuous-target perspective
- `Task E` is useful as contrast, especially if it shows that exact forecasting is the wrong success criterion

If I had to pick one core result around which to center the paper, it would be:

`Task C + Task B`, with `Task E` and `Task D` used to define the broader predictability hierarchy.

## Recommended Main Paper Structure

1. Introduction
2. Related Work
3. Predictability Hierarchy in Chaotic Three-Body Systems
4. Datasets, Representations, and Split Protocol
5. Models and Physically Constrained Predictors
6. Experiments
7. Analysis and Ablations
8. Limitations
9. Conclusion

## Intro Outline

The introduction should make four moves:

1. Chaotic dynamics make long-horizon exact prediction fundamentally hard.
2. But many scientific questions concern coarse future behavior, not exact trajectory matching.
3. Existing ML evaluations often collapse these goals into a single forecasting metric.
4. We propose evaluating a hierarchy of targets and show that substantial predictability survives at coarser levels.

## Concrete Writing Angles

### Angle A: "Forecasting is the wrong benchmark"

This is bold and memorable.
Use if `Task E` clearly underperforms while `Task B/C/D` are strong.

### Angle B: "Predictability is target-dependent"

This is safer and probably the best default framing.

### Angle C: "Ensemble structure is the right inductive bias"

This is attractive if `Task C` remains the strongest section.
It can be either the main framing or a secondary contribution.

## Open Gaps Before Submission

These are likely the highest-value missing pieces.

1. Build one unified cross-task comparison figure.
2. Re-run the strongest tasks under one standardized split/reporting protocol.
3. Add calibration or uncertainty analysis for classification/survival if feasible.
4. Run stronger representation ablations.
5. Tighten the foundation-model comparison so it answers a clear question.
6. Write a clearer conceptual bridge between chaos, coarse-graining, and ML target design.

## Short Abstract Skeleton

Chaotic dynamical systems are often treated as a worst-case setting for machine learning prediction because small errors in state estimation can grow rapidly over time. We argue that this view is incomplete: in chaotic systems, the right notion of predictability depends strongly on the prediction target. Using hierarchical three-body dynamics as a testbed, we compare direct future-state forecasting against a hierarchy of coarser targets, including final instability classification, multi-horizon survival prediction, and instability-timescale regression. We find that although direct trajectory forecasting degrades rapidly, short trajectory prefixes retain substantial signal about future macroscopic behavior. Models that explicitly encode perturbation-ensemble structure outperform flatter baselines, and physically constrained survival heads produce valid monotone predictions with strong accuracy. These results suggest that machine learning for chaotic dynamics should be evaluated not only by trajectory fidelity, but by which scientifically meaningful observables remain predictable.

## Immediate Next Step

The next practical step is to turn this outline into:

1. a Chinese working outline for discussion with your advisor or collaborators
2. an English paper skeleton with section headings and bullet arguments
3. a figure checklist mapped to existing scripts and outputs

I recommend doing `2 + 3` next.
