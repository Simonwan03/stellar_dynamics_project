# Stellar Dynamics Stability Prediction

  

This project focuses on long-term stability prediction for dynamical systems and supports two tasks:

  

1. **Task A**: Probability distribution prediction over a set of perturbations under the same initial condition (outputting the probability distribution of stability $[0,1]$).

2. **Task B**: Binary classification based on the initial condition plus an early prefix of the time series (outputting a single label).

  

## 1. Problem Definition

  

### Task A: Distribution Prediction over Perturbation Sets

  

- **Input**: an set of initial conditions, which each represents a  batch of 8 perturbed trajectories corresponding to the same **initial condition**.

- **Output**: the probability distribution of future outcomes under this initial condition:

  - `P(stable)`and `P(unstable)`

- **Ground truth**: obtained by counting the final outcomes of the 8 perturbed trajectories under the same initial condition (i.e., an empirical distribution).

  - Example: if 6 out of 8 trajectories remain stable and 2 diverge, then the target distribution is `P(stable)=0.75, P(diverge)=0.25`.

  

This task essentially extends **single-trajectory classification** to **uncertainty modeling under the same initial condition**.

  

### Task B: Prefix-to-Label Classification

  

- **Input**:

  - the initial condition;

  - an early prefix of the trajectory time series for that case (for example, the first `T=100` time steps).

- **Output**: a single binary classification label:

  - `stable` (remains stable within the predefined observation window)

  - `unstable` (will later diverge / become unstable)

- The label can be defined using the existing pipeline, for example based on `time_unbound`:

  - if `time_unbound == 5000`, assign the stable class (`1`);

  - otherwise assign the unstable class (`0`).

  

This task is consistent with the current classification pipeline, but explicitly emphasizes **predicting the long-term outcome using only an early trajectory segment**.

  

## 2. Data Understanding and Organization Suggestions

In the initial condition file `./dataset/ICs_sampling_0.hdf5`, we have the following parameters:

```

[DATASET] APs shape=(40000, 2) dtype=float64

[DATASET] LANs shape=(40000, 2) dtype=float64

[DATASET] aratios shape=(40000, 1) dtype=float64

[DATASET] e_uvecs shape=(40000, 2, 3) dtype=float64

[DATASET] eccs shape=(40000, 2) dtype=float64

[DATASET] h_uvecs shape=(40000, 2, 3) dtype=float64

[DATASET] incs shape=(40000, 2) dtype=float64

[DATASET] masses shape=(40000, 3) dtype=float64

[DATASET] mratios shape=(40000, 2) dtype=float64

[DATASET] periods shape=(40000, 2) dtype=float64

[DATASET] positions shape=(40000, 3, 3) dtype=float64

[DATASET] q_uvecs shape=(40000, 2, 3) dtype=float64

[DATASET] rel_positions shape=(40000, 2, 3) dtype=float64

[DATASET] rel_velocities shape=(40000, 2, 3) dtype=float64

[DATASET] smaxes shape=(40000, 2) dtype=float64

[DATASET] tAnos shape=(40000, 2) dtype=float64

[DATASET] velocities shape=(40000, 3, 3) dtype=float64  

```

In the entire time series trajectory file `"Loading the restults_batch_sampling_0.hdf5"`, we have the following parameters:

```

[DATASET] APs_complete shape=(8000, 1001, 2) dtype=float64

[DATASET] LANs_complete shape=(8000, 1001, 2) dtype=float64

[DATASET] angMomentum_complete shape=(8000, 1001, 3) dtype=float64

[DATASET] aratios_complete shape=(8000, 1001, 1) dtype=float64

[DATASET] eccs_complete shape=(8000, 1001, 2) dtype=float64

[DATASET] energy_complete shape=(8000, 1001, 1) dtype=float64

[DATASET] incs_complete shape=(8000, 1001, 2) dtype=float64

[DATASET] masses_complete shape=(8000, 1001, 3) dtype=float64

[DATASET] mratios_complete shape=(8000, 1001, 2) dtype=float64

[DATASET] periods_complete shape=(8000, 1001, 2) dtype=float64

[DATASET] positions_complete shape=(8000, 1001, 3, 3) dtype=float64

[DATASET] smaxes_complete shape=(8000, 1001, 2) dtype=float64

[DATASET] tAnos_complete shape=(8000, 1001, 2) dtype=float64

[DATASET] times_complete shape=(8000, 1001) dtype=float64

[DATASET] velocities_complete shape=(8000, 1001, 3, 3) dtype=float64

```

The parameters will be used as features:

- 2 aratios: Semimajor Axis Ratio

- 1 mratios: Mass Ratios

- 2 eccs: Eccentricities

- 2 incs: Inclinations

- 2 APs: Arguments of Periapsis

- 2 LANs: Longitudes of Ascending Node

  

Since APs and LANs go full circle, it may be best to have them as both sines and cosines: sin(APs), cos(APs), sin(LANs), cos(LANs). This is to ensure that the machine understands that 0 = 2*pi. Since incs go half

circle only, this split is not necessary (we may consider only incs or

cos(incs) as a parameter).So after this change, there are a total of 15 parameters:

- 2 aratios

- 1 mratios:

- 2 eccs

- 2 cos(incs) or just incs

- 2 sin(APs)

- 2 cos(APs)

- 2 sin(LANs)

- 2 cos(LANs)

According to the current description, the data can be understood as having a two-level structure:

  

**Level 1 (case level)**: about `5000` system samples (or a similar scale). \

**Level 2 (perturbation level)**: each initial condition has `8` perturbed versions, used to reflect the uncertainty of outcomes under that initial condition.

  

It is recommended to unify the dataset indexing fields as follows:

  

- `case_id`: initial condition ID (8 perturbation trajectories belong to the same `case_id`)

- `perturb_id`: perturbation ID (`0~7`)

- `x0`: initial condition vector

- `trajectory`: time-series observation (in the form `[C, L]` or `[L, C]`)

- `time_unbound` / `diagnosis`: outcome label field

  

## 3. Label Construction

  

### Task A Labels (Distribution Labels)

  

For each `case_id`, aggregate the 8 perturbation trajectories:

  

- `n_stable = sum(y_i == stable)`

- `n_total = 8`

- `p_stable = n_stable / n_total`

- `p_diverge = 1 - p_stable`

  

Possible training objectives include:

  

- KL divergence / Cross-Entropy (model output distribution vs. empirical distribution)

- Or treating `p_stable` as a regression target (MSE / BCE with soft targets)

  

### Task B Labels (Single Labels)

  

Construct labels independently for each sample:

  

- `y = 1` (**stable**) if `time_unbound == 5000`

- `y = 0` (**unstable**) otherwise

  

The input only uses the prefix `trajectory[:, :T]` (for example, `T=100`).

  
  


## 4. Baseline Implementation

  

### Task A Baseline

3 models have been tested including MLP, Transformer and XGBoost,

#### MLP 3-layers [512,256,128]

`epoch 017 | train_mse 0.064713 | test_mse 0.098856 | test_mae 0.226948` 

`epoch 200 | train_mse 0.008179 | test_mse 0.123569 | test_mae 0.222359` (overfitting)

#### Transformer (encoder-only Transformer + CLS pooling + sigmoid regression head)

`epoch 066 | train_mse 0.085248 | test_mse 0.115305 | test_mae 0.244214`

`epoch 200 | train_mse 0.023949 | test_mse 0.141674 | test_mae 0.244664`

  

#### XGBoost

`XGB | TRAIN: mse 0.005598 | mae 0.053982 | r2 0.967315` 

`XGB | TEST : mse 0.098321 | mae 0.243547 | r2 0.427884`

  

### Task B Baseline

For task B, we have tested linear SVM, 1D-CNN, TCN, and Transformer

#### Linear SVM / 200 epochs

**T=100** 

`train_acc: 0.8500` 

`test_acc : 0.8083`

  

**T=200** 

`train_acc: 0.8955` 

`test_acc : 0.8420`

  

**T=500**

  

#### 1D-CNN / 50 epochs

**T=100**

`TEST | AUC 0.9544 | F1 0.8284 | Acc 0.8674` 

**T=200**

`TEST | AUC 0.9630 | F1 0.8568 | Acc 0.8886` 

**T=500**

`TEST | AUC: 0.9783 | F1: 0.9087 | Acc: 0.9308`

  

#### TCN Temporal Convolutional Network / 20 epochs

**T=100**

`TEST | AUC: 0.9495 | F1: 0.8282 | Acc: 0.8682` 

**T=200**

`TEST | AUC: 0.9603 | F1: 0.8468 | Acc: 0.8750`

**T=500**

`TEST | AUC: 0.9795 | F1: 0.9155 | Acc: 0.9350`
  
  

#### Attention / 50 epochs

**T=100**

`TEST | AUC 0.9515 | F1 0.8307 | Acc 0.8670` 

**T=200**

`TEST | AUC 0.9637 | F1 0.8648 | Acc 0.8974` 

**T=500**

`TEST | AUC 0.9783 | F1 0.9210 | Acc 0.9410`

#### GNN / 20 epochs
**T=100**
0.951738238317718 0.8307851775071262 0.8694

## 5. Task C
Instead of predicting whether one specific trajectory will remain stable in the future, predict what fraction of a small ensemble of perturbed trajectories around the same nominal initial condition remains stable over time.
This reformulates the task from single-trajectory outcome prediction into time-dependent distribution forecasting over a perturbation set.
This is a very reasonable change, because it matches the nature of the problem much better:
- the three-body problem is highly sensitive to perturbations,
- nearly identical initial conditions can lead to very different long-term outcomes,
- so a single hard label may be intrinsically unstable,
- while the statistical behavior of a neighborhood of initial conditions is much more robust and learnable.

### 1. Formal task definition

Suppose you have a nominal initial condition $c$.

Around this nominal condition, you generate or already have $M$ perturbed trajectories:

$$
\tau^{(1)}, \tau^{(2)}, \dots, \tau^{(M)}
$$

For each trajectory, define

$$
s_t^{(m)} \in \{0,1\}
$$

to indicate whether the $m$-th perturbed trajectory is still stable at time $t$:

- $1$: stable
- $0$: unstable / escaped / unbound

Then define the **survival proportion** at time $t$:

$$
p_t = \frac{1}{M} \sum_{m=1}^{M} s_t^{(m)}
$$

This means:

**among all perturbed trajectories associated with this nominal condition, what fraction is still stable at time $t$?**

So the label is no longer a single binary value. Instead, it is a curve:

$$
\mathbf{p} = (p_{t_1}, p_{t_2}, \dots, p_{t_K})
$$

For example, you may choose

$$
t_1 = 200,\quad t_2 = 400,\quad t_3 = 600,\quad t_4 = 800,\quad t_5 = 1000
$$

Then the target becomes

$$
\mathbf{p} = (p_{200}, p_{400}, p_{600}, p_{800}, p_{1000})
$$

Each value lies in $[0,1]$.

## Difference between naive probability regression and a Binomial likelihood

At first glance, both approaches seem to predict the same quantity: a survival probability $p \in [0,1]$.  
However, they differ in how they interpret the supervision signal.

### Naive probability regression

A naive approach is to treat the empirical survival fraction

$$
\hat y = \frac{N}{8}
$$

as an ordinary continuous target in \([0,1]\), and train the model with a regression loss such as mean squared error:

$$
\mathcal{L}_{\text{MSE}} = (p - \hat y)^2.
$$

Under this view, a target such as $$5/8 = 0.625$$ is handled as if it were an exact continuous value that the model should directly approximate.

This is convenient, but it ignores the fact that the observed label is not a true probability. It is only the empirical outcome of a small discrete ensemble of 8 perturbed trajectories.

### Binomial likelihood

A more principled alternative is to model the observed label as a **count**.

Let $N \in \{0,1,\dots,8\}$ denote the number of perturbed trajectories that remain stable up to a given horizon.  
The model predicts a latent survival probability $p$, and the observed count is modeled as

$$
N \mid x \sim \mathrm{Binomial}(8, p).
$$

The corresponding likelihood is

$$
\Pr(N \mid 8, p) = \binom{8}{N} p^N (1-p)^{8-N},
$$

and training is performed by minimizing the negative log-likelihood.

This interpretation is more faithful to the data-generating process: the supervision comes from a small set of binary survival outcomes, not from a truly continuous target.

### What is the essential difference?

The key distinction is:

- **Naive regression** treats $N/8$ as a continuous label.
- **Binomial modeling** treats $N$ as a discrete random count generated by an underlying probability $p$.

In other words, the question is not only *what number should be predicted*, but also *what kind of observation produced the label*.
`
### Relation to soft-label binary cross-entropy

Because the ensemble size is fixed at 8, the Binomial negative log-likelihood can be written as
$$
-\log \Pr(N \mid 8, p)
= -N \log p - (8-N)\log(1-p) + \text{const}.
$$
Dividing by 8 gives

$$
-\frac{N}{8}\log p - \left(1-\frac{N}{8}\right)\log(1-p),
$$

which is exactly the binary cross-entropy form with a soft target \(N/8\), up to an additive or multiplicative constant.

Therefore, in this fixed-size setting:

- **Binomial likelihood**
- **soft-label BCE with target \(N/8\)**

are essentially equivalent from an optimization point of view.

### Practical implication

So the real modeling difference is not mainly between “probability prediction” and “Binomial likelihood,” since the model still predicts a probability \(p\) in both cases.

The more important distinction is between:

1. **naive continuous regression** (for example MSE on \(N/8\)), and  
2. **count-aware probabilistic modeling** (Binomial likelihood, or equivalently soft BCE when \(M=8\) is fixed).

The second view is preferable here because it respects the discrete nature of the supervision signal and better matches the interpretation of the data as repeated survival trials over a perturbation ensemble.

### Caveat

The Binomial assumption is still an approximation.  
The 8 perturbed trajectories are generated from the same nominal initial condition, so they are not guaranteed to be strictly independent or identically distributed. For this reason, the Binomial likelihood should be understood as a practical count-aware model rather than an exact physical law.

If dependence or heterogeneity among perturbations becomes important, richer models such as the **Beta-Binomial** or **Poisson-Binomial** may be explored.