# Methodology

## 1. Problem formulation

We study predictability in hierarchical three-body systems through a hierarchy of learning targets. Each base system is represented by a set of nearby trajectories generated from controlled perturbations of the same underlying initial configuration. Instead of treating long-horizon state forecasting as the only objective, we consider five complementary tasks: prediction from initial conditions (Task A), early-warning classification from short trajectory prefixes (Task B), multi-horizon survival prediction (Task C), finite-time phase-space divergence regression (Task D), and direct multivariate trajectory forecasting (Task E).

Let \(s\in\{1,\ldots,S\}\) index base systems, \(v\in\{1,\ldots,V\}\) index perturbation variations, and \(t\) denote simulation time. In the main dataset, \(S=5000\) and \(V=32\), giving \(160{,}000\) simulated trajectories. The observed sequence for variation \(v\) of system \(s\) is written as

\[
\mathbf{x}_{s,v,1:T}\in\mathbb{R}^{T\times C},
\]

where \(C\) is the number of input channels. The tasks differ in the amount of observed information and in the physical resolution of the prediction target. This design allows us to test which properties remain predictable as the target changes from exact future states to increasingly coarse system-level outcomes.

## 2. Dataset construction

### 2.1 Simulation data

The dataset contains \(160{,}000\) trajectories grouped into \(5000\) base systems, with \(32\) perturbation variations per system. Each trajectory contains \(2001\) recorded time steps. The trajectory index is mapped to its system and perturbation identifiers by

\[
s=\left\lfloor\frac{r}{32}\right\rfloor,\qquad
v=r\bmod 32,
\]

where \(r\) is the global trajectory index. Time-series data are stored in 20 HDF5 shards of 8000 trajectories each. A separate HDF5 file stores static initial conditions, and a completion table provides the unbinding time \(t_{\mathrm{ub}}\) and numerical diagnostic status for each trajectory.

The principal orbital variables are the semimajor-axis ratio, two mass ratios, two eccentricities, two inclinations, two arguments of periapsis, and two longitudes of ascending node. The dataset additionally provides masses, semimajor axes, periods, positions, velocities, energy, and angular momentum. Position and velocity arrays contain three Cartesian coordinates for each of the three bodies.

### 2.2 Validity filtering

Unless otherwise specified, a trajectory is retained only when its completion-table entry satisfies

\[
t_{\mathrm{ub}}\geq 0,\qquad \mathrm{diagnosis}=0.
\]

Samples containing non-finite values in the features required by a task are removed before model fitting. System-level tasks retain only complete groups containing all 32 valid perturbations. After this filtering, 159,953 trajectories have valid completion records and 4,998 systems retain complete 32-trajectory groups.

### 2.3 Input representations

Our main compact representation, denoted **orbital15**, contains 15 channels:

\[
\mathbf{x}^{\mathrm{orb}}=
\left[
a_{\mathrm{ratio}},
\mathbf{m}_{\mathrm{ratio}},
\mathbf{e},
\mathbf{i},
\sin\boldsymbol{\omega},
\cos\boldsymbol{\omega},
\sin\boldsymbol{\Omega},
\cos\boldsymbol{\Omega}
\right].
\]

Here \(a_{\mathrm{ratio}}\) contributes one channel; mass ratios, eccentricities, and inclinations contribute two channels each; and the sine/cosine encodings of the two periapsis and ascending-node angles contribute four channels each. Periodic angles are encoded with sine and cosine to remove the artificial discontinuity at \(2\pi\). Inclinations are retained in their original form.

For the wider prefix representation used by Task C, we concatenate orbital ratios, masses, eccentricities, semimajor axes, energy, angular momentum, angle-encoded orbital variables, and inclinations. We further append nine pairwise geometric channels: relative distance, relative speed, and radial relative velocity for each of the three body pairs. A normalized time coordinate is also included. This gives 36 channels under the default configuration.

For direct Cartesian forecasting, the **raw-state18** representation contains the three-dimensional positions and velocities of all three bodies:

\[
\mathbf{x}^{\mathrm{state}}_t=
\left[
\mathbf{r}_{1,t},\mathbf{r}_{2,t},\mathbf{r}_{3,t},
\mathbf{v}_{1,t},\mathbf{v}_{2,t},\mathbf{v}_{3,t}
\right]\in\mathbb{R}^{18}.
\]

All continuous features are standardized using statistics estimated from the training partition only. Validation and test data are transformed using the same statistics.

## 3. Task A: system stability from initial conditions

Task A measures how much system-level stability information is available before any temporal evolution is observed. For system \(s\), the variation-level survival indicator at horizon \(H\) is

\[
y_{s,v}(H)=\mathbb{I}\!\left[t_{\mathrm{ub},s,v}\geq H\right],
\]

with \(H=10{,}000\) in the default setting. The regression target is the stable fraction across the 32 perturbations,

\[
q_s(H)=\frac{1}{V}\sum_{v=1}^{V}y_{s,v}(H), \qquad V=32.
\]

Because the 32 variations within a system share nearly identical macroscopic initial conditions, the current main pipeline uses the first variation as a representative system descriptor rather than repeating the full set. Its input is the static orbital15 vector. We compare a multilayer perceptron with tree-based gradient boosting. The MLP uses fully connected ReLU layers and dropout, while the gradient-boosted model captures nonlinear feature interactions without a learned temporal representation. Both models predict a value in \([0,1]\). Mean-squared error is the primary regression objective; a binary-cross-entropy objective with soft fraction labels is also supported.

This task isolates predictability encoded in the initial orbital configuration and provides a lower-information counterpart to prefix-based tasks.

## 4. Task B: early-warning prediction from trajectory prefixes

Task B uses only the first \(T_{\mathrm{obs}}\) samples of a simulated trajectory. The default observation length is \(T_{\mathrm{obs}}=100\), with additional experiments using longer prefixes. Two target modes are considered.

### 4.1 Variation-level instability prediction

For each trajectory, the model receives

\[
\mathbf{x}_{s,v,1:T_{\mathrm{obs}}}\in
\mathbb{R}^{T_{\mathrm{obs}}\times 15}
\]

and predicts the binary survival label \(y_{s,v}(H)\) at \(H=10{,}000\). We compare a linear support-vector baseline on flattened prefixes, a one-dimensional convolutional network, a dilated temporal convolutional network, and a Transformer encoder. The convolutional models extract local and multi-scale temporal signatures, whereas self-attention permits direct interaction between distant prefix states.

### 4.2 System-level stable-fraction prediction

In the group setting, all perturbation prefixes are retained:

\[
\mathbf{X}_{s}\in
\mathbb{R}^{T_{\mathrm{obs}}\times V\times C}.
\]

The target is \(q_s(H)\). A concatenation baseline flattens the perturbation dimension into the channel dimension. The set-attention model instead preserves the perturbation axis, applies attention across variations, and subsequently models temporal structure. This construction tests whether explicitly respecting the perturbation ensemble improves prediction over treating the group as an unstructured feature vector.

Binary models are trained with binary cross-entropy, and stable-fraction models are trained as bounded regression models. Early stopping is based on validation performance.

## 5. Task C: perturbation-aware multi-horizon survival prediction

Task C predicts an entire survival curve from an early observation of the 32-trajectory perturbation ensemble. For future horizons

\[
\mathcal{H}=\{500,800,1500,2500,4000,6500,10000\},
\]

the variation-level target is

\[
y_{s,v,k}=\mathbb{I}\!\left[t_{\mathrm{ub},s,v}\geq H_k\right].
\]

The corresponding system-level survival fraction is

\[
q_{s,k}=\frac{1}{V}\sum_{v=1}^{V}y_{s,v,k}.
\]

The default input is a prefix of length 200 with shape

\[
\mathbf{X}_{s}\in\mathbb{R}^{32\times 200\times 36}.
\]

We compare four model classes. The mean baseline predicts the average training-set survival curve for every test system. The direct model averages perturbations before temporal encoding. A richer direct baseline concatenates the perturbation-wise mean, standard deviation, minimum, and maximum at every time step. The trajectory-context model encodes each perturbation independently with a shared temporal encoder and augments each trajectory embedding with the mean group context. Finally, the dual-attention model applies temporal self-attention within each trajectory followed by self-attention across the perturbation set.

To guarantee physically valid predictions, all neural models parameterize discrete conditional hazards rather than directly regressing survival probabilities. If \(z_{s,v,k}\) is the predicted logit for interval \(k\), then

\[
h_{s,v,k}=\sigma(z_{s,v,k}),\qquad
\widehat{S}_{s,v,k}=\prod_{j=1}^{k}(1-h_{s,v,j}).
\]

This construction guarantees

\[
1\geq \widehat{S}_{s,v,1}\geq \cdots \geq
\widehat{S}_{s,v,K}\geq 0.
\]

Variation-aware models are trained using a discrete-time hazard negative log-likelihood evaluated only while a trajectory remains at risk. Aggregated direct models use an equivalent binomial hazard loss derived from the number of surviving and failing variations in each interval. System-level predictions are obtained by averaging variation-level survival probabilities.

## 6. Task D: finite-time phase-space divergence

Task D replaces a binary outcome with a continuous predictability timescale. The 32 perturbations are paired as

\[
(v,v+16),\qquad v=0,\ldots,15,
\]

so that each pair shares the same phase-grid configuration but differs by a controlled perturbation to the inner semimajor axis.

For every trajectory, positions and velocities are transformed to the mass-weighted center-of-mass frame:

\[
\widetilde{\mathbf{r}}_{b}(t)=
\mathbf{r}_{b}(t)-
\frac{\sum_jm_j(t)\mathbf{r}_j(t)}{\sum_jm_j(t)},
\]

with an analogous expression for velocity. For a perturbation pair \((i,j)\), we define normalized position and velocity separations

\[
\delta_r(t)=
\frac{\left\|\widetilde{\mathbf{r}}_i(t)-
\widetilde{\mathbf{r}}_j(t)\right\|_2}{L_0},
\qquad
\delta_v(t)=
\frac{\left\|\widetilde{\mathbf{v}}_i(t)-
\widetilde{\mathbf{v}}_j(t)\right\|_2}{V_0},
\]

where \(L_0\) and \(V_0\) are root-mean-square scales computed from the pair's initial centered states. The phase-space distance and its relative growth are

\[
d_{ij}(t)=\sqrt{\delta_r(t)^2+\delta_v(t)^2},
\qquad
R_{ij}(t)=\frac{d_{ij}(t)}{\max(d_{ij}(0),\epsilon)}.
\]

The pairwise divergence time is the first threshold-crossing time

\[
\tau_{ij}=\inf\{t:R_{ij}(t)>\rho\},
\]

with \(\rho=10^3\). Pairs that never cross the threshold within the recorded interval are treated as right-censored at their last finite time. The system-level divergence time is the median across the 16 valid pairs:

\[
\tau_s=\operatorname{median}_{(i,j)}\tau_{ij}.
\]

The main regression target is

\[
z_s=\log(1+\tau_s).
\]

As a complementary diagnostic, a finite-time growth rate \(\lambda_s\) is obtained by fitting a line to \(\log R_{ij}(t)\) over \(1<R_{ij}(t)<100\) and aggregating pairwise slopes by their median. This quantity is an empirical finite-perturbation growth rate and is not interpreted as an infinitesimal Lyapunov exponent.

We study two input settings. In the initial-condition setting, tabular features at \(t=0\) are used to predict \(z_s\), with a constant-median baseline, random forest, and MLP. In the prefix setting, only systems satisfying \(\tau_s>t_{\mathrm{obs}}\) are retained so that the observed prefix does not already contain the divergence event. The target becomes the remaining divergence time,

\[
z^{\mathrm{rem}}_s=
\log\!\left[1+\tau_s-t_{\mathrm{obs},s}\right].
\]

Prefix models include a CNN over ensemble-aggregated orbital channels, a CNN augmented with centered position and velocity channels, and a shared temporal CNN followed by permutation-invariant mean pooling over the 32 perturbations. Neural regressors use the Huber loss by default.

## 7. Task E: direct trajectory forecasting

Task E evaluates whether models can predict the future multivariate trajectory itself. A sliding window extracts a context

\[
\mathbf{x}_{n,1:L_{\mathrm{in}}}
\]

and a future target

\[
\mathbf{x}_{n,L_{\mathrm{in}}+1:L_{\mathrm{in}}+L_{\mathrm{out}}}.
\]

The primary orbital forecasting configuration uses orbital15 with

\[
L_{\mathrm{in}}=200,\qquad
L_{\mathrm{out}}=50,\qquad
\text{stride}=20.
\]

We evaluate a persistence baseline, ridge regression, a temporal convolutional network, Transformer, PatchTST, and a simplified iTransformer. We additionally consider pretrained forecasting models, including TimesFM, Chronos, and Panda. Persistence repeats the last observed state across the forecast horizon. Ridge regression maps the flattened context to the flattened future. The TCN uses causal dilated convolutions, PatchTST tokenizes each channel into temporal patches, the Transformer attends over time, and iTransformer treats variables as tokens. In the current interface, generic univariate foundation forecasters process each channel independently, while Panda consumes the full multivariate orbital window.

Models are trained in standardized space with mean-squared error. We report normalized and original-scale errors, together with horizon-wise and channel-wise metrics. Original-scale aggregate MSE is interpreted cautiously because channels with large physical variance can dominate the average.

## 8. Data splitting and leakage control

For Tasks A--D, splitting is performed at the base-system level. All 32 perturbations associated with the same system are assigned to the same partition. This prevents nearly identical perturbations from appearing in both training and evaluation data. Task C additionally stratifies systems using their final-horizon survival fraction to preserve the distribution of stable and unstable ensembles.

The current Task E implementation performs a trajectory-level random split before constructing windows, so windows from a given trajectory never cross partitions. However, different perturbations of the same base system can currently be assigned to different partitions. For the final paper benchmark, Task E should be changed to a group-wise split based on \(s=\lfloor r/32\rfloor\), thereby applying the same leakage-control standard across all five tasks.

Unless a task-specific experiment states otherwise, random seeds are fixed to 42, preprocessing statistics are fitted only on training data, and model selection is based exclusively on validation performance.

## 9. Optimization and evaluation

Neural networks are optimized with AdamW, gradient clipping, and early stopping. Classification performance is measured using AUROC, average precision, F1 score, accuracy, and confusion matrices. Stable-fraction and survival-curve predictions are evaluated using MAE and RMSE, with additional per-horizon survival errors and the fraction of monotone predictions. Divergence-time regression is assessed using MAE and RMSE in log-time, Pearson and Spearman correlation, median absolute log error, and the fraction of predictions within factors of two and three of the target. Forecasting is evaluated using normalized and raw-scale MSE and MAE, disaggregated by forecast horizon and feature channel.

Together, Tasks A--E form a controlled predictability hierarchy. Task E asks for the most detailed future description, Tasks B and C predict progressively coarser outcomes from early dynamics, Task D estimates a physically motivated divergence timescale, and Task A quantifies information already present in the initial conditions.

---

## Implementation notes to resolve before submission

1. Some legacy Task A outputs were generated with a flattened 32-variation initial-condition representation, whereas the current main script uses the first variation and therefore a 15-dimensional system descriptor. The paper should report one protocol consistently and rerun the corresponding baselines.
2. Some historical Task C notebook results used 8 perturbations and different future horizons. The current scripted `dataset_v3` protocol uses 32 perturbations and horizons \(\{500,800,1500,2500,4000,6500,10000\}\).
3. Historical Task D notes define divergence from semimajor-axis separation. The current implementation uses center-of-mass position--velocity phase-space distance. The latter should be treated as the definitive paper formulation.
4. Task E should use a base-system-level split before final cross-task comparisons are reported.
