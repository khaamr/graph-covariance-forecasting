# Graph-based realized covariance forecasting

This repository contains an adapted public-data replication of *Graph-based methods for forecasting realized covariances* by [Zhang, Pu, Cucuringu and Dong](https://doi.org/10.1093/jjfinec/nbae026), together with fixed covariance ensembles, adaptive covariance-shape and geometric spectral extensions. The reference experiment evaluates 13 specifications for 27 stocks at horizons of 1, 5 and 22 retained trading sessions.

The public experiment uses a different data feed and trading calendar from the paper's licensed LOBSTER sample. It reproduces the forecasting methodology with documented adaptations; it does not reproduce the original numerical benchmark or establish the paper's full model ranking and portfolio advantage. See the [protocol audit](covariance_replication/docs/protocol-audit.md) and [comparison with the published results](covariance_replication/docs/result-comparison.md).

[Project website](https://khaamr.github.io/graph-covariance-forecasting/) | [Interactive figures](https://khaamr.github.io/graph-covariance-forecasting/research_dashboard.html) | [Graph explanation](https://khaamr.github.io/graph-covariance-forecasting/#learn)

## Empirical results

### Public-data replication

Relative to HAR-DRD, GHAR(GL,L) reduced Euclidean loss by 1.65% at one session and 2.60% at five sessions. At 22 sessions, its Frobenius loss was 0.45% higher. The full model ranking and portfolio advantage did not replicate under the public-data specification. The [result comparison](covariance_replication/docs/result-comparison.md) reports the loss ratios and model-confidence-set results.

### Balanced covariance with a stress response

The selected [balanced covariance model](covariance_replication/docs/balanced-model/results.md) improves all four observed point comparisons against HAR-DRD in both evaluation periods. On the same 397 reused 2025–2026 dates, Euclidean, Frobenius and QLIKE losses fall **2.68%, 2.94% and 1.19%**, and annualized long-only risk falls **9.722% to 9.511%**. The risk interval still includes zero. The candidate was selected retrospectively from the recorded search; it is an experimental extension requiring independent evaluation.

[Comparison with the first version](covariance_replication/docs/balanced-model/results.md#difference-from-the-first-version), [direct forecast API](covariance_replication/balanced_covariance.py), [standalone reproduction](covariance_replication/run_balanced_model.py), and [positive-definiteness and portfolio proofs](covariance_replication/docs/balanced-model/mathematics.md) accompany all search outcomes and uncertainty intervals.

### Spectral extension

Across 443 shared forecast dates in the reused 2025 to October 2026 benchmark, the spectral gate reduced Frobenius loss by 1.79% relative to HAR-DRD. Annualized long-only risk decreased from 9.435% to 9.406% before transaction costs. The paired risk interval included zero, so the comparison does not establish a joint improvement in forecast accuracy and portfolio risk. See the [spectral results and paired intervals](covariance_replication/docs/spectral/results.md).

### Fixed covariance ensemble

The equal-weight pool of five causal covariance forecasts lowered mean Euclidean, Frobenius and QLIKE losses by **1.22%, 1.26% and 0.35%**, respectively, against HAR-DRD on 397 common dates in the reused 2025–2026 evaluation. Annualized long-only risk fell from **9.72% to 9.67%**. QLIKE and risk confidence intervals include zero improvement, so the result does not establish joint statistical superiority. The separately trained selector increased risk and is retained in the comparison.

This is an experimental ensemble extension with a stricter calendar screen, rather than a replacement for the original replication. Its dates differ from the 443-date spectral benchmark above. See the [fixed protocol, controls and paired intervals](covariance_replication/docs/joint-pool/results.md), [direct ensemble implementation](covariance_replication/joint_pool.py), and [GLASSO penalty diagnostic](covariance_replication/docs/glasso-diagnostic.md).

### Literature-guided challengers

Six additional ridge and factor-correlation challengers were evaluated on the same dates. Ridge log-HAR with graph correlations achieved the lowest later-period QLIKE in the 14-model comparison, **1.59% below HAR-DRD**, but increased portfolio risk slightly. None of the six improved all four endpoints. See the [research sources, full comparison and reproduction commands](covariance_replication/docs/literature-challengers/results.md).

### Published-method implementations

DCC-NL with analytical nonlinear shrinkage and factor-VHAR with LASSO/adaptive LASSO are implemented and evaluated on the same dates. The authorized equal-weight market-factor specification and DCC-NL both worsen later-period losses and portfolio risk. The [implementation audit and complete results](covariance_replication/docs/published-methods/results.md) document equations, author-code verification, remaining data differences, and the interface for historical characteristic-factor weights.

### Diagnostic corrections

The [diagnosis study](covariance_replication/docs/model-diagnosis/results.md) identifies variance underprediction, stress-day filtering and substantial omitted cross-sector residuals in the market-only factor setup. Realized-covariance HEAVY narrows the DCC Frobenius gap from 23.22% to 0.20% above HAR-DRD. Raw-input/matrix-smearing factor corrections lower its QLIKE gap from 9.54% to 2.89% above HAR-DRD and reduce annual risk from 10.336% to 9.862%. These improvements do not establish an overall advantage over HAR-DRD.

## Notebooks

| Notebook | Contents |
| --- | --- |
| [Covariance replication](covariance_replication/Covariance_Replication.ipynb) | HAR and graph-based forecasts at 1, 5 and 22 retained sessions |
| [Adaptive covariance shapes](covariance_replication/Extended_Covariance.ipynb) | Covariance-shape mixtures with delayed feedback |
| [Spectral extension](covariance_replication/Spectral_Covariance.ipynb) | Geometric pooling, eigenvalue cleaning and portfolio evaluation |

## Figures and mathematical documentation

| Resource | Contents |
| --- | --- |
| [Replication dashboard](https://khaamr.github.io/graph-covariance-forecasting/research_dashboard.html#section-replication-01) | Forecast losses by horizon, portfolio risk and turnover |
| [Shape-extension dashboard](https://khaamr.github.io/graph-covariance-forecasting/research_dashboard.html#section-shape-extension-01) | Graph contraction, adaptive expert weights and covariance loss |
| [Spectral-extension dashboard](https://khaamr.github.io/graph-covariance-forecasting/research_dashboard.html#section-spectral-extension-01) | Expert weights, model loss and long-only portfolio risk |
| [Figure sources](figures/README.md) | Source inventory for all nine interactive figures |
| [Mathematical notes](covariance_replication/docs/theory.md) | Assumptions, propositions and proofs |

## Reproduction instructions

Use Python 3.10 or 3.11.

```sh
git clone https://github.com/khaamr/graph-covariance-forecasting.git
cd graph-covariance-forecasting/covariance_replication
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Open a notebook using that environment. The first run caches about 0.5 GB of minute bars. Run `python export_figures.py` from the same folder to rebuild the interactive dashboard from saved notebook outputs.
