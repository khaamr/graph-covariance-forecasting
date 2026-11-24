# Graph-based realized covariance forecasting

This repository contains an adapted public-data replication of *Graph-based methods for forecasting realized covariances* by [Zhang, Pu, Cucuringu and Dong](https://doi.org/10.1093/jjfinec/nbae026), together with adaptive covariance-shape and geometric spectral extensions. The reference experiment evaluates 13 specifications for 27 stocks at horizons of 1, 5 and 22 retained trading sessions.

The public experiment uses a different data feed and trading calendar from the paper's licensed LOBSTER sample. It reproduces the forecasting methodology with documented adaptations; it does not reproduce the original numerical benchmark or establish the paper's full model ranking and portfolio advantage. See the [protocol audit](covariance_replication/docs/protocol-audit.md) and [comparison with the published results](covariance_replication/docs/result-comparison.md).

[Project website](https://khaamr.github.io/graph-covariance-forecasting/) | [Interactive figures](https://khaamr.github.io/graph-covariance-forecasting/research_dashboard.html) | [Graph explanation](https://khaamr.github.io/graph-covariance-forecasting/#learn)

## Empirical results

### Public-data replication

Relative to HAR-DRD, GHAR(GL,L) reduced Euclidean loss by 1.65% at one session and 2.60% at five sessions. At 22 sessions, its Frobenius loss was 0.45% higher. The full model ranking and portfolio advantage did not replicate under the public-data specification. The [result comparison](covariance_replication/docs/result-comparison.md) reports the loss ratios and model-confidence-set results.

### Spectral extension

Across 443 shared forecast dates in the reused 2025 to October 2026 benchmark, the spectral gate reduced Frobenius loss by 1.79% relative to HAR-DRD. Annualized long-only risk decreased from 9.435% to 9.406% before transaction costs. The paired risk interval included zero, so the comparison does not establish a joint improvement in forecast accuracy and portfolio risk. See the [spectral results and paired intervals](covariance_replication/docs/spectral/results.md).

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
