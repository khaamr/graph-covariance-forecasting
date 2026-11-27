"""
Walk-forward HAR and GHAR models for full realized covariance matrices.
"""

from dataclasses import dataclass
from itertools import product
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
from statsmodels.stats.correlation_tools import cov_nearest
from sklearn.exceptions import ConvergenceWarning

from market_data import SECTORS


VAR_GRAPHS = ("-", "S", "K", "GL")
CORR_GRAPHS = ("-", "K", "L")
BASELINE = "HAR-DRD"
GRAPH_MODEL = "GHAR(GL,L)"


@dataclass(frozen=True)
class Forecasts:
    dates: pd.DatetimeIndex
    positions: np.ndarray
    actual: np.ndarray
    predictions: dict[str, np.ndarray]
    audit: pd.DataFrame


def har_lags(values):
    """
    Use lag 1, the mean of lags 2–5, and the mean of lags 6–22.
    """
    frame = pd.DataFrame(values)
    return np.stack([
        frame.shift(1).to_numpy(),
        frame.shift(2).rolling(4).mean().to_numpy(),
        frame.shift(6).rolling(17).mean().to_numpy(),
    ], axis=-1)


def normalized_graph(adjacency):
    degree = adjacency.sum(axis=1)
    scale = np.divide(1, np.sqrt(degree), out=np.zeros_like(degree, dtype=float), where=degree > 0)
    return scale[:, None] * adjacency * scale[None, :]


def precision_graph(returns, *, alphas=4):
    """
    Fit the sparse graph using only returns before the first forecast origin.
    """
    scale = np.sqrt(np.var(returns, axis=0).mean())
    samples = returns / scale
    with catch_warnings(record=True) as cv_warnings:
        simplefilter("always")
        search = GraphicalLassoCV(alphas=alphas, cv=5, n_jobs=1, max_iter=180).fit(samples)
    sample_cov = np.cov(samples, rowvar=False, ddof=0)
    np.fill_diagonal(sample_cov, 0)
    diagonal_penalty = np.abs(sample_cov).max() * (1 + 1e-6)
    penalties = sorted({min(search.alpha_ * m, diagonal_penalty) for m in (1, 1.25, 1.5, 2, 4, 8)}
                       | {diagonal_penalty})
    for alpha in penalties:
        with catch_warnings(record=True) as issues:
            simplefilter("always", ConvergenceWarning)
            fit = GraphicalLasso(alpha=alpha, max_iter=800, enet_tol=1e-8).fit(samples)
        if not any(issubclass(issue.category, ConvergenceWarning) for issue in issues):
            break
    else:
        raise RuntimeError("Graphical LASSO did not converge.")
    edges = fit.precision_ != 0
    np.fill_diagonal(edges, False)
    return edges, alpha * scale ** 2, len(cv_warnings), alpha > search.alpha_ * (1 + 1e-4)


def correlation_graph_features(features, pairs, n_assets, graph_effect="all"):
    """
    Connect all distinct correlation pairs that share an endpoint.
    """
    left, right = pairs
    node_sums = np.zeros((len(features), n_assets, features.shape[-1]))
    for k, (i, j) in enumerate(zip(left, right)):
        node_sums[:, i] += features[:, k]
        node_sums[:, j] += features[:, k]
    line = (node_sums[:, left] + node_sums[:, right] - 2 * features) / (2 * (n_assets - 2))
    complete = (features.sum(axis=1, keepdims=True) - features) / (len(left) - 1)
    if graph_effect == "daily":
        complete, line = complete[..., :1], line[..., :1]
    return {
        "-": features,
        "K": np.concatenate([features, complete], axis=-1),
        "L": np.concatenate([features, line], axis=-1),
    }


def panel_ols(features, target, train, test, intercept_mode="series"):
    """
    Fit per-series intercepts and shared slopes with within-series demeaning.
    """
    fit = features[train]
    axes = 0 if intercept_mode == "series" else (0, 1)
    x_mean = fit.mean(axis=axes)
    y_mean = target[train].mean(axis=axes)
    x = (fit - x_mean).reshape(-1, fit.shape[-1])
    y = (target[train] - y_mean).ravel()
    scale = np.sqrt(np.mean(x ** 2, axis=0))
    scale[scale == 0] = 1
    x /= scale
    coef = lstsq(x, y, cond=1e-12, lapack_driver="gelsy")[0] / scale
    return (features[test] - x_mean) @ coef + y_mean


def covariance_components(covariance):
    n_assets = covariance.shape[-1]
    pairs = np.tril_indices(n_assets, k=-1)
    lower = np.tril_indices(n_assets)
    variance = np.diagonal(covariance, axis1=-2, axis2=-1).copy()
    std = np.sqrt(variance)
    correlations = covariance[:, pairs[0], pairs[1]] / (std[:, pairs[0]] * std[:, pairs[1]])
    stabilize = np.maximum(1e-9 - np.linalg.eigvalsh(covariance)[:, 0], 0)
    chol = np.linalg.cholesky(covariance + stabilize[:, None, None] * np.eye(n_assets))
    return variance, correlations, chol[:, lower[0], lower[1]]


def make_pd_correlation(corr, eigen_floor=1e-4):
    """
    Repair small eigenvalues and renormalize the diagonal, as in correlation clipping.
    """
    minimum = np.linalg.eigvalsh(corr)[:, 0]
    needs_repair = minimum < eigen_floor
    for index in np.flatnonzero(needs_repair):
        corr[index] = cov_nearest(corr[index], method="clipped", threshold=eigen_floor)
    return corr, int(needs_repair.sum())


def rebuild_drd(variance, pair_forecast):
    n = variance.shape[-1]
    pairs = np.tril_indices(n, k=-1)
    corr = np.repeat(np.eye(n)[None], len(variance), axis=0)
    corr[:, pairs[0], pairs[1]] = corr[:, pairs[1], pairs[0]] = pair_forecast.clip(-1, 1)
    corr, repaired = make_pd_correlation(corr)
    std = np.sqrt(variance)
    return corr * std[:, :, None] * std[:, None, :], repaired


def forward_matrices(covariance, horizon):
    if not 1 <= horizon <= len(covariance):
        raise ValueError("The forecast horizon must fit inside the observed panel.")
    padded = np.concatenate([np.zeros_like(covariance[:1]), np.cumsum(covariance, axis=0)])
    return padded[horizon:] - padded[:-horizon]


def graph_augmented_features(features, weight, extra=None, graph_effect="all"):
    blocks = [features]
    if weight is not None:
        neighbors = weight @ features
        blocks.append(neighbors[..., :1] if graph_effect == "daily" else neighbors)
    if extra is not None:
        blocks.append(extra)
    return np.concatenate(blocks, axis=-1) if len(blocks) > 1 else features


def transform_drd(variance, correlations, transformation):
    """Appendix E interpretation: transform series before time/graph aggregation."""
    if transformation == "levels":
        return variance, correlations
    if transformation != "log_fisher":
        raise ValueError("Choose levels or log_fisher transformations.")
    if not np.isfinite(variance).all() or (variance <= 0).any():
        raise ValueError("Log variance requires finite positive observations.")
    if not np.isfinite(correlations).all() or (np.abs(correlations) > 1 + 1e-12).any():
        raise ValueError("Fisher transformation requires finite correlations in [-1, 1].")
    return np.log(variance), np.arctanh(correlations.clip(-1 + 1e-12, 1 - 1e-12))


def inverse_variance(values, transformation):
    """Direct inverse, with no undocumented log-retransformation bias correction."""
    if transformation == "levels":
        return values
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        result = np.exp(values)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("Log-variance forecasts cannot be represented as positive finite values.")
    return result


def forecast_grid(
    panel, horizons=(1, 5, 22), lookback=1000, first_forecast=None,
    quarticity=False, quarticity_power=1.0, intercept_mode="series",
    reference_calendar=None, transformation="levels", graph_effect="all", refit_every=None,
    glasso_alphas=4,
):
    """
    Fit monthly folds, purging incomplete labels. With a reference calendar,
    exclude features and targets that span missing reference sessions. The
    lookback remains a retained-row window; eligible label counts may decrease.
    Optional Appendix C-E variants retain the original defaults. log_fisher uses
    direct exp/tanh inverses; the manuscript does not specify a bias correction.
    refit_every counts retained sessions, anchored to the common first test date.
    """
    if not isinstance(glasso_alphas, int) or isinstance(glasso_alphas, bool) or glasso_alphas < 2:
        raise ValueError("glasso_alphas must be an integer of at least two.")
    if transformation not in ("levels", "log_fisher"):
        raise ValueError("Choose levels or log_fisher transformations.")
    if graph_effect not in ("all", "daily"):
        raise ValueError("Choose all graph lags or the daily graph effect.")
    if refit_every is not None and (isinstance(refit_every, bool) or not isinstance(refit_every, int) or refit_every < 1):
        raise ValueError("refit_every must be a positive integer, or None for calendar months.")
    if quarticity and transformation != "levels":
        raise ValueError("The manuscript does not define transformed HARQ interactions.")
    if intercept_mode not in ("series", "shared"):
        raise ValueError("Choose series or shared intercepts.")
    if quarticity_power not in (0.5, 1.0):
        raise ValueError("Use quarticity_power=1.0 for the manuscript or 0.5 for the protocol.")
    n_dates, n_assets = panel.graph_return.shape
    lag_start = 22
    feature_calendar_ok = np.ones(n_dates, dtype=bool)
    target_calendar_ok = {h: np.ones(n_dates, dtype=bool) for h in horizons}
    if reference_calendar is not None:
        reference = pd.DatetimeIndex(reference_calendar)
        if (not reference.is_unique or not reference.is_monotonic_increasing
                or not panel.dates.is_unique or not panel.dates.is_monotonic_increasing):
            raise ValueError("Panel and reference dates must be unique and increasing.")
        calendar_positions = reference.get_indexer(panel.dates)
        if (calendar_positions < 0).any():
            raise ValueError("Every panel date must occur in the reference calendar.")
        feature_calendar_ok[:] = False
        feature_calendar_ok[lag_start:] = calendar_positions[lag_start:] - calendar_positions[:-lag_start] == lag_start
        for h in horizons:
            if not 1 <= h <= n_dates:
                raise ValueError("The forecast horizon must fit inside the observed panel.")
            target_calendar_ok[h][:] = False
            target_calendar_ok[h][:n_dates-h+1] = calendar_positions[h-1:] - calendar_positions[:n_dates-h+1] == h-1
    test = np.arange(lookback + lag_start, n_dates)
    if first_forecast is not None:
        test = test[panel.dates[test] >= first_forecast]
    if not len(test):
        raise ValueError(f"Need more than {lookback + lag_start} aligned sessions.")
    month_index = panel.dates.to_period("M")
    if month_index[test[0]] == month_index[test[0] - 1]:
        test = test[month_index[test] != month_index[test[0]]]
    if not len(test):
        raise ValueError("No full forecast month after calibration.")
    months = month_index[test]
    pairs = np.tril_indices(n_assets, k=-1)
    lower = np.tril_indices(n_assets)
    labels = [SECTORS[s] for s in panel.symbols]
    sectors = np.equal.outer(labels, labels) & ~np.eye(n_assets, dtype=bool)
    base_graphs = {"-": None, "S": normalized_graph(sectors), "K": normalized_graph(1 - np.eye(n_assets))}
    daily_var, daily_corr, daily_chol = covariance_components(panel.covariance)
    daily_var, daily_corr = transform_drd(daily_var, daily_corr, transformation)
    var_x, chol_x = har_lags(daily_var), har_lags(daily_chol)
    corr_x = correlation_graph_features(har_lags(daily_corr), pairs, n_assets, graph_effect)
    quart_x = np.full_like(var_x[:, :, :1], np.nan)
    if quarticity:
        quart_x[1:, :, 0] = daily_var[:-1] * panel.quarticity[:-1] ** quarticity_power
    realized = {h: forward_matrices(panel.covariance, h) for h in horizons}
    targets = {}
    for h in horizons:
        v, r, c = covariance_components(realized[h])
        v, r = transform_drd(v, r, transformation)
        targets[h] = v, r, c
    forecasts = {h: {} for h in horizons}
    rows = {h: [] for h in horizons}
    audit = {h: [] for h in horizons}

    folds = months if refit_every is None else np.arange(len(test)) // refit_every
    for fold in pd.unique(folds):
        month_test = test[folds == fold]
        origin = month_test[0]
        month = month_index[origin]
        start = origin - lookback
        graph_options = {} if glasso_alphas == 4 else {"alphas": glasso_alphas}
        graph, alpha, warnings, fallback = precision_graph(panel.graph_return[start:origin], **graph_options)
        graphs = base_graphs | {"GL": normalized_graph(graph)}
        day_slice = slice(start, month_test[-1] + 1)
        monthly_var = {
            key: graph_augmented_features(var_x[day_slice], weight, graph_effect=graph_effect)
            for key, weight in graphs.items()
        }
        for h in horizons:
            future = month_test[month_test + h <= n_dates]
            future = future[feature_calendar_ok[future] & target_calendar_ok[h][future]]
            if not len(future):
                continue
            train = np.arange(start, origin - h + 1)
            assert train[-1] + h <= origin
            assert len(train) == lookback - h + 1
            candidate_labels = len(train)
            train = train[feature_calendar_ok[train] & target_calendar_ok[h][train]]
            if not len(train):
                raise ValueError(f"No calendar-complete training labels in {month} for horizon {h}.")
            fit_local, future_local = train - start, future - start
            v, r, c = targets[h]
            all_var = {}
            for key, inputs in monthly_var.items():
                var = panel_ols(inputs, v[day_slice], fit_local, future_local, intercept_mode)
                var = inverse_variance(var, transformation)
                floor = inverse_variance(v[train].min(axis=0), transformation)
                all_var[key] = np.where(var > 0, var, floor)
                if quarticity and h == 1:
                    q = np.concatenate([inputs, quart_x[day_slice]], axis=-1)
                    var_q = panel_ols(q, v[day_slice], fit_local, future_local, intercept_mode)
                    all_var[key + "Q"] = np.where(var_q > 0, var_q, v[train].min(axis=0))
            all_corr = {
                key: panel_ols(inputs[day_slice], r[day_slice], fit_local, future_local, intercept_mode)
                for key, inputs in corr_x.items()
            }
            if transformation == "log_fisher":
                all_corr = {key: np.tanh(value) for key, value in all_corr.items()}
            chol = panel_ols(chol_x[day_slice], c[day_slice], fit_local, future_local, intercept_mode)
            triangles = np.zeros((len(future), n_assets, n_assets))
            triangles[:, lower[0], lower[1]] = chol
            result = {"HAR-Cholesky": triangles @ triangles.swapaxes(-1, -2)}
            repairs = 0
            for vol_key, cor_key in product(all_var, CORR_GRAPHS):
                suffix = "Q" if vol_key.endswith("Q") else ""
                vol_graph = vol_key.rstrip("Q")
                model = f"HAR{suffix}-DRD" if (vol_graph, cor_key) == ("-", "-") else f"GHAR{suffix}{'-S' if graph_effect == 'daily' else ''}({vol_graph},{cor_key})"
                result[model], count = rebuild_drd(all_var[vol_key], all_corr[cor_key])
                repairs += count
            for model, matrices in result.items():
                forecasts[h].setdefault(model, []).append(matrices)
            rows[h].append(future)
            audit[h].append({
                "month": str(month), "first_forecast": panel.dates[origin],
                "first_train_feature": panel.dates[start],
                "last_train_feature": panel.dates[origin - 1],
                "last_train_target": panel.dates[train[-1] + h - 1],
                "graph_return_through": panel.dates[origin - 1],
                "training_labels": len(train), "test_origins": len(future),
                "calendar_excluded_labels": candidate_labels - len(train),
                "reference_calendar_supplied": reference_calendar is not None,
                "first_used_train_feature": panel.dates[train[0]],
                "quarticity_power": quarticity_power if quarticity and h == 1 else np.nan,
                "intercept_mode": intercept_mode,
                "transformation": transformation, "graph_effect": graph_effect,
                "refit_every": refit_every if refit_every is not None else "calendar month",
                "glasso_edges": int(graph.sum() // 2), "glasso_alpha": alpha,
                "glasso_cv_alphas": glasso_alphas,
                "glasso_cv_warnings": warnings, "penalty_increased": fallback,
                "correlation_repairs": repairs,
            })

    result = {}
    for h in horizons:
        if not rows[h]:
            raise ValueError(f"No complete held-out targets for horizon {h}.")
        pos = np.concatenate(rows[h])
        estimates = {k: np.concatenate(v) for k, v in forecasts[h].items()}
        result[h] = Forecasts(panel.dates[pos], pos, realized[h][pos], estimates, pd.DataFrame(audit[h]))
    return result
