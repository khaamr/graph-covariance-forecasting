"""DCC-NL: Gaussian GARCH(1,1), nonlinear target, and 2MSCLE dynamics.

Equations 2.1-2.4 of Engle, Ledoit and Wolf (2019). The nonlinear target uses
the analytical Ledoit-Wolf estimator endorsed as a fast QuEST replacement in
the authors' package. Source and empirical adaptations are documented with
the benchmark; this module never uses realized future covariances for fitting.
"""
from dataclasses import dataclass
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd
from arch import arch_model
from numba import njit
from scipy.optimize import minimize


MODEL = "DCC-NL analytical target"


def nonlinear_covariance(samples, *, demean=False, lost_df=0):
    """Analytical nonlinear shrinkage, including the p > n case.

    Implements the published kernel formula; bandwidth n**(-1/3). GARCH
    residuals use demean=False, lost_df=0, matching the author's target call.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or not np.isfinite(samples).all():
        raise ValueError("Expected a finite observations-by-assets matrix.")
    if not isinstance(lost_df, int) or isinstance(lost_df, bool) or lost_df < 0:
        raise ValueError("lost_df must be a nonnegative integer.")
    if demean:
        samples = samples - samples.mean(axis=0)
        lost_df += 1
    n, p = len(samples) - lost_df, samples.shape[1]
    if n < 12 or p < 1:
        raise ValueError("Analytical shrinkage requires effective n >= 12 and assets >= 1.")
    values, vectors = np.linalg.eigh(samples.T @ samples / n)
    positive = values[max(0, p - n):]
    if positive.min() <= 0:
        raise ValueError("Samples have an unsupported degenerate covariance spectrum.")
    bandwidth = n ** (-1 / 3)
    width = bandwidth * positive[None, :]
    difference = (positive[:, None] - positive[None, :]) / width
    root = np.sqrt(5.)
    density = (3 / (4 * root) * np.maximum(1 - difference**2 / 5, 0) / width).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        hilbert_kernel = -3 * difference / (10 * np.pi) + 3 / (4 * root * np.pi) * (
            1 - difference**2 / 5) * np.log(np.abs((root - difference) / (root + difference)))
    endpoint = np.isclose(np.abs(difference), root, rtol=0, atol=1e-14)
    hilbert_kernel[endpoint] = -3 * difference[endpoint] / (10 * np.pi)
    hilbert = (hilbert_kernel / width).mean(axis=1)
    if p <= n:
        ratio = p / n
        shrunk = positive / ((np.pi * ratio * positive * density)**2 + (
            1 - ratio - np.pi * ratio * positive * hilbert)**2)
    else:
        hilbert_zero = (3 / (10 * bandwidth**2) + 3 / (4 * root * bandwidth) * (
            1 - 1 / (5 * bandwidth**2)) * np.log((1 + root * bandwidth) / (
                1 - root * bandwidth))) * np.mean(1 / positive) / np.pi
        null_eigenvalue = 1 / (np.pi * (p - n) / n * hilbert_zero)
        other = 1 / (np.pi**2 * positive * (density**2 + hilbert**2))
        shrunk = np.r_[np.repeat(null_eigenvalue, p - n), other]
    if not np.isfinite(shrunk).all() or (shrunk <= 0).any():
        raise RuntimeError("Nonlinear shrinkage produced an invalid spectrum.")
    result = (vectors * shrunk) @ vectors.T
    return (result + result.T) / 2


def correlation(covariance):
    std = np.sqrt(np.diag(covariance))
    result = covariance / std[:, None] / std[None, :]
    return (result + result.T) / 2


def dcc_backcast(standardized):
    length = min(len(standardized), int(np.sqrt(len(standardized))) + 1)
    weights = .94 ** np.arange(length)
    return np.einsum("t,ti,tj->ij", weights / weights.sum(), standardized[:length], standardized[:length])


@njit(cache=True)
def composite_objective(parameters, standardized, target, backcast):
    """Mean contiguous-pair negative log likelihood and analytic gradient."""
    alpha, beta = parameters
    t_count, n = standardized.shape
    diagonal = np.empty(n)
    off = np.empty(n - 1)
    da_diag = np.empty(n)
    db_diag = np.empty(n)
    da_off = np.empty(n - 1)
    db_off = np.empty(n - 1)
    for i in range(n):
        diagonal[i] = (1 - alpha - beta) * target[i, i] + (alpha + beta) * backcast[i, i]
        da_diag[i] = db_diag[i] = backcast[i, i] - target[i, i]
    for i in range(n - 1):
        off[i] = (1 - alpha - beta) * target[i, i + 1] + (alpha + beta) * backcast[i, i + 1]
        da_off[i] = db_off[i] = backcast[i, i + 1] - target[i, i + 1]
    loss = 0.
    gradient = np.zeros(2)
    for t in range(t_count):
        if t:
            for i in range(n):
                old = diagonal[i]
                innovation = standardized[t - 1, i]**2
                diagonal[i] = (1 - alpha - beta) * target[i, i] + alpha * innovation + beta * old
                da_diag[i] = innovation - target[i, i] + beta * da_diag[i]
                db_diag[i] = old - target[i, i] + beta * db_diag[i]
            for i in range(n - 1):
                old = off[i]
                innovation = standardized[t - 1, i] * standardized[t - 1, i + 1]
                off[i] = (1 - alpha - beta) * target[i, i + 1] + alpha * innovation + beta * old
                da_off[i] = innovation - target[i, i + 1] + beta * da_off[i]
                db_off[i] = old - target[i, i + 1] + beta * db_off[i]
        for i in range(n - 1):
            scale = np.sqrt(diagonal[i] * diagonal[i + 1])
            rho = off[i] / scale
            denominator = 1 - rho**2
            if denominator <= 0 or not np.isfinite(denominator):
                return np.inf, np.zeros(2)
            x, y = standardized[t, i], standardized[t, i + 1]
            numerator = x*x + y*y - 2*rho*x*y
            loss += .5 * (2 * np.log(2*np.pi) + np.log(denominator) + numerator / denominator)
            derivative = -rho / denominator - x*y / denominator + rho * numerator / denominator**2
            grad_a = da_off[i] / scale - .5 * rho * (da_diag[i] / diagonal[i] + da_diag[i + 1] / diagonal[i + 1])
            grad_b = db_off[i] / scale - .5 * rho * (db_diag[i] / diagonal[i] + db_diag[i + 1] / diagonal[i + 1])
            gradient[0] += derivative * grad_a
            gradient[1] += derivative * grad_b
    return loss / (t_count * (n - 1)), gradient / (t_count * (n - 1))


def filter_q(standardized, target, parameters, backcast):
    alpha, beta = parameters
    q = (1 - alpha - beta) * target + (alpha + beta) * backcast
    for z in standardized[:-1]:
        q = (1 - alpha - beta) * target + alpha * np.outer(z, z) + beta * q
    return q


@dataclass(frozen=True)
class DCCFit:
    mean: np.ndarray
    garch: np.ndarray
    last_variance: np.ndarray
    last_residual: np.ndarray
    target: np.ndarray
    parameters: np.ndarray
    last_q: np.ndarray
    diagnostics: dict


def fit_dcc_nl(returns):
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 2 or returns.shape[1] < 2 or len(returns) < 50 or not np.isfinite(returns).all():
        raise ValueError("DCC requires >=50 finite returns on >=2 assets.")
    mean = returns.mean(axis=0)
    residual = returns - mean
    variances = np.empty_like(residual)
    garch = []
    retries = 0
    roundoff_repairs = 0
    for i in range(returns.shape[1]):
        # Unit-variance fitting improves SLSQP conditioning; undo scale exactly.
        scale = residual[:, i].std()
        if scale <= 0:
            raise ValueError("GARCH cannot fit a constant return series.")
        model = arch_model(residual[:, i] / scale, mean="Zero", vol="GARCH", p=1, o=0, q=1,
                           dist="normal", rescale=False)
        with catch_warnings():
            simplefilter("ignore")
            fitted = model.fit(disp="off", show_warning=False, tol=1e-8,
                               options={"maxiter": 1500})
        if fitted.convergence_flag != 0:
            candidates = []
            for a, b in [(.05, .9), (.1, .8), (.03, .95)]:
                retries += 1
                with catch_warnings():
                    simplefilter("ignore")
                    retry = model.fit(disp="off", show_warning=False, tol=1e-8,
                                      starting_values=np.array([1-a-b, a, b]),
                                      options={"maxiter": 2000})
                if retry.convergence_flag == 0:
                    candidates.append(retry)
            if not candidates:
                raise RuntimeError(f"GARCH asset {i} failed: {fitted.optimization_result.message}")
            fitted = max(candidates, key=lambda candidate: candidate.loglikelihood)
        params = fitted.params.to_numpy().copy()
        if params[0] <= 0 or (params[1:] < 0).any() or params[1:].sum() > 1 + 1e-6:
            raise RuntimeError("Invalid GARCH parameters.")
        if params[1:].sum() > 1:
            # SLSQP can report success a few 1e-8 outside its linear constraint.
            # Restore feasibility and re-filter under the restored parameters.
            params[1:] /= params[1:].sum() * (1 + 1e-12)
            unit_residual = residual[:, i] / scale
            unit_variance = model.volatility.compute_variance(
                params, unit_residual, np.empty(len(returns)),
                model.volatility.backcast(unit_residual), model.volatility.variance_bounds(unit_residual))
            variances[:, i] = unit_variance * scale**2
            roundoff_repairs += 1
        else:
            variances[:, i] = fitted.conditional_volatility**2 * scale**2
        params[0] *= scale**2
        garch.append(params)
    standardized = np.ascontiguousarray(residual / np.sqrt(variances))
    target = correlation(nonlinear_covariance(standardized))
    backcast = dcc_backcast(standardized)
    objective = lambda value: composite_objective(value, standardized, target, backcast)
    starts = [np.array([a, persistence - a]) for a in (.01, .03, .05, .1)
              for persistence in (.99, .97, .95)]
    starts.sort(key=lambda value: objective(value)[0])
    fitted = minimize(objective, starts[0], jac=True, method="SLSQP", bounds=[(0, .99998)]*2,
                      constraints=[{"type": "ineq", "fun": lambda p: .99998 - p.sum(),
                                    "jac": lambda p: -np.ones(2)}],
                      options={"maxiter": 500, "ftol": 1e-10})
    if not fitted.success or not np.isfinite(fitted.fun) or fitted.x.sum() > .99998001:
        raise RuntimeError(f"DCC likelihood optimization failed: {fitted.message}")
    q = filter_q(standardized, target, fitted.x, backcast)
    return DCCFit(mean, np.array(garch), variances[-1], residual[-1], target, fitted.x, q,
                  {"alpha": fitted.x[0], "beta": fitted.x[1], "composite_nll": fitted.fun,
                   "optimizer_iterations": fitted.nit, "garch_failures": 0,
                   "garch_retry_attempts": retries,
                   "garch_constraint_roundoff_repairs": roundoff_repairs,
                   "target_condition": np.linalg.cond(target)})


def advance_dcc(fit, count_returns):
    """Forecast each incoming date before revealing that date's observed return."""
    alpha, beta = fit.parameters
    omega, arch_coef, garch_coef = fit.garch.T
    variance, residual, q = fit.last_variance.copy(), fit.last_residual.copy(), fit.last_q.copy()
    output = []
    for observed in count_returns:
        z = residual / np.sqrt(variance)
        q = (1 - alpha - beta) * fit.target + alpha * np.outer(z, z) + beta * q
        variance = omega + arch_coef * residual**2 + garch_coef * variance
        std = np.sqrt(variance)
        output.append(correlation(q) * std[:, None] * std[None, :])
        residual = observed - fit.mean
    return np.asarray(output)


def dcc_forecasts(dates, returns, positions, lookback=1250, progress=None):
    dates = pd.DatetimeIndex(dates)
    positions = np.asarray(positions)
    if not dates.is_unique or not dates.is_monotonic_increasing or np.any(np.diff(positions) <= 0):
        raise ValueError("Dates and forecast positions must be ordered and unique.")
    if len(returns) != len(dates) or len(positions) == 0 or positions.min() < lookback or positions.max() >= len(dates):
        raise ValueError("Returns and requested origins must align and have a full lookback.")
    months = dates[positions].to_period("M")
    output = np.empty((len(positions), returns.shape[1], returns.shape[1]))
    audits, parameters = [], []
    for month in months.unique():
        rows = np.flatnonzero(months == month)
        origin, end = positions[rows[0]], positions[rows[-1]]
        fit = fit_dcc_nl(returns[origin - lookback:origin])
        filtered = advance_dcc(fit, returns[origin:end + 1])
        output[rows] = filtered[positions[rows] - origin]
        audits.append({"month": str(month), "first_forecast": dates[origin],
                       "last_training_return": dates[origin - 1], "training_returns": lookback,
                       "forecast_count": len(rows), **fit.diagnostics})
        parameters.extend({"month": str(month), "asset": i, "omega": values[0],
                           "arch": values[1], "garch": values[2]} for i, values in enumerate(fit.garch))
        if progress:
            progress(f"{month}: DCC alpha={fit.parameters[0]:.5f}, beta={fit.parameters[1]:.5f}")
    np.linalg.cholesky(output)
    return output, pd.DataFrame(audits), pd.DataFrame(parameters)
