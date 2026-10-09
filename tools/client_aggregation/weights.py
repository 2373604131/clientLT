"""CPU-only convex weight solver. No model, feedback, or test data is used."""
import math
import time

import numpy as np


def problem(counts, strength):
    counts = np.asarray(counts, dtype=np.float64)
    if (counts.ndim != 2 or min(counts.shape) < 1 or not np.isfinite(counts).all()
            or (counts < 0).any() or (counts.sum(1) <= 0).any()
            or (counts.sum(0) <= 0).any() or not math.isfinite(strength) or strength <= 0):
        raise ValueError('Positive lambda, nonempty clients and coverage of every class are required')
    n = counts.sum(1)
    return n / n.sum(), counts / n[:, None], np.full(counts.shape[1], 1. / counts.shape[1])


def objective_gradient(weights, counts, strength):
    q, local, target = problem(counts, strength)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != q.shape or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('Invalid aggregation weights')
    mixture = local.T @ w
    if (mixture <= 0).any():
        raise ValueError('A class has zero mixture support')
    kl = float(np.sum(target * np.log(target / mixture)))
    chi = float(np.sum((w - q) ** 2 / q))
    gradient = -(local @ (target / mixture)) + strength * (w - q) / q
    return kl + .5 * strength * chi, gradient


def audit_weights(weights, counts, strength):
    q, local, target = problem(counts, strength)
    w = np.asarray(weights, dtype=np.float64)
    value, grad = objective_gradient(w, counts, strength)
    if abs(float(w.sum()) - 1.) > 1e-8:
        raise ValueError('Aggregation weights must sum to one')
    active = w > 1e-8
    multiplier = float(grad[active].mean())
    error = float(np.max(np.abs(grad[active] - multiplier)))
    if (~active).any():
        error = max(error, float(np.maximum(multiplier - grad[~active], 0).max()))
    baseline = objective_gradient(q, counts, strength)[0]
    if error > 1e-4 or value > baseline + 1e-9:
        raise ValueError('Convex solution failed KKT/objective audit: ' + str(error))
    mixture = local.T @ w
    return dict(objective=value, baseline_objective=baseline, kkt_max_error=error,
        kl_uniform_to_mixture=float(np.sum(target * np.log(target / mixture))),
        chi_square_to_sample=float(np.sum((w - q) ** 2 / q)),
        weights=w.tolist(), sample_weights=q.tolist(), mixture=mixture.tolist(), target=target.tolist())


def solve_weights(counts, strength=.1):
    from scipy.optimize import minimize
    q, _, _ = problem(counts, strength)
    started = time.perf_counter()
    # A numerical floor prevents log(0) at intermediate SLSQP trial points.
    result = minimize(lambda w: objective_gradient(w, counts, strength), q, jac=True,
        method='SLSQP', bounds=[(1e-12, 1.)] * len(q),
        constraints=[dict(type='eq', fun=lambda w: w.sum() - 1., jac=lambda w: np.ones(len(q)))],
        options=dict(ftol=1e-12, maxiter=2000))
    if not result.success:
        raise ValueError('Weight solver did not converge: ' + str(result.message))
    return dict(audit_weights(result.x, counts, strength), lambda_value=strength,
        solver='scipy SLSQP fp64', iterations=int(result.nit), seconds=time.perf_counter() - started)
