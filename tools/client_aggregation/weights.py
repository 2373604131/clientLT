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


def audit_weights(weights, counts, strength, lower_bounds=None):
    if lower_bounds is not None:
        return audit_lower_bounded_weights(weights, counts, strength, lower_bounds)
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


def audit_lower_bounded_weights(weights, counts, strength, lower_bounds):
    """KKT check at the actual lower bound, without modifying the supplied vector."""
    q, local, target = problem(counts, strength)
    w, lower = np.asarray(weights, dtype=np.float64), np.asarray(lower_bounds, dtype=np.float64)
    if (lower.shape != q.shape or not np.isfinite(lower).all() or (lower < 0).any()
            or (lower > q).any() or lower.sum() >= 1.):
        raise ValueError('Invalid lower bounds; FedAvg must be feasible with positive residual mass')
    value, gradient = objective_gradient(w, counts, strength)
    sum_error = abs(float(w.sum())-1.)
    slack = w-lower
    if sum_error > 1e-10 or float(slack.min()) < -1e-10:
        raise ValueError('Lower-bound feasibility/normalization audit failed')
    free = slack > 1e-8
    if not free.any():
        raise ValueError('No free variable for lower-bound KKT audit')
    kappa = float(gradient[free].mean())
    r_free = float(np.abs(gradient[free]-kappa).max())
    r_bound = float(np.maximum(kappa-gradient[~free], 0).max()) if (~free).any() else 0.
    error = max(r_free, r_bound)
    baseline = objective_gradient(q, counts, strength)[0]
    if error > 1e-4 or value > baseline+1e-9:
        raise ValueError('Lower-bound KKT/objective audit failed: '+str(error))
    mixture = local.T@w
    return dict(objective=value, baseline_objective=baseline, kkt_max_error=error,
        kkt_free_residual=r_free, kkt_bound_residual=r_bound, kappa=kappa,
        active_tol=1e-8, feasibility_tol=1e-10, kkt_tol=1e-4, objective_tol=1e-9,
        sum_error=sum_error, minimum_slack=float(slack.min()),
        free_clients=np.flatnonzero(free).tolist(), bound_clients=np.flatnonzero(~free).tolist(),
        lower_bounds=lower.tolist(), kl_uniform_to_mixture=float(np.sum(target*np.log(target/mixture))),
        chi_square_to_sample=float(np.sum((w-q)**2/q)), weights=w.tolist(),
        sample_weights=q.tolist(), mixture=mixture.tolist(), target=target.tolist())


def floor16_parameters(counts, tail_ids):
    q, _, _ = problem(counts, .1)
    matrix = np.asarray(counts, dtype=np.float64)
    if (not np.array_equal(matrix, np.floor(matrix)) or len(tail_ids) != 20
            or len(set(tail_ids)) != 20 or any(type(c) is not int or not 0 <= c < matrix.shape[1] for c in tail_ids)):
        raise ValueError('Floor16 requires integer counts and 20 distinct training-defined tail IDs')
    n = matrix.sum(1)
    total, tail = int(n.sum()), int(matrix[:,tail_ids].sum())
    denominator = total+16*tail
    gamma = total/denominator
    lower = gamma*q
    direct = n/denominator
    cross_error = float(np.max(np.abs(lower-direct)))
    if not 0 < gamma < 1 or cross_error > 1e-15:
        raise ValueError('Floor16 lower-bound cross-check failed')
    return dict(N=total, N_tail=tail, denominator=denominator, tailrw_multiplier=16,
        fedavg_floor_fraction=gamma, lower_bounds=lower.tolist(), lower_direct=direct.tolist(),
        lower_cross_check_max_error=cross_error,
        tailrw_weights=((n+16*matrix[:,tail_ids].sum(1))/denominator).tolist())


def audit_floor16(weights, counts, tail_ids, joint_weights, strength=.1):
    if strength != .1:
        raise ValueError('joint_floor16 fixes aggregation lambda at 0.1')
    params = floor16_parameters(counts, tail_ids)
    result = audit_weights(weights, counts, strength, lower_bounds=params['lower_bounds'])
    # Recompute every objective with the SAME counts/function/full-precision weights.
    original = audit_weights(joint_weights, counts, strength)
    q, _, _ = problem(counts, strength)
    tailrw = np.asarray(params['tailrw_weights'])
    if abs(float(tailrw.sum())-1.) > 1e-10 or np.min(tailrw-params['lower_bounds']) < -1e-10:
        raise ValueError('TailRW16 is not feasible for the floor constraint')
    objectives = dict(joint=original['objective'], joint_floor16=result['objective'],
        fedavg=objective_gradient(q, counts, strength)[0],
        tailrw16=objective_gradient(tailrw, counts, strength)[0])
    if (objectives['joint'] > objectives['joint_floor16']+1e-9
            or objectives['joint_floor16'] > min(objectives['fedavg'],objectives['tailrw16'])+1e-9):
        raise ValueError('Full-precision objective ordering failed')
    return dict(result, **{k:v for k,v in params.items() if k != 'lower_bounds'},
        objectives=objectives, joint_reference_weights=list(joint_weights),
        postprocessing='none; original solver vector', lambda_value=.1)


def solve_floor16(counts, tail_ids, joint_weights=None):
    """The single fixed revision; no gamma/lambda search and no weight mixing."""
    from scipy.optimize import minimize
    q, _, _ = problem(counts, .1)
    params = floor16_parameters(counts, tail_ids)
    if joint_weights is None:
        joint_weights = solve_weights(counts, .1)['weights']
    started = time.perf_counter()
    result = minimize(lambda w: objective_gradient(w, counts, .1), q, jac=True, method='SLSQP',
        bounds=list(zip(params['lower_bounds'], np.ones(len(q)))),
        constraints=[dict(type='eq', fun=lambda w: w.sum()-1., jac=lambda w: np.ones(len(q)))],
        options=dict(ftol=1e-12, maxiter=2000))
    if not result.success:
        raise ValueError('Floor16 solver failed: '+str(result.message))
    return dict(audit_floor16(result.x, counts, tail_ids, joint_weights),
        solver='scipy SLSQP fp64', iterations=int(result.nit), seconds=time.perf_counter()-started)
