"""Numerical design check only: no secure aggregation or model training is implemented.

Use archived CIFAR client counts as a local oracle to compare a one-aggregate
local scoring proposal with a two-aggregate restricted optimization proposal.
The latter is NOT recommended: sparse-class counts can be reconstructed when
its outputs are combined with the class-presence metadata exposed by old B.
This is not a distribution-privacy proof or a replacement training algorithm.
"""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.client_aggregation.weights import solve_weights, objective_gradient


def client_score(local_counts, global_counts):
    """Computed at a client, from its own counts and the released global sum."""
    return float(np.mean(local_counts / global_counts))


def server_line_search(global_frequency, candidate_mixture, prior, candidate, strength):
    """The server receives aggregates and scalar weights, not client histograms."""
    target = np.full(len(global_frequency), 1.0 / len(global_frequency))
    distance = float(np.sum((candidate - prior) ** 2 / prior))

    def objective(alpha):
        mixture = (1.0 - alpha) * global_frequency + alpha * candidate_mixture
        if np.any(mixture <= 0):
            return float('inf')
        return float(np.sum(target * np.log(target / mixture)) +
                     0.5 * strength * alpha ** 2 * distance)

    result = minimize_scalar(objective, bounds=(0., 1.), method='bounded',
                             options={'xatol': 1e-12})
    alpha = min([0., float(result.x), 1.], key=objective)
    return alpha, (1. - alpha) * prior + alpha * candidate, objective(alpha)


def main():
    source = ROOT / 'references/joint_floor16_seed42/aggregation_spec.json'
    archive = json.loads(source.read_text(encoding='utf-8'))
    counts = np.asarray(archive['matrix'], dtype=float)
    sizes = counts.sum(axis=1)
    prior = sizes / sizes.sum()
    local = counts / sizes[:, None]
    global_counts = counts.sum(axis=0)  # ORACLE emulation of first secure sum.
    frequency = global_counts / global_counts.sum()
    target = np.full(counts.shape[1], 1. / counts.shape[1])
    strength = .1
    full = solve_weights(counts, strength)
    full_weights = np.asarray(full['weights'])
    candidate = np.asarray([client_score(row, global_counts) for row in counts])
    assert abs(candidate.sum() - 1.) < 1e-12
    score_from_derivative = local @ (target / frequency)
    assert np.allclose(candidate, prior * score_from_derivative)
    tail_ids = np.asarray(archive['counts']['tail_ids'])
    top_tail = np.argsort(counts[:, tail_ids].sum(axis=1))[-3:]

    def metrics(weights):
        objective, _ = objective_gradient(weights, counts, strength)
        mixture = local.T @ weights
        return dict(objective=float(objective),
                    kl_uniform_to_mixture=float(np.sum(target * np.log(target / mixture))),
                    tail20_mixture_mass=float(mixture[tail_ids].sum()),
                    top3_tail_holder_weight=float(weights[top_tail].sum()),
                    weight_l1_to_full_joint=float(np.abs(weights - full_weights).sum()),
                    min_weight=float(weights.min()), max_weight=float(weights.max()))

    comparisons = dict(fedavg=metrics(prior), full_joint=metrics(full_weights),
                       local_class_balanced_weight=metrics(candidate))
    quantized_score = np.rint(candidate * (2 ** 12 - 1)).astype(int)
    quantized_candidate = quantized_score / quantized_score.sum()
    comparisons['local_class_balanced_12bit'] = metrics(quantized_candidate)
    details = dict(local_class_balanced_12bit=dict(
        candidate_weights=quantized_candidate.tolist(),
        transmitted_scores=quantized_score.tolist()))
    for name, bits in [('restricted_unquantized', None), ('restricted_12bit', 12)]:
        if bits is None:
            b = candidate.copy()
            transmitted = None
        else:
            transmitted = np.rint(candidate * (2 ** bits - 1)).astype(int)
            assert transmitted.sum() > 0
            b = transmitted / transmitted.sum()
        mixture = sum(b[j] * local[j] for j in range(len(prior)))
        # ORACLE emulation of second secure sum. Never a plaintext client upload.
        alpha, weights, value = server_line_search(frequency, mixture, prior, b, strength)
        direct = objective_gradient(weights, counts, strength)[0]
        assert abs(value - direct) < 1e-12
        assert abs(weights.sum() - 1.) < 1e-12 and weights.min() >= 0
        assert value <= comparisons['fedavg']['objective'] + 1e-12
        assert value >= comparisons['full_joint']['objective'] - 1e-9
        for point in [0., .2, .7, 1.]:
            trial = (1. - point) * prior + point * b
            direct_trial = objective_gradient(trial, counts, strength)[0]
            surrogate_mixture = (1. - point) * frequency + point * mixture
            expanded = (np.sum(target * np.log(target / surrogate_mixture)) +
                        .5 * strength * point ** 2 * np.sum((b-prior)**2/prior))
            assert abs(direct_trial-expanded) < 1e-12
        comparisons[name] = dict(alpha=alpha, **metrics(weights))
        details[name] = dict(candidate_weights=b.tolist(), weights=weights.tolist(),
                             transmitted_scores=None if transmitted is None else transmitted.tolist())

    # Counterexample to treating arbitrary repeated secure sums as private:
    # thirty known, nonnegative, normalized queries suffice to recover P.
    queries = .5 * np.eye(len(prior)) + .5 * np.tile(prior, (len(prior), 1))
    released = queries @ local
    recovered = np.linalg.solve(queries, released)
    recovery_error = float(np.max(np.abs(recovered-local)))
    assert recovery_error < 1e-12
    # A second weighted histogram can reveal individual rare-class counts when
    # old B makes their two holders known. The sum alone did not expose these.
    sparse_recoveries = []
    weighted_histogram = local.T @ quantized_candidate
    for label in np.flatnonzero((counts > 0).sum(axis=0) == 2):
        holders = np.flatnonzero(counts[:, label] > 0)
        system = np.vstack([np.ones(2), quantized_candidate[holders] / sizes[holders]])
        if np.linalg.matrix_rank(system) < 2:
            continue
        recovered_counts = np.linalg.solve(system, [global_counts[label], weighted_histogram[label]])
        error = float(np.max(np.abs(recovered_counts-counts[holders, label])))
        assert error < 1e-8
        sparse_recoveries.append(dict(class_id=int(label), holders=holders.tolist(),
                                      maximum_count_error=error))
    report = dict(
        source=str(source.relative_to(ROOT)), matrix_sha256=archive['matrix_sha256'],
        shape=list(counts.shape), lambda_value=strength, comparisons=comparisons,
        numerical_checks='PASS',
        naive_repeated_query_counterexample=dict(queries=len(prior),
            rank=int(np.linalg.matrix_rank(queries)), maximum_reconstruction_error=recovery_error),
        two_aggregate_sparse_recovery=sparse_recoveries,
        decision='Prefer testing one-aggregate local class-balanced weights. Reject the '
                 'two-aggregate variant as a privacy improvement for the present A+B protocol.',
        scope=dict(secure_aggregation_implemented=False, training_experiments_run=False,
                   formal_distribution_privacy_guarantee=False,
                   statement='Only algebra and proxy objectives were checked on archived counts.'),
        recommended_candidate_disclosure=dict(global_histogram=True, one_quantized_scalar_per_client=True,
                        one_fixed_weighted_histogram=False,
                        final_client_weights=True, raw_client_histograms=False,
                        caveat='Aggregate and weight outputs can leak; auxiliary information, '
                               'small counts and collusion remain relevant. No DP claim.'),
        candidate_details=details)
    destination = Path(__file__).with_name('numerical_design_check.json')
    destination.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('candidate_details',)}, indent=2))


if __name__ == '__main__':
    main()
