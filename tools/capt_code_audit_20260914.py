"""Read-only CAPT source/artifact audit; writes only to its own output directory."""
from pathlib import Path
import ast
import csv
import json
import statistics
import hashlib
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'capt_code_audit_20260914'


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def matrix(path):
    records = rows(path / 'client_class_counts.csv')
    records.sort(key=lambda r: int(r['client_id']))
    return np.array([[int(float(r[f'class_{c}'])) for c in range(100)] for r in records])


def audit_run(path):
    n = matrix(path)
    proportions = n / n.sum(axis=1, keepdims=True)
    eligible = proportions > 0.1
    metrics = rows(path / 'round_metrics.csv')
    last = max(metrics, key=lambda r: int(r['epoch']))
    result = dict(path=str(path.relative_to(ROOT)), evaluated_rows=len(metrics),
                  last_epoch=int(last['epoch']),
                  final={k: float(last[k]) for k in ['overall_acc', 'head_acc', 'tail_acc']},
                  clients=len(n), samples=int(n.sum()),
                  class_count_min=int(n.sum(0).min()), class_count_max=int(n.sum(0).max()))
    for label, sl in [('all', slice(None)), ('head', slice(0, 80)), ('tail', slice(80, 100))]:
        any_eligible = eligible.any(0)[sl]
        result[label] = dict(classes_with_eligible_client=int(any_eligible.sum()),
                             classes_never_eligible=int((~any_eligible).sum()),
                             mean_support_clients=float((n > 0).sum(0)[sl].mean()),
                             mean_eligible_clients=float(eligible.sum(0)[sl].mean()),
                             mean_effective_carriers=float((n.sum(0)**2 / (n**2).sum(0))[sl].mean()))
    result['eligible_tail_classes'] = (np.where(eligible.any(0)[80:])[0] + 80).tolist()
    snapshots = sorted((path/'prompt_params').glob('prompt_params_epoch_*.pth'),
                       key=lambda p: int(p.stem.rsplit('_', 1)[-1]))
    if len(snapshots) >= 2:
        first = torch.load(snapshots[0], map_location='cpu', weights_only=True)
        last_snapshot = torch.load(snapshots[-1], map_location='cpu', weights_only=True)
        first_p = first['class_aware_prompt'].float()
        delta = last_snapshot['class_aware_prompt'].float() - first_p
        row_norm = delta.flatten(1).norm(dim=1)
        never_eligible = torch.from_numpy(~eligible.any(0))
        result['snapshots'] = dict(first_epoch=int(first['epoch']), last_epoch=int(last_snapshot['epoch']),
                                  tail_row_delta_norms=row_norm[80:].tolist(),
                                  never_eligible_delta_max=float(row_norm[never_eligible].max()) if never_eligible.any() else None,
                                  never_eligible_relative_delta_max=float((row_norm[never_eligible]/first_p.flatten(1).norm(dim=1)[never_eligible]).max()) if never_eligible.any() else None,
                                  general_delta_norm=float((last_snapshot['general_prompt'].float()-first['general_prompt'].float()).norm()))
    schedule_file = path / 'selected_clients.csv'
    if schedule_file.exists():
        schedules = {}
        for r in rows(schedule_file):
            schedules.setdefault(int(r['epoch_index']), []).append(int(r['client_id']))
        # Number of rounds where the actual threshold branch can replace a row.
        active = np.stack([eligible[ids].any(0) for _, ids in sorted(schedules.items())])
        result['executed_rounds'] = len(active)
        result['tail_upload_rounds_mean'] = float(active[:, 80:].sum(0).mean())
        result['tail_preserve_fraction'] = float((~active[:, 80:]).mean())
        result['tail_upload_rounds_by_class'] = active[:, 80:].sum(0).tolist()
    return result


def source_checks():
    # Execute just the original aggregation functions; do not import training entry points.
    source = (ROOT / 'federated_main.py').read_text(encoding='utf-8-sig')
    wanted = {'aggregate_class_aware_prompts', 'communicate_within_cluster_similarity'}
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<original CAPT aggregation>', 'exec'), namespace)
    key = 'prompt_learner.class_aware_ctx'
    local = [{key: torch.tensor([[[2.]], [[4.]]])}, {key: torch.tensor([[[8.]], [[10.]]])}]
    base = torch.tensor([[[99.]], [[88.]]])
    aggregated = namespace['aggregate_class_aware_prompts'](
        np.array([[.2, .1], [.7, .1]]), local, [0, 1], 2, base)
    assert torch.equal(aggregated, torch.tensor([[[5.]], [[88.]]]))
    namespace['communicate_within_cluster_similarity']([0, 1], local)
    cluster_result = namespace['aggregate_class_aware_prompts'](
        np.array([[.2, .1], [0., .1]]), local, [0, 1], 2, base)
    assert cluster_result[0].item() == 5.  # ineligible client contributes through cluster averaging

    torch.manual_seed(7)
    s = torch.randn(8, 1, dtype=torch.double, requires_grad=True)
    general = -(s.exp() / s.exp().sum()).log().mean()
    expected = torch.logsumexp(s.flatten(), 0) - s.mean()
    assert torch.allclose(general, expected)
    g1 = torch.autograd.grad(general, s, retain_graph=True)[0]
    g2 = torch.autograd.grad(expected, s)[0]
    assert torch.allclose(g1, g2)
    prior = torch.tensor([.7, .2, .1], dtype=torch.double)
    aux = torch.randn(8, 3, dtype=torch.double)
    y = torch.tensor([0, 1, 2, 1, 2, 2, 0, 1])
    original_aux = -(prior[y] * aux.gather(1, y[:, None]).squeeze().exp()
                     / (prior[None] * aux.exp()).sum(1)).log().mean()
    assert torch.allclose(original_aux, F.cross_entropy(aux + prior.log(), y))
    # Demonstrate step-then-mask cannot undo the update of an absent class.
    w = torch.nn.Parameter(torch.zeros(3, dtype=torch.double))
    opt = torch.optim.SGD([w], lr=.1)
    F.cross_entropy(w[None], torch.tensor([0])).backward()
    opt.step()
    w.grad[1:] = 0
    assert w[1].item() != 0
    return dict(equal_weight_aggregation=aggregated.flatten().tolist(),
                ineligible_client_survives_cluster=cluster_result[0].item(),
                general_loss_equals_logsumexp_minus_mean=True,
                general_loss_shift_invariant=bool(torch.allclose(
                    general, torch.logsumexp((s+10).flatten(), 0)-(s+10).mean())),
                class_aware_loss_equals_prior_adjusted_ce=True,
                post_step_mask_absent_class_change=w[1].item(),
                general_loss_lower_bound_batch32=float(np.log(32)),
                trainable_params_100classes_4ctx_1general=512+100*3*512+512*768+768)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = list((ROOT/'output/cifar100_LT/capt_main_matched').glob('seed*/frac0.4/*'))
    candidates += list((ROOT/'output/online_sca_seed42_v2/stage1b_capt').glob('capt_*'))
    candidates += list((ROOT/'output/expC_lambda_response_frac02/method=capt').glob('lambda=*/seed=*'))
    candidates += [ROOT/'output/v2_capt_analysis/output/cifar100_LT/v2_matched/seed42/capt']
    candidates += list((ROOT/'output/cifar100_LT/CAPT_cluster_vit_b16_batchSize32/ExperimentD_AlignedPartitionCompare').glob('partition=*'))
    results = [audit_run(p) for p in sorted(candidates) if (p/'client_class_counts.csv').exists()]
    payload = dict(source_checks=source_checks(), runs=results,
                   source_sha256={str(p): hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                                  for p in ['trainers/capt.py', 'loss/prompt_loss.py', 'federated_main.py',
                                            'clip/model.py', 'utils/datasplit.py', 'Dassl/dassl/engine/trainer.py']})
    (OUT/'audit.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(payload['source_checks'], indent=2))
    for r in results:
        print(r['path'], 'epoch', r['last_epoch'], 'metrics', r['final'],
              'eligible_all/tail', r['all']['classes_with_eligible_client'], r['tail']['classes_with_eligible_client'],
              'tail_preserve', r.get('tail_preserve_fraction'))


if __name__ == '__main__':
    main()
