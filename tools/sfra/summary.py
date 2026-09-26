"""Offline summaries and an analysis-only archive; never select a checkpoint."""
import csv
import json
from pathlib import Path
import tarfile


METRICS = ('overall_acc', 'head20_acc', 'middle60_acc', 'bottom20_tail_acc', 'non_tail_acc')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root, reference=None):
    root = Path(root)
    out = root/'analysis'
    out.mkdir(parents=True, exist_ok=True)
    paths = [p.parent for p in sorted(root.glob('seed*/*/*/*/sfra_config.json'))]
    if reference is not None:
        paths.append(Path(reference))
    performance, costs, curves, mechanisms, corrections, status = [], [], [], [], [], []
    transfer_rounds, transfer_receivers, transfer_steps, transfer_probes = [], [], [], []
    transfer_loss_traces = []
    problem2_tables = {name: [] for name in ('class_weights', 'class_loss_trace', 'class_changes', 'class_change_summary')}
    aggregation_rows = []
    for run in paths:
        config_file = run/'sfra_config.json'
        new = config_file.is_file()
        cfg = json.loads((config_file if new else run/'control_config.json').read_text(encoding='utf-8'))
        label = str(run)
        complete = (run/'completion.json').is_file()
        status.append(dict(run=label, complete=complete))
        if not complete:
            continue
        rows = read_csv(run/'round_metrics.csv')
        if sorted(int(r['round']) for r in rows) != list(range(101)):
            raise ValueError(f'Expected exactly rounds 0..100: {run}')
        final20 = [r for r in rows if 81 <= int(r['round']) <= 100]
        info = dict(run=label, method=cfg['variant'] if new else cfg['method'],
                    partition=cfg['partition'] if new else cfg['topology'], seed=cfg['seed'],
                    retention_weight=cfg.get('retention_weight', ''),
                    classification_weight=cfg.get('classification_weight', ''),
                    b_aggregation=cfg.get('b_aggregation', {}).get('mode', 'sample'),
                    b_transfer_enabled=bool(cfg.get('b_transfer')))
        if info['b_aggregation'] == 'uniform-transfer-rounds':
            info['method'] += '+uniform8'
        transfer = cfg.get('b_transfer')
        if transfer:
            mode = transfer.get('mode', 'local')
            info.update(method=info['method']+('+B-shared' if mode == 'shared' else '+B-transfer'),
                        transfer_mode=mode, transfer_lr=transfer['learning_rate'],
                        transfer_probe_step=transfer['probe_step'], transfer_reg=transfer['regularization'])
            if mode == 'shared':
                info.update(transfer_non_tail_sampling=transfer.get('non_tail_sampling', 'sample'),
                            transfer_tail_weight=transfer.get('tail_weight', .5))
                if transfer.get('calibration_profile') == 'coverage_tradeoff':
                    info['method'] += '-tradeoff'
                if transfer.get('calibration_profile') == 'problem2':
                    info.update(problem2_variant=transfer['problem2_variant'], harm_beta=transfer['harm_beta'])
                    info['method'] += '-problem2-' + transfer['problem2_variant']
        row = dict(info)
        for metric in METRICS:
            row['last20_'+metric] = sum(float(r[metric]) for r in final20)/20
        if transfer and transfer.get('calibration_profile') == 'problem2':
            from tools.sfra.b_problem2 import frequency_metrics
            row.update(frequency_metrics(run))
        trained = [r for r in rows if int(r['round']) > 0]
        peak = max(trained, key=lambda r:float(r['bottom20_tail_acc']))
        row['tail_peak_round'] = int(peak['round'])
        row['tail_peak_to_final'] = float(peak['bottom20_tail_acc'])-float(rows[-1]['bottom20_tail_acc'])
        performance.append(row)
        curves.extend({**info, **r} for r in rows)
        if new:
            if cfg.get('b_aggregation'):
                sizes = json.loads((run/'bridge_metadata.json').read_text(encoding='utf-8'))['client_sample_counts']
                for event in read_csv(run/'event_manifest.csv'):
                    clients = json.loads(event['selected_client_ids'])
                    weights = json.loads(event['server_weights'])
                    for client, weight in zip(clients, weights):
                        aggregation_rows.append({**info, 'round':int(event['round']), 'phase':event['phase'],
                            'client_id':client, 'sample_weight':sizes[client]/sum(sizes),
                            'aggregation_weight':weight})
            rr = read_csv(run/'sfra_rounds.csv')
            cc = read_csv(run/'sfra_costs.csv')
            costs.append({**info, **{name:sum(float(r[name]) for r in cc)
                          for name in ('seconds','forward_images','backward_images','downlink_bytes','upload_bytes')}})
            for name in ('classification_forward_images', 'classification_backward_images'):
                if any(name in r for r in cc):
                    costs[-1][name] = sum(float(r.get(name) or 0) for r in cc)
            mechanisms.extend({**info, **r} for r in rr)
            for path in sorted((run/'sfra_rounds').glob('r*/correction_steps.csv')):
                corrections.extend({**info, 'round':int(path.parent.name[1:]), **r} for r in read_csv(path))
            if transfer:
                transfer_rounds.extend({**info, **r} for r in read_csv(run/'b_transfer_rounds.csv'))
                for folder in sorted((run/'b_transfer_rounds').glob('r*')):
                    for name, destination in [('receiver_summary', transfer_receivers),
                                               ('optimization_steps', transfer_steps),
                                               ('probe_metrics', transfer_probes)]:
                        destination.extend({**info, **r} for r in read_csv(folder/f'{name}.csv'))
                    trace_path = folder/'c_loss_trace.csv'
                    if trace_path.is_file():
                        transfer_loss_traces.extend({**info, **r} for r in read_csv(trace_path))
                    for name, destination in problem2_tables.items():
                        if (folder/f'{name}.csv').is_file():
                            destination.extend({**info, **r} for r in read_csv(folder/f'{name}.csv'))
    for name, rows in [('performance',performance),('functional_costs',costs),('curves',curves),
                       ('mechanisms',mechanisms),('correction_steps',corrections),('status',status)]:
        write_csv(out/f'{name}.csv', rows)
    if aggregation_rows:
        write_csv(out/'aggregation_weights.csv', aggregation_rows)
    if transfer_rounds:
        for name, rows in [('b_transfer_rounds', transfer_rounds), ('b_transfer_receivers', transfer_receivers),
                           ('b_transfer_steps', transfer_steps), ('b_transfer_probes', transfer_probes)]:
            write_csv(out/f'{name}.csv', rows)
    if transfer_loss_traces:
        write_csv(out/'b_transfer_loss_trace.csv', transfer_loss_traces)
    for name, rows in problem2_tables.items():
        if rows:
            write_csv(out/f'b_problem2_{name}.csv', rows)
    lines = ['# SFRA results', '', 'Primary endpoint: committed rounds 81–100, no best-checkpoint selection.', '',
             '| Run | Lambda | Mu | B non-tail sampling | B tail weight | Overall | Head20 | Middle60 | Tail20 | Tail peak-to-final |',
             '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in performance:
        values = ' | '.join(f'{r["last20_"+key]:.3f}' for key in METRICS[:4])
        lines.append(f'| {r["run"]} | {r["retention_weight"]} | {r["classification_weight"]} | '
                     f'{r.get("transfer_non_tail_sampling", "")} | {r.get("transfer_tail_weight", "")} | '
                     f'{values} | {r["tail_peak_to_final"]:.3f} |')
    lines += ['', f'Completed: {len(performance)} / discovered: {len(status)}.',
              'Compare Full vs S, Full vs Current, and Full vs Flat at the SAME lambda.',
              'For full-cp, compare against the existing Full-10 first: retention lambda stays 10; mu is the new search parameter.',
              'Functional costs exclude unchanged local training and official test; these are in budget.csv and evaluation_budget.csv.',
              'Classification forward_images are shared with functional evaluation, not an extra count to add to total forward_images; backward_images already includes both gradient traversals.',
              'tokens.npz rows correspond to private_witness_manifest.json, with an explicit active/history-valid mask.',
              'F_post_B(t+1) provides the next-round B retention measurement for F_committed(t).',
              'Current/Full/Flat have the same rules and budget, not numerically identical evolving targets.',
              'This is a single-seed development comparison, not significance or test-independent parameter selection.']
    if aggregation_rows:
        lines += ['', '## Eight-round B aggregation control', '',
                  'uniform-transfer-rounds changes the WHOLE local B aggregation to 1/K at rounds 30,40,...,100 only.',
                  'The no-transfer control uses exactly the same eight intervention rounds.',
                  'All other B rounds, all A proposals, and the classification-preservation weights remain sample-weighted.',
                  'Donor selection and C optimization still use the original local B; this is not shared-C training.',
                  'Compare transfer on/off under the SAME B aggregation rule; then compare the transfer increment across rules.',
                  'aggregation_weights.csv contains the actual audited weights for both B and A phases.']
    if transfer_rounds:
        lines += ['', '## Donor B transfer', '',
                  'Keep A settings fixed. In the original sweep only C learning rate varies; in the aggregation control C learning is unchanged.',
                  'Compare A+B against A-only with the SAME variant, lambda, mu, partition and training protocol.',
                  'Use performance.csv for official test outcomes; transfer probe scores are training-side diagnostics.',
                  'b_transfer_rounds.csv contains additional algorithm/diagnostic image counts and communication costs, separate from SFRA functional costs.',
                  'Local-mode receiver rows are local before/after; shared-mode rows compare the same ordinary/transferred shared B.',
                  'committed_global includes the subsequent A update (none in round 100); it is not a no-transfer counterfactual.',
                  'Local non-tail monitoring is a fixed at-most-16-image subset, not the full non-tail test set.',
                  'Transfer group scores average present classes within each receiver, then average receivers; they are not official global class-macro accuracy.']
    if any(r.get('transfer_mode') == 'shared' for r in performance):
        lines += ['', '## Shared-model B transfer', '',
                  'B-shared uses the original tail-present recipients, two calibration batches and unchanged tail sample positions.',
                  'The original profile keeps image-uniform non-tail sampling and 50:50 group LA; the tradeoff profile is labeled separately.',
                  'All clients see the same C, with one C regularizer. Original/tradeoff profiles average recipient losses; problem2 weights are recorded separately.',
                  'Two global optimizer steps are NOT two image batches: compare algorithm_backward_images and client_backward_batches.',
                  'Ordinary B FedAvg and all A rules are unchanged. The residual is added once AFTER ordinary B aggregation.',
                  'Normal-B event dumps remain raw ordinary FedAvg; their state_role and shared_transfer_state_path link the separate commit.pt.',
                  'No local_tail_la_gain is reported for shared C; shared_receiver_tail_la_gain_mean uses the shared model.',
                  'The old local donor screen and new shared donor screen can accept different donors; this is not a pure placement-only change.']
        if transfer_loss_traces:
            lines += ['', '### C learning-rate diagnostics', '',
                      'b_transfer_loss_trace.csv evaluates the SAME TWO calibration batches at C=0, after step 1 and after step 2.',
                      'Use fixed_pool_objective for cross-step comparisons; problem2 additionally includes its configured harm term. Original la_before values use different step batches.',
                      'b_transfer_steps.csv also records same_batch_la_after and objective_after_same_batch for paired before/after checks.',
                      'Tail/non-tail accuracy is raw-logit, group sample-mean per client/batch, then averaged; it is neither the official test metric nor independent validation.',
                      'Post-step traces are diagnostic forwards; problem2 counts C=0 reference forwards as algorithm work for every arm. No trace picks a step or reverts an update.',
                      'Earlier completed rounds have no trace if they ran before diagnostic logging was added; resume only records subsequent rounds.']
    if any(r.get('method', '').endswith('+B-shared-tradeoff') for r in performance):
        lines += ['', '## Shared-C coverage / tradeoff pilot', '',
                  'Only non-tail calibration sampling and the tail/non-tail LA loss fraction change.',
                  'Class-cyclic sampling keeps each original tail batch and the exact non-tail image count; a separate RNG preserves training randomness.',
                  'transfer_tail_weight is a LOCAL LOSS fraction, NOT an aggregation weight, donor multiplier or A retention parameter.',
                  'Tail-only recipients still use their full tail mean. Shared C and the single regularizer are unchanged.',
                  'Donors, rank-by-rank C, two synchronous Adam steps, transfer rounds, A and ordinary B FedAvg remain unchanged.',
                  'calibration_manifest.json stores the actual positions; client_feedback_steps.csv records the actual group loss weights.',
                  'Compare the 0.5 coverage-only configuration against the existing shared-C run, then compare the 0.35 / 0.2 tradeoff settings.',
                  'Report the whole predeclared sweep; no test-based update gate or checkpoint rollback is introduced.']
    if any(r.get('problem2_variant') for r in performance):
        lines += ['', '## Problem 2 shared-C controls', '',
                  'E00/E10/E01/E11 use all selected original donor updates; the positive-gain union no longer selects sources.',
                  'Class averaging preserves each step\'s original actual tail/non-tail mass, including tail-only clients.',
                  'Harm is the positive per-receiver-class loss increase relative to C=0 on the SAME step images.',
                  'Class changes use common macro metrics and record calibration overlap; they are not held-out test estimates.',
                  'performance.csv includes problem2_variant, harm_beta and available last20 many/medium/few accuracy from epochs 80..99.',
                  'If frequency_metrics_available is false, recover the class prior and all per-class files before interpreting those groups.']
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'Summary written: {out/"report.md"}', flush=True)


def pack(root):
    root = Path(root).resolve()
    destination = root.parent/(root.name+'_analysis.tar.gz')
    # Keep scalar/token evidence, drop large models, per-client full updates and raw images.
    allowed = {'.csv','.json','.yaml','.md','.npz'}
    with tarfile.open(destination, 'w:gz') as archive:
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in allowed:
                archive.add(path, arcname=str(Path(root.name)/path.relative_to(root)))
    print(f'Analysis archive: {destination} (not a training/resume checkpoint)', flush=True)
