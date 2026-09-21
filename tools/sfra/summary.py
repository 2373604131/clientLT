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
                    classification_weight=cfg.get('classification_weight', ''))
        transfer = cfg.get('b_transfer')
        if transfer:
            info.update(method=info['method']+'+B-transfer', transfer_lr=transfer['learning_rate'],
                        transfer_probe_step=transfer['probe_step'], transfer_reg=transfer['regularization'])
        row = dict(info)
        for metric in METRICS:
            row['last20_'+metric] = sum(float(r[metric]) for r in final20)/20
        trained = [r for r in rows if int(r['round']) > 0]
        peak = max(trained, key=lambda r:float(r['bottom20_tail_acc']))
        row['tail_peak_round'] = int(peak['round'])
        row['tail_peak_to_final'] = float(peak['bottom20_tail_acc'])-float(rows[-1]['bottom20_tail_acc'])
        performance.append(row)
        curves.extend({**info, **r} for r in rows)
        if new:
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
    for name, rows in [('performance',performance),('functional_costs',costs),('curves',curves),
                       ('mechanisms',mechanisms),('correction_steps',corrections),('status',status)]:
        write_csv(out/f'{name}.csv', rows)
    if transfer_rounds:
        for name, rows in [('b_transfer_rounds', transfer_rounds), ('b_transfer_receivers', transfer_receivers),
                           ('b_transfer_steps', transfer_steps), ('b_transfer_probes', transfer_probes)]:
            write_csv(out/f'{name}.csv', rows)
    lines = ['# SFRA results', '', 'Primary endpoint: committed rounds 81–100, no best-checkpoint selection.', '',
             '| Run | Lambda | Mu | Overall | Head20 | Middle60 | Tail20 | Tail peak-to-final |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in performance:
        values = ' | '.join(f'{r["last20_"+key]:.3f}' for key in METRICS[:4])
        lines.append(f'| {r["run"]} | {r["retention_weight"]} | {r["classification_weight"]} | {values} | {r["tail_peak_to_final"]:.3f} |')
    lines += ['', f'Completed: {len(performance)} / discovered: {len(status)}.',
              'Compare Full vs S, Full vs Current, and Full vs Flat at the SAME lambda.',
              'For full-cp, compare against the existing Full-10 first: retention lambda stays 10; mu is the new search parameter.',
              'Functional costs exclude unchanged local training and official test; these are in budget.csv and evaluation_budget.csv.',
              'Classification forward_images are shared with functional evaluation, not an extra count to add to total forward_images; backward_images already includes both gradient traversals.',
              'tokens.npz rows correspond to private_witness_manifest.json, with an explicit active/history-valid mask.',
              'F_post_B(t+1) provides the next-round B retention measurement for F_committed(t).',
              'Current/Full/Flat have the same rules and budget, not numerically identical evolving targets.',
              'This is a single-seed development comparison, not significance or test-independent parameter selection.']
    if transfer_rounds:
        lines += ['', '## Donor B transfer', '',
                  'Only C learning rate varies in the initial sweep; keep A settings fixed.',
                  'Compare A+B against A-only with the SAME variant, lambda, mu, partition and training protocol.',
                  'Use performance.csv for official test outcomes; transfer probe scores are training-side diagnostics.',
                  'b_transfer_rounds.csv contains additional algorithm/diagnostic image counts and communication costs, separate from SFRA functional costs.',
                  'Local before/after and ordinary/transferred global B scores use the same fixed witness images.',
                  'committed_global includes the subsequent A update (none in round 100); it is not a no-transfer counterfactual.',
                  'Local non-tail monitoring is a fixed at-most-16-image subset, not the full non-tail test set.',
                  'Transfer group scores average present classes within each receiver, then average receivers; they are not official global class-macro accuracy.']
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
