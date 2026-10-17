"""Validate completed J sweeps and plot measured, unsmoothed tail accuracy."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def collect_results(args):
    from scripts.run_cliplora_tail_concentration import run_path, verify_prepared
    summary, curves, evidence = [], [], []
    for value in sorted(args.lambdas):
        run = run_path(args, value)
        info = verify_prepared(args, value, run)
        done = read_json(run / 'completion.json')
        if done.get('completed_round') != 100 or done.get('normal_ab_events') != 100 or done.get('a_refresh_events') != 9:
            raise ValueError(f'Incomplete or wrong J training schedule: {run}')
        if done['total_optimizer_steps'] != info['statistics']['expected_optimizer_steps']:
            raise ValueError(f'Optimizer budget does not match actual client sizes: {run}')
        source = read_csv(run / 'protocol/partition_source.csv')
        actual = read_csv(run / 'partition_manifest.csv')
        canonical = lambda rows: sorted(tuple(int(r[k]) for k in ('raw_sample_id', 'class_id', 'client_id', 'local_position')) for r in rows)
        if canonical(source) != canonical(actual):
            raise ValueError(f'Training did not use the prepared partition: {run}')
        config = read_json(run / 'control_config.json')
        expected = {'method': 'j', 'normal_trainable_factor': 'AB', 'la_tau': 1.,
                    'a_lr_mult': 1., 'aggregation': 'sample_weighted_factor_fedavg',
                    'extra_trainable_factor': 'A', 'control_enabled': False}
        if any(config.get(k) != v for k, v in expected.items()):
            raise ValueError(f'Unexpected training baseline: {run}')
        slots = info['specification']['client_concurrency']
        if config.get('client_execution', {}).get('max_concurrent_clients') != slots:
            raise ValueError(f'Actual client concurrency differs from the registered setting: {run}')
        meta = read_json(run / 'bridge_metadata.json')
        if meta['client_sample_counts'] != info['statistics']['client_totals']:
            raise ValueError(f'Client totals changed in training: {run}')
        if slots > 1:
            for rnd, factor in [(r, 'AB') for r in range(1, 101)] + [(r, 'A') for r in range(10, 100, 10)]:
                execution = read_json(run / 'parallel_execution' / f'r{rnd:03d}_{factor}.json')
                clients = execution['selected_client_ids']
                audits = execution['client_audits']
                if (execution['slots'] != slots or execution['clients'] != 30
                        or execution['round'] != rnd or execution['factor'] != factor
                        or clients != meta['schedule'][rnd - 1]
                        or sorted(clients) != list(range(30))
                        or sorted(row['client_id'] for row in audits) != list(range(30))
                        or any(not 0 <= row['slot'] < slots for row in audits)):
                    raise ValueError(f'Invalid parallel execution record: {run}, round {rnd}, {factor}')
        common_keys = ('pool_sha256', 'test_sha256', 'probe_images_sha256',
                       'probe_manifest_sha256', 'schedule_sha256', 'initial_lora_sha256',
                       'frozen_model_sha256', 'training_code_hashes')
        common = {k: meta[k] for k in common_keys}
        if evidence and common != evidence[0]:
            raise ValueError(f'Global data, initialization, schedule or training code differs: {run}')
        evidence.append(common)
        rows = sorted(read_csv(run / 'round_metrics.csv'), key=lambda r: int(r['round']))
        if [int(r['round']) for r in rows] != list(range(101)):
            raise ValueError(f'Expected one observation at every round 0..100: {run}')
        if any(r['method'] != 'j' or int(r['seed']) != args.seed for r in rows):
            raise ValueError(f'Unexpected metric provenance: {run}')
        tail = np.array([float(r['bottom20_tail_acc']) for r in rows])
        if not np.all(np.isfinite(tail)) or np.any((tail < 0) | (tail > 100)):
            raise ValueError(f'Invalid tail accuracy: {run}')
        peak_index = 1 + int(np.argmax(tail[1:]))
        stats = info['statistics']
        summary.append({
            'lambda': value, 'actual_tail_share_percent': 100 * stats['actual_tail_share'],
            'tail_client_fedavg_weight_percent': 100 * stats['tail_client_fedavg_weight'],
            'mean_tail_client_coverage': stats['mean_tail_client_coverage'],
            'initial_tail_acc': float(tail[0]), 'peak_round': peak_index,
            'peak_tail_acc': float(tail[peak_index]), 'final_tail_acc': float(tail[-1]),
            'last20_tail_acc': float(tail[81:101].mean()),
            'peak_to_final_pp': float(tail[peak_index] - tail[-1]),
            'peak_to_last20_pp': float(tail[peak_index] - tail[81:101].mean()),
            'last20_overall_acc': float(np.mean([float(r['overall_acc']) for r in rows[81:101]])),
            'optimizer_steps': done['total_optimizer_steps'], 'client_concurrency': slots,
        })
        curves.extend({'lambda': value, 'round': int(r['round']),
                       'tail_acc': float(r['bottom20_tail_acc']),
                       'overall_acc': float(r['overall_acc'])} for r in rows)
    return summary, curves, evidence


def plot_results(args):
    # Refuse partial comparisons: every requested run must finish and pass audit.
    summary, curves, evidence = collect_results(args)
    from scripts.cliplora_tail_concentration_protocol import write_csv, write_json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    destination = args.output_root.resolve() / f'seed{args.seed}_split{args.protocol_seed}' / 'analysis'
    destination.mkdir(exist_ok=True)
    write_csv(destination / 'summary.csv', summary)
    write_csv(destination / 'curves.csv', curves)
    write_json(destination / 'comparison_audit.json', {
        'all_requested_runs_complete': True, 'common_evidence': evidence[0],
        'lambdas': sorted(args.lambdas), 'seed': args.seed,
        'smoothing': False, 'uncertainty_bands': False,
        'decline_definition': 'max(tail accuracy over rounds 1..100) minus tail accuracy at round 100',
        'scope': 'Paired single-seed intervention; client totals, weights and coverage can change.',
    })
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'legend.fontsize': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#81858B', 'axes.linewidth': .6,
                         'savefig.facecolor': 'white'})
    colors = {0.: '#5975A2', .25: '#8DAACF', .5: '#76896B', .75: '#C28256', 1.: '#9B7E9C'}

    def save(fig, stem):
        for extension in ('pdf', 'svg', 'png'):
            fig.savefig(destination / f'{stem}.{extension}', dpi=600)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(84 / 25.4, 59 / 25.4), layout='constrained')
    for value in sorted(args.lambdas):
        points = [r for r in curves if r['lambda'] == value]
        ax.plot([r['round'] for r in points], [r['tail_acc'] for r in points],
                color=colors[value], linewidth=1.2, label=rf'$\lambda={value:g}$')
    ax.set(xlabel='Communication round', ylabel='Tail accuracy (%)', xlim=(0, 100))
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis='y', color='#E5E7EB', linewidth=.5)
    ax.legend(frameon=False, ncol=3, handlelength=1.2, columnspacing=.8,
              loc='lower center', bbox_to_anchor=(.5, 1.01), borderaxespad=0.)
    save(fig, 'tail_accuracy_by_round')

    fig, ax = plt.subplots(figsize=(84 / 25.4, 52 / 25.4), layout='constrained')
    x = [r['actual_tail_share_percent'] for r in summary]
    y = [r['peak_to_final_pp'] for r in summary]
    ax.plot(x, y, color='#C28256', marker='o', markersize=3.2, linewidth=1.2)
    ax.set(xlabel='Tail samples on designated clients (%)', ylabel='Peak-to-final decline (pp)')
    ax.grid(axis='y', color='#E5E7EB', linewidth=.5)
    save(fig, 'decline_by_concentration')
    (destination / 'caption.txt').write_text(
        'Tail-class retention under increasing tail concentration. The same J baseline '
        '(CLIP ViT-B/16, rank-4 LoRA, LA tau=1, sample-weighted factor FedAvg, joint A/B '
        'training for three local epochs per round and one extra A epoch at rounds 10,20,...,90) '
        'is trained for 100 rounds on CIFAR-100-LT (IF=100, 30 clients, full participation). '
        'The global image pool, alpha=0.5, initialization, participation schedule and all '
        'non-tail sample-to-client assignments are fixed across lambda values. The fraction '
        'of tail samples allocated to clients 27-29 follows 0.1+0.9*lambda, with integer rounding. '
        'Client totals and aggregation weights vary naturally. Lambda=0 is not ordinary '
        'class-wise Dirichlet. Curves are unsmoothed observations from training seed '
        f'{args.seed} and split seed {args.protocol_seed}; no multi-seed uncertainty is shown.\n',
        encoding='utf-8')
    print(f'Validated results and vector figures: {destination}', flush=True)
