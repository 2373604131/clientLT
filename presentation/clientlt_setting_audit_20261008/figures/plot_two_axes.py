"""Vivid marginal-heatmap recipe adapted as a 2 x 2 partition construction.
Counts come from repository functions on constructed labels, not training runs.
"""
import json
import importlib.util
import logging
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from figure_common import HERE, ROOT, setup, save
from plot_partition import marginal_panel, colorbar
sys.path.insert(0, str(ROOT))
from utils.datasplit import partition_client_longtail
logging.getLogger('fontTools.subset').setLevel(logging.WARNING)
spec = importlib.util.spec_from_file_location('repo_long_tail', ROOT / 'datasets/long_tail.py')
long_tail = importlib.util.module_from_spec(spec)
spec.loader.exec_module(long_tail)
_get_img_num_per_cls = long_tail._get_img_num_per_cls

setup()
fig = plt.figure(figsize=(7.2, 6.9))
gs = fig.add_gridspec(2, 2, left=.08, right=.96, bottom=.22, top=.79,
                      wspace=.29, hspace=.76)
records = []
for row, imbalance in enumerate([10, 100]):
    counts = np.array(_get_img_num_per_cls(np.empty(50000), 100, 1 / imbalance, 'exp'))
    labels = np.repeat(np.arange(100), counts)
    if imbalance == 100:
        actual = pd.read_csv(HERE / 'data/counts_client-longtail.csv').iloc[:, 1:].to_numpy()
        assert np.array_equal(counts, actual.sum(axis=0))
    for col, concentration in enumerate([0., .75]):
        allocation = partition_client_longtail(
            labels, 30, 100, specialization_lambda=concentration,
            intra_group_alpha=.5, head_leakage_scale=3., rng=np.random.RandomState(42))
        matrix = np.array([np.bincount(labels[allocation[k]], minlength=100) for k in range(30)])
        assert np.array_equal(matrix.sum(axis=0), counts)
        assert sum(map(len, allocation.values())) == len(labels)
        share = matrix[27:, 80:].sum() / matrix[:, 80:].sum()
        rec = {'IF': imbalance, 'lambda_T': concentration, 'alpha_T': .5, 'q_T': .1,
               'rho': 3., 'seed': 42, 'source_type': 'constructed_labels_only_no_training',
               'samples': int(matrix.sum()), 'tail_samples': int(matrix[:, 80:].sum()),
               'tail_share_clients_27_29': float(share)}
        records.append(rec)
        pd.DataFrame(matrix, columns=[f'class_{i}' for i in range(100)]).rename_axis('client_id').to_csv(
            HERE / f'data/construction_IF{imbalance}_lambda{concentration:g}.csv')
        title = f'IF = {imbalance}, ' + rf'$\lambda_T={concentration:g}$' + f'\nTail in clients 27–29: {share:.1%}'
        sm = marginal_panel(fig, gs[row, col], matrix, title,
                            right_max=1400, label_y=(col == 0))
colorbar(fig, sm, y=.107)
fig.text(.08, .975, 'Two controls: class imbalance and client concentration', fontsize=11,
         fontweight='bold', va='top')
fig.text(.08, .926, r'Across: $\lambda_T$ increases  |  Down: IF increases  |  $\alpha_T=0.5$, $q_T=0.1$, $\rho=3$', fontsize=8)
fig.text(.08, .883, 'Partition construction only; these four panels are not training results.', fontsize=8)
fig.text(.08, .024, r'$\lambda_T=0$ is still Client-LT; it is not class-wise Dirichlet.', fontsize=8)
(HERE / 'data/two_axis_construction.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
save(fig, 'two_axis_construction')
