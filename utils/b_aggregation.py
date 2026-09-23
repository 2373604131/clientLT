"""The eight-round B aggregation intervention; no change to local training."""

B_TRANSFER_ROUNDS = tuple(range(30, 101, 10))


def phase_weights(sample_weights, selected, round_id, mode='sample',
                  factor='B', extra=False, branch='main'):
    if (mode == 'uniform-transfer-rounds' and round_id in B_TRANSFER_ROUNDS
            and factor == 'B' and not extra and branch == 'main'):
        return {k: 1. / len(selected) for k in selected}
    return sample_weights
