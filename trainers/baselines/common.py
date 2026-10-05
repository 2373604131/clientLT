"""Small, explicit aggregation/state operations shared by external baselines."""
import torch


def trainable_state(model):
    return {k: p.detach().cpu().clone() for k, p in model.named_parameters() if p.requires_grad}


def load_trainable(model, state):
    parameters = dict(model.named_parameters())
    expected = {k for k, p in parameters.items() if p.requires_grad}
    if set(state) != expected:
        raise ValueError('Trainable state keys differ')
    with torch.no_grad():
        for key, value in state.items():
            parameters[key].copy_(value)


def weighted_average(states, client_ids, sizes):
    if not client_ids or len(client_ids) != len(sizes) or any(x <= 0 for x in sizes):
        raise ValueError('Invalid client weights')
    total = float(sum(sizes))
    result = {}
    for client, size in zip(client_ids, sizes):
        if result and set(states[client]) != set(result):
            raise ValueError('Mismatched client parameter keys')
        for key, value in states[client].items():
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError('Nonfinite client update')
            if key not in result:
                result[key] = value * (size / total)
            else:
                result[key].add_(value, alpha=size / total)
    return result
