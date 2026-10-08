# python version 3.7.1
# -*- coding: utf-8 -*-
import numpy as np

MIN_CLIENT_SAMPLES = 10


def client_sizes(dict_users, num_users):
    return [len(dict_users.get(i, set())) for i in range(num_users)]


def has_insufficient_client(dict_users, num_users, min_samples=MIN_CLIENT_SAMPLES):
    return any(len(dict_users.get(i, set())) < min_samples for i in range(num_users))


def normalize_dict_users(dict_users, num_users):
    return {i: set(dict_users.get(i, set())) for i in range(num_users)}


def _validate_min_samples_feasible(n_samples, num_users, min_samples=MIN_CLIENT_SAMPLES):
    required = num_users * min_samples
    if required > n_samples:
        raise ValueError(
            f"Cannot assign min_samples={min_samples} to each of {num_users} clients: "
            f"need at least {required} samples but only {n_samples} available"
        )


def ensure_min_client_samples(dict_users, num_users, min_samples=MIN_CLIENT_SAMPLES, seed=0):
    """Move samples between clients until every client has at least min_samples."""
    _validate_min_samples_feasible(
        sum(len(dict_users.get(i, set())) for i in range(num_users)),
        num_users,
        min_samples,
    )
    rng = np.random.RandomState(seed)
    dict_users = normalize_dict_users(dict_users, num_users)
    padded_clients = []

    def needy_clients():
        return [i for i in range(num_users) if len(dict_users[i]) < min_samples]

    def donor_clients(exclude):
        donors = [
            i for i in range(num_users)
            if i not in exclude and len(dict_users[i]) > min_samples
        ]
        if donors:
            return donors
        return [
            i for i in range(num_users)
            if i not in exclude and len(dict_users[i]) > 1
        ]

    max_iters = num_users * min_samples * 20
    for _ in range(max_iters):
        needy = needy_clients()
        if not needy:
            break
        target = needy[0]
        donors = donor_clients(exclude={target})
        if not donors:
            raise RuntimeError(
                f"Cannot enforce min_samples={min_samples}: "
                f"clients_sizes={client_sizes(dict_users, num_users)}"
            )
        donor = int(rng.choice(donors))
        idx = int(rng.choice(list(dict_users[donor])))
        dict_users[donor].remove(idx)
        dict_users[target].add(idx)
        if len(dict_users[target]) == min_samples and target not in padded_clients:
            padded_clients.append(target)
    else:
        raise RuntimeError(
            f"ensure_min_client_samples exceeded iteration limit; "
            f"clients_sizes={client_sizes(dict_users, num_users)}"
        )

    if padded_clients:
        print(
            f"ensure_min_client_samples: padded clients {padded_clients} "
            f"to min_samples={min_samples}"
        )
    return dict_users


def ensure_min_pool_samples(dict_users, num_users, pool, min_samples=MIN_CLIENT_SAMPLES, seed=0):
    """Add samples from pool (with replacement) until every client has min_samples."""
    rng = np.random.RandomState(seed)
    dict_users = normalize_dict_users(dict_users, num_users)
    pool = np.asarray(pool)
    if pool.size == 0:
        raise ValueError("pool must be non-empty")
    padded_clients = []
    for client_id in range(num_users):
        if len(dict_users[client_id]) >= min_samples:
            continue
        padded_clients.append(client_id)
        while len(dict_users[client_id]) < min_samples:
            dict_users[client_id].add(int(rng.choice(pool)))
    if padded_clients:
        print(
            f"ensure_min_pool_samples: padded clients {padded_clients} "
            f"to min_samples={min_samples}"
        )
    return dict_users


def _iid_sampling_once(n_train, num_users, seed):
    np.random.seed(seed)
    num_items = int(n_train / num_users)
    dict_users, all_idxs = {}, [i for i in range(n_train)]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def iid_sampling(n_train, num_users, seed):
    dict_users = _iid_sampling_once(n_train, num_users, seed)
    return normalize_dict_users(dict_users, num_users)


def _non_iid_dirichlet_sampling_once(y_train, num_classes, p, num_users, seed, alpha_dirichlet=100):
    np.random.seed(seed)
    p = 1
    Phi = np.random.binomial(1, p, size=(num_users, num_classes))
    n_classes_per_client = np.sum(Phi, axis=1)
    while np.min(n_classes_per_client) == 0:
        invalid_idx = np.where(n_classes_per_client == 0)[0]
        Phi[invalid_idx] = np.random.binomial(1, p, size=(len(invalid_idx), num_classes))
        n_classes_per_client = np.sum(Phi, axis=1)
    Psi = [list(np.where(Phi[:, j] == 1)[0]) for j in range(num_classes)]
    num_clients_per_class = np.array([len(x) for x in Psi])
    dict_users = {}
    for class_i in range(num_classes):
        all_idxs = np.where(y_train == class_i)[0]
        if len(all_idxs) == 0:
            continue
        p_dirichlet = np.random.dirichlet([alpha_dirichlet] * num_clients_per_class[class_i])
        assignment = np.random.choice(Psi[class_i], size=len(all_idxs), p=p_dirichlet.tolist())

        for client_k in Psi[class_i]:
            assigned = set(all_idxs[(assignment == client_k)])
            if client_k in dict_users:
                dict_users[client_k] = set(dict_users[client_k] | assigned)
            else:
                dict_users[client_k] = assigned
    return dict_users


def non_iid_dirichlet_sampling(
    y_train, num_classes, p, num_users, seed, alpha_dirichlet=100
):
    dict_users = _non_iid_dirichlet_sampling_once(
        y_train, num_classes, p, num_users, seed, alpha_dirichlet
    )
    return normalize_dict_users(dict_users, num_users)
