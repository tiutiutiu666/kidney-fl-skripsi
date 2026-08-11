import numpy as np
from config import NUM_CLIENTS, SEED


def dirichlet_partition(indices, labels, alpha, num_clients=NUM_CLIENTS, seed=SEED):
    """
    Membagi indices ke num_clients klien menggunakan distribusi Dirichlet.
    Return: list of np.array, masing-masing berisi indeks tiap klien.
    """
    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(num_clients)]

    for kelas in np.unique(labels):
        idx_kelas = indices[labels[indices] == kelas].copy()
        rng.shuffle(idx_kelas)

        proporsi     = rng.dirichlet([alpha] * num_clients)
        batas        = np.round(np.cumsum(proporsi) * len(idx_kelas)).astype(int)
        batas[-1]    = len(idx_kelas)

        prev = 0
        for k in range(num_clients):
            client_indices[k].extend(idx_kelas[prev:batas[k]].tolist())
            prev = batas[k]

    for k in range(num_clients):
        arr = np.array(client_indices[k])
        rng.shuffle(arr)
        client_indices[k] = arr

    return client_indices