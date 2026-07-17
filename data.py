import numpy as np


def build_dataset(p: int):
    """All ordered pairs (a, b) in [0, p) x [0, p), tokenized as [a, b, "="] with
    "=" given token id p, labeled with (a + b) mod p."""
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    a, b = a.ravel(), b.ravel()
    labels = (a + b) % p
    eq = np.full_like(a, p)
    tokens = np.stack([a, b, eq], axis=1).astype(np.int32)
    return tokens, labels.astype(np.int32)


def split(n: int, seed: int, frac_train: float):
    """Deterministic, seeded train/test index split."""
    perm = np.random.default_rng(seed).permutation(n)
    n_train = int(frac_train * n)
    return perm[:n_train], perm[n_train:]
