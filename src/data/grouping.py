"""
src/data/grouping.py

Pengelompokan citra yang near-identical (diduga berasal dari pasien/scan yang
sama) agar pembagian data dapat dilakukan secara GROUP-AWARE.

LATAR BELAKANG
--------------
Dataset CT Kidney terdiri dari banyak irisan (slice) berurutan dari scan yang
sama. Irisan-irisan ini hampir identik secara visual. Jika data dibagi secara
acak per-CITRA, irisan dari scan yang sama akan tersebar di train, val, dan
test sekaligus — sehingga model cukup "mengingat" anatomi pasien tersebut dan
metrik uji menjadi terlalu optimistis (mendekati 100%).

Modul ini mengelompokkan citra berdasarkan kemiripan konten, lalu menghasilkan
label grup yang bisa dipakai oleh GroupShuffleSplit / StratifiedGroupKFold
sehingga seluruh citra dari satu grup selalu berada di sisi split yang sama.

METODE
------
1. Setiap citra diubah menjadi deskriptor: grayscale, resize ke R×R,
   dikurangi mean, lalu dinormalisasi ke unit-norm. Dot product antar
   deskriptor = korelasi Pearson antar citra.
2. Dua citra dianggap satu grup jika korelasi >= threshold.
3. Grup final = connected components dari graf kemiripan tersebut.
"""
import os
import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from config import RESULT_DIR, GROUP_DESC_RES, GROUP_SIM_THRESHOLD


def _build_descriptors(all_paths, resolution):
    """Deskriptor ternormalisasi (zero-mean, unit-norm) untuk tiap citra."""
    desc = np.zeros((len(all_paths), resolution * resolution), dtype=np.float32)
    for i, p in enumerate(all_paths):
        im = Image.open(p).convert("L").resize((resolution, resolution),
                                               Image.BILINEAR)
        d  = np.asarray(im, dtype=np.float32).ravel()
        d -= d.mean()
        n  = np.linalg.norm(d)
        desc[i] = d / n if n > 1e-6 else d
        if (i + 1) % 3000 == 0:
            print(f"    deskriptor {i+1:,}/{len(all_paths):,}")
    return desc


def group_aware_holdout(idx, labels, groups, test_size, seed):
    """
    Pisahkan `idx` menjadi (sisa, holdout) secara group-aware DAN stratified.

    GroupShuffleSplit mengabaikan label sama sekali, sehingga komposisi kelas
    pada holdout bisa melenceng jauh dari populasi (mis. Cyst 29,8% -> 42%).
    StratifiedGroupKFold menjaga proporsi kelas sekaligus memastikan satu grup
    tidak pernah terbelah antar split. n_splits dipilih dari test_size
    (0.20 -> 5 fold), lalu fold pertama diambil sebagai holdout.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    n_splits = int(round(1.0 / test_size))
    sgkf     = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=seed)
    rest_rel, hold_rel = next(sgkf.split(idx, labels[idx], groups=groups[idx]))
    return idx[rest_rel], idx[hold_rel]


def compute_group_labels(all_paths, resolution=GROUP_DESC_RES,
                         threshold=GROUP_SIM_THRESHOLD,
                         cache_path=None, force=False):
    """
    Kembalikan array label grup (shape = [len(all_paths)]).
    Citra dengan label grup sama = near-identical (diduga satu pasien/scan).

    Hasil di-cache ke disk karena perhitungannya O(N^2) dan memakan waktu.
    """
    if cache_path is None:
        cache_path = os.path.join(RESULT_DIR, "group_labels.npy")

    if os.path.exists(cache_path) and not force:
        labels = np.load(cache_path)
        if len(labels) == len(all_paths):
            print(f"  Label grup dimuat dari cache: {cache_path} "
                  f"({labels.max()+1:,} grup)")
            return labels
        print("  Cache label grup tidak cocok ukurannya — hitung ulang.")

    print(f"  Menghitung label grup (resolusi {resolution}×{resolution}, "
          f"threshold korelasi {threshold})...")
    desc = _build_descriptors(all_paths, resolution)

    N    = len(all_paths)
    rows, cols = [], []
    for s in range(0, N, 512):
        sim  = desc[s:s + 512] @ desc.T
        r, c = np.nonzero(sim >= threshold)
        rows.append(r + s)
        cols.append(c)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)

    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)),
                       shape=(N, N))
    n_comp, labels = connected_components(graph, directed=False)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, labels)

    _, sizes = np.unique(labels, return_counts=True)
    print(f"  {N:,} citra -> {n_comp:,} grup "
          f"(rata-rata {N/n_comp:.1f} citra/grup, terbesar {sizes.max():,})")
    print(f"  Label grup disimpan: {cache_path}")
    return labels
