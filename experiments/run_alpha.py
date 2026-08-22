import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from sklearn.model_selection import train_test_split

from config import (ALPHAS, RESULT_DIR, TEST_SPLIT, SEED, NUM_CLIENTS,
                    GROUP_AWARE_SPLIT, dir_alpha)
from src.seeding import set_seed
from src.data.data_loader import load_dataset
from src.data.grouping import compute_group_labels, group_aware_holdout
from src.data.partitioner import dirichlet_partition
from src.federated.server import run_simulation

# ── GANTI untuk memilih alpha ────────────────────────────────────────────
ALPHA_TARGET = 0.3   # pilih: 0.1 | 0.3 | 0.5 | 0.7 | 1.0
# ─────────────────────────────────────────────────────────────────────────────

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_DIR_ALPHA = dir_alpha(ALPHA_TARGET)
os.makedirs(RESULT_DIR_ALPHA, exist_ok=True)

set_seed()
print(f"Device      : {DEVICE}")
print(f"Alpha target: {ALPHA_TARGET}")
print(f"Hasil di    : {RESULT_DIR_ALPHA}")

# Load dataset
all_paths, all_labels = load_dataset()

# Pisah test set (dikunci)
idx_all = np.arange(len(all_paths))

# GROUP_AWARE_SPLIT: cegah irisan near-identical dari scan yang sama tersebar
# ke train dan test sekaligus (lihat src/data/grouping.py).
groups = None
if GROUP_AWARE_SPLIT:
    print("Group-aware splitting AKTIF — mengelompokkan citra serupa...")
    groups = compute_group_labels(all_paths)
    idx_tv, idx_test = group_aware_holdout(
        idx_all, all_labels, groups, TEST_SPLIT, SEED)
else:
    idx_tv, idx_test = train_test_split(
        idx_all, test_size=TEST_SPLIT,
        stratify=all_labels, random_state=SEED
    )

# Simpan idx_test agar konsisten di semua eksperimen.
# Nama file dibedakan per mode split — lihat penjelasan di main.py.
suffix        = "_group" if GROUP_AWARE_SPLIT else ""
test_idx_path = os.path.join(RESULT_DIR, f"test_indices{suffix}.npy")
if not os.path.exists(test_idx_path):
    os.makedirs(RESULT_DIR, exist_ok=True)
    np.save(test_idx_path, idx_test)
    print(f"Test indices disimpan: {test_idx_path}")
else:
    idx_test = np.load(test_idx_path)
    print(f"Test indices dimuat dari: {test_idx_path}")

overlap = np.intersect1d(idx_tv, idx_test)
if len(overlap) > 0:
    raise RuntimeError(
        f"Test set tumpang tindih dengan train+val sebanyak {len(overlap):,} "
        f"indeks!\nFile '{test_idx_path}' kemungkinan berasal dari skenario "
        f"split yang berbeda. Hapus/arsipkan file tersebut lalu jalankan ulang."
    )

# Partisi Dirichlet
partitions = dirichlet_partition(idx_tv, all_labels, ALPHA_TARGET)
print(f"\nDistribusi klien (α={ALPHA_TARGET}):")
from collections import Counter
for k, part in enumerate(partitions):
    dist = Counter(all_labels[part])
    print(f"  Klien {k+1}: {len(part):,} gambar | {dict(dist)}")

# Jalankan simulasi FL
print(f"\nMemulai simulasi FL untuk α = {ALPHA_TARGET}...")
run_simulation(
    all_paths    = all_paths,
    all_labels   = all_labels,
    partitions   = partitions,
    idx_tv       = idx_tv,
    result_dir   = RESULT_DIR_ALPHA,
    device       = DEVICE,
    groups       = groups
)

print(f"\nEksperimen α = {ALPHA_TARGET} selesai.")