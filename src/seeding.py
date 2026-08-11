"""
src/seeding.py

Penetapan seed global agar eksperimen dapat direproduksi.

LATAR BELAKANG
--------------
Sebelumnya SEED pada config.py hanya dipakai untuk pembagian data (sklearn),
sementara inisialisasi bobot classifier head, pengacakan batch pada DataLoader,
dan augmentasi acak tidak pernah di-seed. Akibatnya dua run pada split yang
persis sama dapat menghasilkan selisih beberapa poin pada metrik uji, sehingga
perbedaan kecil antar skenario tidak dapat dibedakan dari derau.
"""
import os
import random

import numpy as np
import torch

from config import SEED


def set_seed(seed: int = SEED, deterministic: bool = True):
    """
    Set seed untuk random, numpy, dan torch (CPU + semua GPU).

    deterministic=True memaksa cuDNN memakai algoritma deterministik dan
    mematikan autotuner benchmark. Ini membuat hasil konvolusi identik antar
    run dengan biaya sedikit penurunan kecepatan — pertukaran yang sepadan
    untuk eksperimen skripsi yang harus dapat direproduksi.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    return seed
