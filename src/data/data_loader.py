import os
import numpy as np
from config import DATASET_PATH, KELAS_LIST


def load_dataset():
    """
    Memuat semua path gambar dan labelnya dari folder dataset.
    Return: all_paths (np.array), all_labels (np.array)
    """
    label_map  = {k: i for i, k in enumerate(KELAS_LIST)}
    all_paths, all_labels = [], []

    for kelas in KELAS_LIST:
        folder = os.path.join(DATASET_PATH, kelas)
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder tidak ditemukan: {folder}")
        # sorted() WAJIB: os.listdir() mengembalikan urutan yang bergantung
        # filesystem (arbitrer di ext4/Linux, alfabetis di NTFS/Windows).
        # Karena seluruh pembagian train/val/test ditentukan oleh POSISI indeks
        # pada array ini, urutan yang tidak deterministik membuat split berbeda
        # di setiap mesin — eksperimen jadi tidak reproducible dan cache label
        # grup tidak valid bila dipindah antar mesin.
        files = sorted(f for f in os.listdir(folder)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        for f in files:
            all_paths.append(os.path.join(folder, f))
            all_labels.append(label_map[kelas])

    all_paths  = np.array(all_paths)
    all_labels = np.array(all_labels)

    print(f"Dataset berhasil dimuat: {len(all_paths):,} gambar")
    for kelas in KELAS_LIST:
        n = (all_labels == label_map[kelas]).sum()
        print(f"  {kelas:10s}: {n:,}")

    return all_paths, all_labels