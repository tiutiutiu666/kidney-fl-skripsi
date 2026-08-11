"""
experiments/distribusi_partisi.py

Menampilkan dan menyimpan distribusi data hasil partisi Dirichlet untuk setiap
nilai alpha, guna memperlihatkan dua bentuk non-IID yang disyaratkan:

  1. Label distribution skew : proporsi kelas berbeda antar klien
  2. Quantity skew           : jumlah data berbeda antar klien

Keluaran:
  - Tabel per alpha di terminal
  - results/final/distribusi_partisi.csv
  - results/final/distribusi_partisi.png   (gambar untuk Bab IV)

Cara pakai:
    python experiments/distribusi_partisi.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from config import (ALPHAS, RESULT_DIR, TEST_SPLIT, SEED,
                    NUM_CLIENTS, KELAS_LIST)
from src.data.data_loader import load_dataset
from src.data.grouping import compute_group_labels, group_aware_holdout
from src.data.partitioner import dirichlet_partition

FINAL_DIR = os.path.join(RESULT_DIR, "final")
os.makedirs(FINAL_DIR, exist_ok=True)

WARNA = ["#4C72B0", "#DD8452", "#C44E52", "#55A868"]   # Cyst/Normal/Stone/Tumor


def hitung_distribusi(idx_tv, all_labels, label_skenario):
    """Kembalikan list baris {skenario, alpha, klien, kelas..., total}."""
    baris = []
    for alpha in ALPHAS:
        partisi = dirichlet_partition(idx_tv, all_labels, alpha)

        print(f"\n  α = {alpha}")
        print(f"  {'Klien':<8}{'Total':>8}   " +
              "".join(f"{k:>16}" for k in KELAS_LIST))

        jumlah_klien = [len(p) for p in partisi]
        for k, part in enumerate(partisi):
            n = len(part)
            per_kelas = [int((all_labels[part] == c).sum())
                         for c in range(len(KELAS_LIST))]
            teks = "".join(
                f"{v:>9,} ({v/n*100 if n else 0:4.1f}%)" for v in per_kelas)
            print(f"  {'Klien '+str(k+1):<8}{n:>8,}   {teks}")

            baris.append({
                "skenario": label_skenario,
                "alpha":    alpha,
                "klien":    k + 1,
                **{KELAS_LIST[c]: per_kelas[c] for c in range(len(KELAS_LIST))},
                "total":    n,
            })

        # Indikator quantity skew: rasio klien terbesar terhadap terkecil
        terkecil = max(min(jumlah_klien), 1)
        print(f"  {'':<8}{'':>8}   Quantity skew — terbesar/terkecil = "
              f"{max(jumlah_klien)/terkecil:.2f}×")
    return baris


def gambar(rows_per_skenario, nama_file):
    """Stacked bar: komposisi kelas tiap klien, satu kolom per alpha."""
    n_sken = len(rows_per_skenario)
    fig, axes = plt.subplots(n_sken, len(ALPHAS),
                             figsize=(4.0 * len(ALPHAS), 4.2 * n_sken),
                             squeeze=False)

    for r, (judul, rows) in enumerate(rows_per_skenario):
        # Sumbu-y disamakan antar alpha agar quantity skew terlihat jujur
        y_max = max(b["total"] for b in rows) * 1.18

        for c, alpha in enumerate(ALPHAS):
            ax   = axes[r][c]
            data = [b for b in rows if b["alpha"] == alpha]
            x    = np.arange(1, NUM_CLIENTS + 1)
            dasar = np.zeros(NUM_CLIENTS)

            for i, kelas in enumerate(KELAS_LIST):
                nilai = np.array([b[kelas] for b in data], dtype=float)
                ax.bar(x, nilai, bottom=dasar, color=WARNA[i],
                       label=kelas if (r == 0 and c == 0) else None,
                       edgecolor="white", linewidth=0.6, width=0.62)
                dasar += nilai

            for i, b in enumerate(data):
                ax.text(x[i], b["total"] + y_max * 0.02, f"{b['total']:,}",
                        ha="center", va="bottom", fontsize=8.5,
                        fontweight="bold", color="#333")

            ax.set_title(f"α = {alpha}", fontsize=12, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([f"Klien {i}" for i in x], fontsize=9)
            ax.set_ylim(0, y_max)
            ax.grid(axis="y", alpha=0.25, linestyle="--")
            ax.set_axisbelow(True)
            if c == 0:
                ax.set_ylabel(f"{judul}\n\nJumlah citra", fontsize=10,
                              fontweight="bold")
            else:
                ax.tick_params(labelleft=False)

    fig.suptitle("Distribusi Data Antar Klien Hasil Partisi Dirichlet",
                 fontsize=15, fontweight="bold")
    fig.legend(loc="lower center", ncol=len(KELAS_LIST), fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.035, 1, 0.96])
    plt.savefig(nama_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nGambar disimpan: {nama_file}")


if __name__ == "__main__":
    all_paths, all_labels = load_dataset()
    idx_all = np.arange(len(all_paths))

    semua_baris = []
    per_skenario = []

    # ── Skenario I — pembagian acak per citra ────────────────────────────────
    print("\n" + "=" * 78)
    print("SKENARIO I — PEMBAGIAN ACAK PER CITRA")
    print("=" * 78)
    idx_tv_1, _ = train_test_split(idx_all, test_size=TEST_SPLIT,
                                   stratify=all_labels, random_state=SEED)
    baris_1 = hitung_distribusi(idx_tv_1, all_labels, "I - acak per citra")
    semua_baris += baris_1
    per_skenario.append(("Skenario I\n(acak per citra)", baris_1))

    # ── Skenario II — pembagian group-aware ──────────────────────────────────
    print("\n" + "=" * 78)
    print("SKENARIO II — PEMBAGIAN GROUP-AWARE")
    print("=" * 78)
    groups = compute_group_labels(all_paths)
    idx_tv_2, _ = group_aware_holdout(idx_all, all_labels, groups,
                                      TEST_SPLIT, SEED)
    baris_2 = hitung_distribusi(idx_tv_2, all_labels, "II - group-aware")
    semua_baris += baris_2
    per_skenario.append(("Skenario II\n(group-aware)", baris_2))

    # ── Simpan CSV ───────────────────────────────────────────────────────────
    out_csv = os.path.join(FINAL_DIR, "distribusi_partisi.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["skenario", "alpha", "klien", *KELAS_LIST, "total"])
        writer.writeheader()
        writer.writerows(semua_baris)
    print(f"\nTabel disimpan: {out_csv}")

    gambar(per_skenario, os.path.join(FINAL_DIR, "distribusi_partisi.png"))
