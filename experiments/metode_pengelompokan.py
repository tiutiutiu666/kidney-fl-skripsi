"""
experiments/metode_pengelompokan.py

Menjelaskan dan memvalidasi metode pengelompokan citra (src/data/grouping.py)
secara visual:

  Panel A : contoh citra dari beberapa grup — memperlihatkan bahwa satu grup
            memang berisi irisan berurutan dari pemindaian yang sama
  Panel B : sebaran ukuran grup
  Panel C : sebaran korelasi pasangan SEGRUP vs BEDA GRUP — memperlihatkan
            bahwa ambang 0,99 memisahkan keduanya dengan sangat tegas

Keluaran: results/final/metode_pengelompokan.png

Cara pakai:
    python experiments/metode_pengelompokan.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from config import RESULT_DIR, KELAS_LIST, GROUP_DESC_RES, GROUP_SIM_THRESHOLD
from src.data.data_loader import load_dataset
from src.data.grouping import compute_group_labels, _build_descriptors

N_GRUP_CONTOH  = 3     # berapa grup ditampilkan pada Panel A
N_CITRA_CONTOH = 5     # berapa citra per grup
N_SAMPEL_PAIR  = 20000 # jumlah pasangan yang disampel untuk Panel C

BIRU  = "#4C72B0"
MERAH = "#C44E52"


def main():
    all_paths, all_labels = load_dataset()
    groups = compute_group_labels(all_paths)
    uniq, sizes = np.unique(groups, return_counts=True)

    print("\nMenghitung deskriptor untuk analisis korelasi...")
    desc = _build_descriptors(all_paths, GROUP_DESC_RES)

    rng = np.random.default_rng(0)

    # ── Sampel pasangan SEGRUP dan BEDA GRUP ────────────────────────────────
    besar = uniq[sizes >= 2]
    korel_segrup = []
    while len(korel_segrup) < N_SAMPEL_PAIR:
        g = rng.choice(besar)
        anggota = np.flatnonzero(groups == g)
        i, j = rng.choice(anggota, size=2, replace=False)
        korel_segrup.append(float(desc[i] @ desc[j]))
    korel_segrup = np.array(korel_segrup)

    korel_beda = []
    while len(korel_beda) < N_SAMPEL_PAIR:
        i, j = rng.integers(0, len(all_paths), size=2)
        if groups[i] != groups[j]:
            korel_beda.append(float(desc[i] @ desc[j]))
    korel_beda = np.array(korel_beda)

    # ── Gambar ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs  = fig.add_gridspec(2, 2, height_ratios=[1.15, 1],
                           hspace=0.32, wspace=0.22)
    fig.suptitle("Metode Pengelompokan Citra Near-Identical",
                 fontsize=15.5, fontweight="bold", y=0.965)

    # Panel A — contoh grup
    axA = fig.add_subplot(gs[0, :]); axA.axis("off")
    axA.set_title(f"A. Contoh isi grup — tiap baris satu grup "
                  f"(korelasi ≥ {GROUP_SIM_THRESHOLD})",
                  fontsize=11.5, fontweight="bold", pad=14)

    kandidat = uniq[sizes >= N_CITRA_CONTOH]
    pilihan  = rng.choice(kandidat, size=N_GRUP_CONTOH, replace=False)

    for r, g in enumerate(pilihan):
        anggota = np.flatnonzero(groups == g)
        ambil   = anggota[np.linspace(0, len(anggota) - 1,
                                      N_CITRA_CONTOH).astype(int)]
        for c, idx in enumerate(ambil):
            ax = fig.add_axes([0.135 + c * 0.115,
                               0.735 - r * 0.088, 0.105, 0.082])
            ax.imshow(Image.open(all_paths[idx]).convert("L"), cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(BIRU); s.set_linewidth(2)
            if c == 0:
                ax.set_ylabel(f"Grup {g}\n{len(anggota)} citra\n"
                              f"{KELAS_LIST[all_labels[idx]]}",
                              fontsize=8.5, fontweight="bold", color=BIRU,
                              rotation=0, ha="right", va="center",
                              labelpad=42)

    # Panel B — sebaran ukuran grup
    axB = fig.add_subplot(gs[1, 0])
    axB.hist(sizes, bins=30, color=BIRU, edgecolor="white", alpha=0.85)
    axB.axvline(sizes.mean(), color=MERAH, linestyle="--", linewidth=2,
                label=f"rata-rata {sizes.mean():.1f} citra")
    axB.set_title(f"B. Sebaran ukuran grup ({len(uniq)} grup dari "
                  f"{len(all_paths):,} citra)",
                  fontsize=11, fontweight="bold")
    axB.set_xlabel("Jumlah citra dalam satu grup")
    axB.set_ylabel("Banyak grup")
    axB.legend(fontsize=9)
    axB.grid(alpha=0.25, linestyle="--"); axB.set_axisbelow(True)

    # Panel C — sebaran korelasi
    axC = fig.add_subplot(gs[1, 1])
    bins = np.linspace(-0.2, 1.0, 90)
    axC.hist(korel_beda, bins=bins, color=MERAH, alpha=0.72,
             label=f"Beda grup (median {np.median(korel_beda):.2f})")
    axC.hist(korel_segrup, bins=bins, color=BIRU, alpha=0.78,
             label=f"Segrup (median {np.median(korel_segrup):.4f})")
    axC.axvline(GROUP_SIM_THRESHOLD, color="black", linestyle="--",
                linewidth=2, label=f"ambang {GROUP_SIM_THRESHOLD}")
    axC.set_title("C. Korelasi pasangan citra — segrup vs beda grup",
                  fontsize=11, fontweight="bold")
    axC.set_xlabel("Korelasi antar deskriptor")
    axC.set_ylabel("Banyak pasangan")
    axC.set_yscale("log")
    axC.legend(fontsize=8.5, loc="upper left")
    axC.grid(alpha=0.25, linestyle="--"); axC.set_axisbelow(True)

    out = os.path.join(RESULT_DIR, "final", "metode_pengelompokan.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("\n" + "=" * 62)
    print("RINGKASAN")
    print("=" * 62)
    print(f"  Jumlah grup            : {len(uniq)}")
    print(f"  Ukuran grup            : min {sizes.min()}, "
          f"median {int(np.median(sizes))}, max {sizes.max()}, "
          f"rata-rata {sizes.mean():.1f}")
    print(f"  Grup berisi 1 citra    : {(sizes == 1).sum()}")
    print(f"\n  Korelasi SEGRUP        : median {np.median(korel_segrup):.4f} "
          f"(persentil 5 = {np.percentile(korel_segrup, 5):.4f})")
    print(f"  Korelasi BEDA GRUP     : median {np.median(korel_beda):.4f} "
          f"(persentil 95 = {np.percentile(korel_beda, 95):.4f})")
    print(f"\nGambar disimpan: {out}")


if __name__ == "__main__":
    main()
