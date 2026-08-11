# Klasifikasi Multi-Kelas CT Scan Ginjal dengan Federated Learning (FedProx), EfficientNet-B0, dan Grad-CAM

Implementasi skripsi S1 Informatika — Fakultas Teknologi Informasi dan Sains Data,
Universitas Sebelas Maret.

**Farrelly Theo Ariela (L0122061)** · Pembimbing: Winarno, S.Si., M.Eng.

---

## Ringkasan

Sistem ini mensimulasikan **Federated Learning** dengan 3 klien (rumah sakit virtual)
pada satu perangkat komputasi. Setiap klien melatih **EfficientNet-B0** secara lokal
menggunakan fungsi objektif **FedProx**, lalu hanya mengirimkan bobot model ke server
untuk diagregasi — data mentah tidak pernah berpindah. Kondisi data tidak seimbang
(non-IID) antar klien disimulasikan dengan **distribusi Dirichlet** pada lima tingkat
heterogenitas. Model global terbaik kemudian diinterpretasi menggunakan **Grad-CAM**.

| Komponen | Nilai |
| --- | --- |
| Arsitektur | EfficientNet-B0 (pretrained ImageNet, full fine-tuning) |
| Agregasi | FedProx (μ = 0,1) |
| Jumlah klien | 3 |
| Non-IID | Distribusi Dirichlet, α ∈ {0,1 · 0,3 · 0,5 · 0,7 · 1,0} |
| Validasi | Stratified 5-Fold Cross-Validation |
| Pembagian data | 64% latih · 16% validasi · 20% uji |
| Optimizer | Adam, lr = 1×10⁻⁴, batch size 32 |
| Ronde maksimum | 100 (early stopping, patience 15) |
| Epoch lokal per ronde | 5 |
| XAI | Grad-CAM (`pytorch-grad-cam`) |
| Metrik | Accuracy, Precision, Recall, F1 — **macro-average** |

---

## Dataset

**CT Kidney Dataset: Normal–Cyst–Tumor–and–Stone** (Kaggle) — 12.446 citra CT 2D
berformat `.jpg`, terbagi atas 4 kelas.

| Kelas | Jumlah | Proporsi |
| --- | ---: | ---: |
| Normal | 5.077 | 40,8% |
| Cyst | 3.709 | 29,8% |
| Tumor | 2.283 | 18,3% |
| Stone | 1.377 | 11,1% |

> **Dataset tidak disertakan dalam repositori ini** karena ukurannya ±1,5 GB.
> Unduh dari Kaggle, lalu tempatkan sehingga strukturnya menjadi:
>
> ```
> data/raw/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/
>         CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/
>             Cyst/  Normal/  Stone/  Tumor/
> ```
>
> Path ini dapat diubah melalui `DATASET_PATH` di `config.py`.

---

## Instalasi

Membutuhkan Python 3.10+ dan GPU dengan CUDA (dapat berjalan di CPU, tetapi jauh
lebih lambat).

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Cara Menjalankan

### 1. Baseline terpusat (pembanding)

Melatih EfficientNet-B0 secara konvensional tanpa Federated Learning.

```bash
python experiments/baseline/run_baseline.py
```

### 2. Satu nilai α saja

Ubah `ALPHA_TARGET` di dalam berkas, lalu jalankan. Cocok untuk menjalankan
eksperimen bertahap.

```bash
python experiments/run_alpha.py
```

### 3. Seluruh eksperimen (5 α × 5 fold = 25 eksperimen)

```bash
python main.py
```

Mencakup pelatihan seluruh skenario, pemilihan model terbaik, evaluasi test set,
visualisasi Grad-CAM, kurva pelatihan, dan tabel perbandingan antar α.

### 4. Memantau dan merangkum

```bash
python experiments/check_status.py       # progres tiap α dan fold
python experiments/compare_results.py    # tabel perbandingan hasil
```

### 5. Prediksi satu citra

Ubah `IMAGE_PATH` di dalam berkas, lalu jalankan. Menampilkan kelas prediksi,
probabilitas tiap kelas, dan overlay Grad-CAM.

```bash
python predict.py
```

> **Catatan:** pelatihan penuh memakan waktu lama. Gunakan `tmux` atau `nohup` agar
> proses tetap berjalan meskipun koneksi terputus. Checkpoint tersimpan otomatis
> setiap 10 ronde dan pelatihan dapat dilanjutkan dari titik terakhir; fold yang
> sudah selesai akan dilewati secara otomatis.

---

## Dua Skenario Pembagian Data

Repositori ini mendukung dua skenario yang dikendalikan oleh satu flag di `config.py`:

```python
GROUP_AWARE_SPLIT = False   # Skenario I
GROUP_AWARE_SPLIT = True    # Skenario II
```

**Skenario I — pembagian acak per citra.** Pendekatan konvensional yang umum dipakai
pada penelitian terdahulu dengan dataset ini.

**Skenario II — pembagian group-aware.** Citra dikelompokkan berdasarkan kemiripan
konten terlebih dahulu, kemudian pembagian dilakukan per kelompok menggunakan
`StratifiedGroupKFold` sehingga satu kelompok tidak pernah terbelah antara data latih,
validasi, dan uji.

> Mengubah flag ini mengubah komposisi seluruh pembagian data, sehingga hasil dari
> kedua skenario tidak dapat dibandingkan langsung tanpa menjalankan ulang eksperimen.
> Berkas indeks data uji disimpan terpisah per skenario, dan program akan berhenti
> dengan galat apabila mendeteksi data uji bertumpang tindih dengan data latih.

---

## Temuan: Kebocoran Data pada Pembagian Acak

Pada Skenario I, seluruh metrik evaluasi mencapai nilai mendekati sempurna — baseline
terpusat memperoleh accuracy, precision, recall, dan F1 sebesar **100%**. Investigasi
menunjukkan bahwa hal ini disebabkan **kebocoran data pada tingkat irisan/pasien**,
bukan kesalahan implementasi.

Pemeriksaan yang dilakukan:

| Pemeriksaan | Hasil |
| --- | --- |
| Tumpang tindih indeks latih/validasi/uji | 0 — bersih |
| Duplikat byte-identik (MD5) pada data uji | 152 dari 2.490 (6,10%) |
| Jumlah kelompok citra near-identical | **250** dari 12.446 citra (±49,8 citra/kelompok) |
| Citra uji yang kelompoknya juga ada di data latih | 2.484 dari 2.490 (**99,76%**) |
| Korelasi tetangga terdekat vs pasangan acak | 0,9985 vs 0,38 |

Penyebabnya adalah karakteristik dataset: satu pemindaian CT menghasilkan banyak irisan
berurutan yang nyaris identik. Pembagian acak per citra menyebarkan irisan dari
pemindaian yang sama ke data latih sekaligus data uji, sehingga model cukup mengenali
anatomi pasien yang sudah pernah dilihat.

Setelah pembagian group-aware diterapkan, tingkat kebocoran turun menjadi **0,00%** dan
performa baseline turun dari 100% menjadi sekitar **78–81%** accuracy
(macro F1 ±0,75–0,78) — dengan penurunan terbesar pada kelas minoritas *Stone*.

Bukti visual tersedia pada `results/final/bukti_leakage.png`, menampilkan citra uji
berdampingan dengan citra latih termiripnya (korelasi 1,00000).

---

## Struktur Direktori

```
config.py                        Seluruh hyperparameter dan path
main.py                          Orkestrasi 25 eksperimen + evaluasi akhir
predict.py                       Prediksi dan Grad-CAM untuk satu citra

src/
  seeding.py                     Penetapan seed global (reproducibility)
  data/
    data_loader.py               Pemuatan path dan label citra
    preprocessing.py             Resize, denoising, normalisasi, augmentasi
    augmentation.py              Kelas Dataset PyTorch
    partitioner.py               Partisi Dirichlet ke 3 klien
    grouping.py                  Pengelompokan citra near-identical
  federated/
    client.py                    Klien Flower + fungsi objektif FedProx
    server.py                    Strategy kustom, K-Fold, early stopping, checkpoint
    aggregation.py               Agregasi weighted average
  models/
    efficientnet_model.py        EfficientNet-B0 dengan classifier Linear(1280 → 4)
    gradcam.py                   Visualisasi Grad-CAM
  evaluation/
    evaluate.py                  Metrik, confusion matrix, FLOPs, kurva, tabel α

experiments/
  baseline/run_baseline.py       Pelatihan terpusat sebagai pembanding
  run_alpha.py                   Menjalankan satu nilai α
  check_status.py                Memeriksa progres eksperimen
  compare_results.py             Menyusun tabel perbandingan

results/                         Seluruh keluaran eksperimen
```

---

## Keluaran yang Dihasilkan

| Berkas | Isi |
| --- | --- |
| `results/alpha_*/fold_*/metrics.csv` | Metrik tiap ronde komunikasi |
| `results/alpha_*/fold_status.json` | Status penyelesaian tiap fold |
| `results/final/confusion_matrix_final.png` | Confusion matrix 4×4 |
| `results/final/training_curves_alpha_*.png` | Kurva loss dan macro F1 per ronde |
| `results/final/alpha_comparison_test_metrics.csv` | Perbandingan metrik antar α |
| `results/final/laporan_efisiensi.txt` | Jumlah parameter, FLOPs, waktu inferensi |
| `results/final/gradcam/gradcam_hasil.png` | Visualisasi Grad-CAM seluruh kelas |

---

## Catatan Reproducibility

Seed global ditetapkan melalui `SEED` di `config.py` dan diterapkan pada `random`,
`numpy`, serta `torch` (CPU dan GPU), termasuk pada tiap klien Flower. Urutan pemuatan
berkas citra diurutkan secara eksplisit agar pembagian data identik di semua sistem
operasi.

Perlu diperhatikan bahwa dengan hanya 250 kelompok data independen, variansi antar
percobaan cukup besar. Perbedaan kecil antar skenario α sebaiknya tidak ditafsirkan
sebagai perbedaan yang bermakna tanpa memperhatikan sebaran hasil antar fold.

---

## Referensi Utama

- Li, T. et al. (2020). *Federated Optimization in Heterogeneous Networks* (FedProx). MLSys.
- Tan, M. & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for CNNs*. ICML.
- Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks*. ICCV.
- McMahan, H. B. et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS.
