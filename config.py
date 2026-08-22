import os

# ── Path ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "raw",
               "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone",
               "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone")
RESULT_DIR   = os.path.join(BASE_DIR, "results")
BASELINE_DIR = os.path.join(RESULT_DIR, "baseline")

# ── Dataset ───────────────────────────────────────────────────────────────────
NUM_CLASSES  = 4
KELAS_LIST   = ["Cyst", "Normal", "Stone", "Tumor"]
IMG_SIZE     = 224

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE   = 32
LOCAL_EPOCHS = 5
NUM_ROUNDS   = 100
NUM_CLIENTS  = 3
LR           = 1e-4
MU_FEDPROX   = 0.1
K_FOLD       = 5
TEST_SPLIT   = 0.20
SEED         = 42

# ── Preprocessing ────────────────────────────────────────────────────────────
DENOISE_SIGMA = 0.5          # radius Gaussian blur untuk denoising

# ── Checkpoint & Early Stopping ───────────────────────────────────────────────
CHECKPOINT_INTERVAL       = 10
EARLY_STOPPING_PATIENCE   = 15  # ronde tanpa peningkatan sebelum berhenti

# ── Eksperimen ────────────────────────────────────────────────────────────────
ALPHAS = [0.1, 0.3, 0.5, 0.7, 1.0]

# ── Group-aware splitting (pencegahan data leakage) ──────────────────────────
# Dataset CT Kidney berisi banyak irisan (slice) near-identical dari scan yang
# sama. Split acak per-CITRA menempatkan irisan dari scan yang sama sekaligus di
# train dan test, sehingga metrik uji terlalu optimistis (mendekati 100%).
#
# Jika True: citra dikelompokkan berdasarkan kemiripan konten, lalu split
# train/val/test dilakukan per-GRUP (GroupShuffleSplit + StratifiedGroupKFold)
# sehingga satu grup tidak pernah terbelah antar split.
#
# PERHATIAN: mengubah ini ke True mengubah komposisi seluruh split, sehingga
# hasil eksperimen lama tidak lagi sebanding — semua eksperimen harus diulang.
GROUP_AWARE_SPLIT    = True
GROUP_DESC_RES       = 64     # resolusi deskriptor citra untuk uji kemiripan
GROUP_SIM_THRESHOLD  = 0.99   # korelasi minimum agar 2 citra dianggap 1 grup

# ── Regularisasi tambahan (penanganan overfitting) ────────────────────────────
# Pada skenario group-aware, model mencapai akurasi latih 99,8% namun hanya
# 80,5% pada data uji — selisih 19,3 poin. Dataset hanya memuat sekitar 250
# kasus independen, sehingga model menghafal kasus latih.
#
# Ablation pada model terpusat (experiments/ablasi_regularisasi.py) menunjukkan
# augmentasi yang diperkuat dipadu label smoothing menaikkan macro F1 uji dari
# 0,7742 menjadi 0,8599 (+8,57 poin) sekaligus memangkas selisih latih-uji
# menjadi 10,7 poin.
#
# Jika True:
#   - Parameter augmentasi diperluas (lihat AUG_* di bawah)
#   - Loss lokal maupun terpusat memakai label smoothing
#   - Seluruh keluaran ditulis ke folder ber-akhiran SUFIKS_EKSPERIMEN,
#     sehingga hasil eksperimen lama TIDAK tertimpa dan tetap dapat direproduksi
#     dengan mengembalikan flag ini ke False.
#
# PERHATIAN: mengubah ini ke True mengubah konfigurasi pelatihan, sehingga
# seluruh eksperimen harus dijalankan ulang agar sebanding.
REGULARISASI_KUAT = True

# Parameter augmentasi. Nilai "standar" mengikuti Tabel 3.1 proposal; nilai
# "kuat" memperluas rentang teknik yang sama tanpa menambah teknik baru,
# sehingga tetap sejalan dengan pertimbangan citra medis pada proposal.
if REGULARISASI_KUAT:
    AUG_ROTATION   = 25            # derajat
    AUG_TRANSLATE  = 0.15          # proporsi lebar/tinggi
    AUG_SCALE      = (0.8, 1.2)
    AUG_BRIGHTNESS = 0.2
    AUG_CONTRAST   = 0.2
    LABEL_SMOOTHING = 0.1
else:
    AUG_ROTATION   = 15
    AUG_TRANSLATE  = 0.1
    AUG_SCALE      = (0.9, 1.1)
    AUG_BRIGHTNESS = 0.1
    AUG_CONTRAST   = 0.0
    LABEL_SMOOTHING = 0.0

# Akhiran folder keluaran. Dipakai pada nama folder alpha_* dan baseline agar
# hasil kedua konfigurasi tersimpan berdampingan tanpa saling menimpa.
SUFIKS_EKSPERIMEN = "_reg" if REGULARISASI_KUAT else ""


def dir_alpha(alpha):
    """Path folder hasil untuk satu nilai alpha, sesuai konfigurasi aktif."""
    return os.path.join(RESULT_DIR, f"alpha_{alpha}{SUFIKS_EKSPERIMEN}")


def dir_baseline():
    """Path folder hasil baseline, sesuai konfigurasi aktif."""
    return os.path.join(RESULT_DIR, f"baseline{SUFIKS_EKSPERIMEN}")


def dir_final():
    """Path folder keluaran akhir (evaluasi, Grad-CAM, kurva, tabel)."""
    return os.path.join(RESULT_DIR, f"final{SUFIKS_EKSPERIMEN}")