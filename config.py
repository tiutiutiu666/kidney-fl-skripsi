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