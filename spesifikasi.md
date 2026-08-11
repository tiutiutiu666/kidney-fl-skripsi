# Spesifikasi Aplikasi: Federated Learning EfficientNet + Grad-CAM untuk CT Scan Ginjal

> Dokumen ini dihasilkan dari proposal skripsi Farrelly Theo Ariela (L0122061)
> Tujuan: digunakan sebagai acuan pengecekan kesesuaian kode yang sudah dibangun

---

## 1. GAMBARAN UMUM SISTEM

Sistem ini adalah **pipeline Federated Learning (FL)** yang mensimulasikan 3 klien (rumah sakit virtual) dalam satu perangkat lokal. Setiap klien melatih model EfficientNet-B0 secara lokal, lalu hanya mengirim bobot (weight) ke server pusat untuk diagregasi menggunakan **FedProx**. Setelah pelatihan selesai, model terbaik divisualisasikan menggunakan **Grad-CAM** untuk interpretabilitas.

---

## 2. DATASET

| Parameter      | Nilai                                                               |
| -------------- | ------------------------------------------------------------------- |
| Sumber         | Kaggle — "CT Kidney Dataset: Normal–Cyst–Tumor–and–Stone"           |
| Format citra   | .jpg, 2D (bukan volumetrik NIfTI)                                   |
| Jumlah kelas   | 4: Normal, Cyst, Stone, Tumor                                       |
| Jumlah data    | Normal: 5077 · Cyst: 3709 · Stone: 1377 · Tumor: 2283 (total 12446) |
| Label tambahan | TIDAK ADA (tanpa data klinis seperti usia, jenis kelamin, riwayat)  |

**Catatan penting:** Kelas Stone adalah kelas minoritas (1377 citra) — penting diperhatikan saat evaluasi macro-average.

---

## 3. ALUR PIPELINE SECARA KESELURUHAN

```
Akuisisi Data
     ↓
Data Preprocessing Global
     ↓
Data Partitioning (Distribusi Dirichlet, per nilai α)
     ↓
Data Augmentation (hanya pada data latih)
     ↓
Data Splitting (Stratified 5-Fold Cross-Validation + 20% test set)
     ↓
Pembangunan Lingkungan FL (Flower/flwr, simulasi 3 klien)
     ↓
Pelatihan Model Lokal EfficientNet-B0 dengan FedProx
     ↓
Agregasi Server → Model Global
     ↓
XAI — Grad-CAM pada model global terbaik
     ↓
Evaluasi (Klasifikasi + Kompleksitas)
```

Seluruh alur di atas **diulang untuk setiap nilai α ∈ {0.1, 0.3, 0.5, 0.7, 1.0}** dan setiap dari **5 fold**.

---

## 4. TAHAP 1 — DATA PREPROCESSING GLOBAL

Preprocessing dilakukan **sebelum** partisi data ke klien, bersifat global.

### 4.1 Resize

- Ukuran target: **224 × 224 piksel**
- Alasan: sesuai input standar EfficientNet-B0

### 4.2 Denoising

- Metode: **Gaussian Blur** atau **Median Filter**
- Tujuan: mengurangi noise/artefak CT scan tanpa menghilangkan struktur organ

### 4.3 Normalisasi

- Nilai piksel diubah ke rentang **[0, 1]**
- Menggunakan **mean dan standar deviasi ImageNet** (karena model pretrained ImageNet)
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]

**Tidak boleh ada preprocessing yang berbeda antar klien pada tahap ini.**

---

## 5. TAHAP 2 — DATA PARTITIONING (DISTRIBUSI DIRICHLET)

### 5.1 Konfigurasi

- Jumlah klien: **3** (diasumsikan sebagai 3 rumah sakit virtual)
- Jumlah kelas: **4**
- Nilai α yang diuji: **{0.1, 0.3, 0.5, 0.7, 1.0}**

### 5.2 Cara Kerja Distribusi Dirichlet

- Semakin **kecil α** → data makin tidak merata (extreme non-IID, tiap klien dominan 1-2 kelas)
- Semakin **besar α** → data makin seimbang (mendekati IID)
- Formula PDF Dirichlet:

```
f(p1, p2, ..., pK; α) = [Γ(K·α) / Γ(α)^K] × ∏ pk^(α-1)
```

### 5.3 Algoritma Partisi (wajib sesuai Algoritma 3.1 di proposal)

```
FUNCTION dirichlet_partition(dataset, num_clients=3, alpha, num_classes=4):
  client_indices = [[] for each client]
  FOR each class c IN {Normal, Cyst, Stone, Tumor}:
    idx_c = semua indeks sampel dengan label == c
    SHUFFLE idx_c secara acak
    proportions = Dirichlet(alpha=[alpha * num_clients])  # vektor 3 nilai
    splits = CUMULATIVE_SUM(proportions) × len(idx_c)
    splits = ROUND(splits) sebagai integer
    FOR k = 1 TO 3:
      APPEND idx_c[splits[k-1]:splits[k]] TO client_indices[k]
  FOR k = 1 TO 3:
    SHUFFLE client_indices[k]
  RETURN client_indices
```

### 5.4 Dua Jenis Non-IID yang Harus Muncul

1. **Label distribution skew**: proporsi kelas berbeda antar klien
2. **Quantity skew**: jumlah total data berbeda antar klien

---

## 6. TAHAP 3 — DATA AUGMENTATION

Augmentasi **hanya diterapkan pada data latih** di setiap iterasi fold. Data validasi dan data uji **TIDAK** diaugmentasi.

| Teknik Augmentasi      | Parameter         | Keterangan                           |
| ---------------------- | ----------------- | ------------------------------------ |
| Random Horizontal Flip | Probability = 0.5 | Membalik citra secara horizontal     |
| Random Rotation        | Range = ±15°      | Rotasi kecil, struktur tetap terjaga |
| Translation            | Shift = ±10%      | Geser horizontal/vertikal            |
| Scaling (Zoom)         | Range = 0.9 – 1.1 | Zoom proporsional                    |
| Intensity Adjustment   | Brightness ±10%   | Variasi pencahayaan piksel           |

**Catatan:** Parameter augmentasi tidak ekstrem, dirancang untuk menjaga makna klinis citra medis.

---

## 7. TAHAP 4 — DATA SPLITTING

### 7.1 Pembagian Awal (per klien)

- **20%** → **Test set** (disimpan, tidak digunakan selama pelatihan sama sekali)
- **80%** → digunakan untuk K-Fold Cross-Validation

### 7.2 Stratified 5-Fold Cross-Validation

- K = **5**
- Pendekatan: **Stratified** (proporsi kelas dijaga di setiap fold)
- Setiap iterasi:
  - 4 fold sebagai **data latih** (64% dari total data klien)
  - 1 fold sebagai **data validasi** (16% dari total data klien)
- Proses ini **diulang 5 kali**, setiap fold menjadi validasi tepat 1 kali

### 7.3 Komposisi Akhir (dari keseluruhan data klien)

```
64%  → Data Latih   (80% × 80%)
16%  → Data Validasi (20% × 80%)
20%  → Data Uji (terpisah, hanya untuk evaluasi akhir)
```

### 7.4 Augmentasi pada Splitting

- Augmentasi diterapkan **hanya pada 64% data latih**
- Validasi dan uji: **tanpa augmentasi**

---

## 8. TAHAP 5 — PEMBANGUNAN LINGKUNGAN FL

### 8.1 Framework

- **Flower (flwr)**, modul `flwr.simulation`
- Berjalan pada **1 perangkat lokal** (bukan jaringan fisik)

### 8.2 Konfigurasi Server

```
μ (proximal_mu)       = 0.1
num_rounds            = 100
min_fit_clients       = 3
min_available_clients = 3
Strategi agregasi     = FedProx
Bobot awal            = EfficientNet-B0 pretrained ImageNet
Early stopping        = aktif (berhenti jika validation loss global tidak turun signifikan)
```

### 8.3 Konfigurasi Klien

```
num_clients           = 3
Local epochs per ronde = 5
Optimizer             = Adam, lr = 1e-4
Loss function         = CrossEntropyLoss + FedProx proximal term
Batch size            = 32
```

### 8.4 Fungsi Objektif FedProx pada Klien (wajib ada proximal term)

```
loss_total = CrossEntropyLoss(y_pred, y) + (μ/2) × ||w_k − w*||²
```

di mana `w*` adalah bobot global dari server dan `w_k` adalah bobot lokal klien ke-k.

### 8.5 Alur Setiap Ronde FL

```
Server broadcast bobot global w* ke semua klien
  → Setiap klien:
      - Load EfficientNet-B0, set bobot = w*
      - Latih selama 5 local epoch dengan loss FedProx
      - Kirim bobot lokal w_k ke server
  → Server:
      - Agregasi w_k dari semua klien → w* baru (FedProx aggregate)
      - Hitung validation loss global
      - Cek early stopping
Ulangi hingga num_rounds=100 atau konvergen
```

### 8.6 Iterasi Eksperimen

- Seluruh proses ini **diulang untuk setiap kombinasi**:
  - 5 nilai α × 5 fold = **25 kali eksperimen** total

---

## 9. TAHAP 6 — MODEL (EfficientNet-B0)

### 9.1 Arsitektur

- **EfficientNet-B0** (backbone pretrained ImageNet)
- Lapisan klasifikasi diganti: `Linear(1280 → 4)`
  - 1280 = dimensi output Global Average Pooling EfficientNet-B0
  - 4 = jumlah kelas (Normal, Cyst, Stone, Tumor)

### 9.2 Strategi Pelatihan

- **Full fine-tuning** — seluruh layer diperbarui (tidak ada layer yang dibekukan/frozen)
- Bobot awal dari ImageNet sebagai titik mulai, disesuaikan via backpropagation

### 9.3 Input Model

- Ukuran: **224 × 224 piksel**
- Menerima dua jenis irisan CT scan: **aksial dan koronal** (tidak dibedakan, diproses seragam)

### 9.4 Komponen Arsitektur (dari proposal)

```
Input (224×224) → Preprocessing → Stem Conv 3×3
  → MBConv blok 1–7
  → Conv Head 1×1
  → Global Average Pooling → Feature Vector (1280)
  → Linear(1280 → 4) [classifier head baru]
  → Softmax → probabilitas 4 kelas
```

### 9.5 Hyperparameter (Tabel 3.2 Proposal)

| Parameter             | Nilai                                 |
| --------------------- | ------------------------------------- |
| Arsitektur            | EfficientNet-B0                       |
| Bobot awal            | ImageNet (pretrained)                 |
| Jumlah kelas output   | 4                                     |
| Optimizer             | Adam                                  |
| Learning rate         | 1 × 10⁻⁴                              |
| Local epoch per ronde | 5                                     |
| Loss function         | Cross-Entropy + FedProx proximal term |
| Batch size            | 32                                    |

---

## 10. TAHAP 7 — XAI (GRAD-CAM)

### 10.1 Kapan Dijalankan

- Setelah diperoleh **model global terbaik** dari seluruh eksperimen FL
- Tidak mengubah arsitektur atau melatih ulang

### 10.2 Library

- `pytorch-grad-cam`

### 10.3 Target Layer

- Lapisan konvolusi terakhir pada blok MBConv terakhir: **`features[-1]`**
- Alasan: lapisan ini menangkap representasi fitur tingkat tinggi yang paling relevan

### 10.4 Sampel Uji

- Dipilih secara representatif dari **masing-masing 4 kelas**
- Menggunakan data test set (yang tidak pernah dilihat model saat training)

### 10.5 Proses Grad-CAM (Algoritma 3.5)

```
FOR each sample (image, true_label) IN test_samples:
  pred_class = model.predict(image)
  cam_map = GradCAM.compute(image, target_class=pred_class)
  cam_map = normalize(cam_map, range=[0,1])
  cam_map = resize(cam_map, size=(224,224), method=bilinear)
  heatmap = apply_colormap(cam_map, colormap='jet')
  overlay = superimpose(heatmap, image, alpha=0.5)
  SAVE overlay
```

### 10.6 Interpretasi Warna Heatmap

- **Merah/terang** = area paling berpengaruh terhadap prediksi
- **Biru/gelap** = kontribusi rendah

### 10.7 Evaluasi Grad-CAM

- **Kualitatif** — tidak kuantitatif (karena dataset tidak memiliki anotasi segmentasi piksel)
- Cek kesesuaian area aktivasi dengan struktur anatomi yang relevan secara klinis (posisi tumor, batu, kista)

### 10.8 Formula Matematis Grad-CAM

Bobot feature map ke-k:

```
α_k^c = (1/Z) × ΣΣ (∂y^c / ∂A_ij^k)
```

Heatmap akhir:

```
L_GradCAM^c = ReLU(Σ_k α_k^c × A^k)
```

---

## 11. TAHAP 8 — EVALUASI

### 11.1 Pemilihan Model Terbaik

- Berdasarkan **rata-rata macro F1-score pada data validasi** dari seluruh fold dan seluruh skenario α
- Model dengan nilai macro F1 tertinggi → model final

### 11.2 Evaluasi Kinerja Klasifikasi

- Menggunakan **test set** yang terpisah (20%)
- Metrik:

| Metrik    | Formula                                    | Pendekatan    |
| --------- | ------------------------------------------ | ------------- |
| Accuracy  | ΣTP_i / N                                  | Global        |
| Precision | TP_i / (TP_i + FP_i) per kelas → rata-rata | Macro-average |
| Recall    | TP_i / (TP_i + FN_i) per kelas → rata-rata | Macro-average |
| F1-Score  | 2×P×R / (P+R) per kelas → rata-rata        | Macro-average |

**Macro-average wajib digunakan** (bukan weighted-average) karena data tidak seimbang dan semua kelas diperlakukan setara — termasuk kelas minoritas Stone yang penting klinis.

- **Confusion Matrix 4×4** disajikan untuk melihat pola kesalahan (Normal vs Cyst vs Stone vs Tumor)
- Kurva **training loss dan validation loss per ronde** disajikan untuk melihat konvergensi

### 11.3 Evaluasi Kompleksitas Model

**FLOPs** (Floating Point Operations):

```
FLOPs = 2 × H_o × W_o × C_o × (K_h × K_w × C_i)
```

- Dihitung dengan library **`torchinfo`**

**Waktu Inferensi:**

```
T = t_end − t_start
```

- Dihitung sebagai **rata-rata dari 100 kali percobaan** pada 1 sampel (untuk meredam variasi sistem)

### 11.4 Tabel Perbandingan Antar Skenario α

Hasil disajikan dalam tabel yang membandingkan metrik untuk setiap α:

| α   | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 |
| --- | -------- | --------------- | ------------ | -------- |
| 0.1 | ...      | ...             | ...          | ...      |
| 0.3 | ...      | ...             | ...          | ...      |
| 0.5 | ...      | ...             | ...          | ...      |
| 0.7 | ...      | ...             | ...          | ...      |
| 1.0 | ...      | ...             | ...          | ...      |

Tabel ini digunakan untuk menganalisis **pengaruh tingkat non-IID terhadap performa model FedProx**.

---

## 12. CHECKLIST KESESUAIAN KODE

Gunakan checklist berikut untuk memverifikasi kode yang sudah dibangun:

### Preprocessing

- [ ] Resize ke 224×224
- [ ] Denoising menggunakan Gaussian Blur atau Median Filter
- [ ] Normalisasi menggunakan mean/std ImageNet

### Partisi Data

- [ ] Menggunakan Distribusi Dirichlet
- [ ] 3 klien
- [ ] 5 variasi α: {0.1, 0.3, 0.5, 0.7, 1.0}
- [ ] Memunculkan label skew DAN quantity skew

### Augmentasi

- [ ] Hanya pada data latih (bukan validasi/uji)
- [ ] 5 teknik: Flip, Rotation ±15°, Translation ±10%, Scaling 0.9–1.1, Brightness ±10%

### Data Splitting

- [ ] 20% test set dipisah dulu sebelum fold
- [ ] Stratified 5-Fold pada 80% sisanya
- [ ] Rasio akhir: 64% train / 16% val / 20% test

### Lingkungan FL

- [ ] Framework: Flower (flwr), modul `flwr.simulation`
- [ ] 3 klien virtual
- [ ] Strategi agregasi: FedProx (bukan FedAvg)
- [ ] μ (proximal_mu) = 0.1
- [ ] num_rounds = 100
- [ ] min_fit_clients = 3
- [ ] Early stopping aktif
- [ ] FedProx proximal term: `(μ/2) × ||w_k − w*||²`

### Model

- [ ] Arsitektur: EfficientNet-B0
- [ ] Bobot awal: ImageNet pretrained
- [ ] Classifier head diganti: Linear(1280 → 4)
- [ ] Full fine-tuning (tidak ada layer frozen)
- [ ] Optimizer: Adam, lr = 1e-4
- [ ] Loss: CrossEntropyLoss + proximal term
- [ ] Batch size: 32
- [ ] Local epochs per ronde: 5

### XAI

- [ ] Library: pytorch-grad-cam
- [ ] Target layer: features[-1] (konvolusi terakhir MBConv)
- [ ] Normalisasi cam_map ke [0,1]
- [ ] Resize ke 224×224 (bilinear interpolation)
- [ ] Colormap: jet
- [ ] Overlay alpha = 0.5
- [ ] Sampel dari semua 4 kelas
- [ ] Evaluasi bersifat kualitatif

### Evaluasi

- [ ] Metrik: Accuracy, Precision, Recall, F1-Score
- [ ] Pendekatan: Macro-average
- [ ] Confusion matrix 4×4
- [ ] Kurva loss per ronde
- [ ] Pemilihan model terbaik berdasarkan rata-rata macro F1 validasi
- [ ] FLOPs dihitung dengan `torchinfo`
- [ ] Waktu inferensi: rata-rata 100 percobaan
- [ ] Tabel perbandingan semua nilai α

---

## 13. HAL-HAL YANG BUKAN BAGIAN DARI SCOPE (Batasan Masalah)

Berdasarkan Subbab 1.3 proposal, hal berikut **tidak termasuk** dalam implementasi:

- ❌ Penerapan pada jaringan rumah sakit nyata
- ❌ Data klinis pasien (usia, jenis kelamin, riwayat penyakit)
- ❌ Dataset selain CT Kidney Dataset dari Kaggle
- ❌ Arsitektur selain EfficientNet-B0
- ❌ Algoritma agregasi selain FedProx
- ❌ Lebih dari 3 klien
- ❌ Evaluasi Grad-CAM secara kuantitatif (tidak ada ground-truth segmentasi)
- ❌ Arsitektur FL lain seperti FedMilNet atau metode distilasi pengetahuan

---

_Dokumen ini dibuat berdasarkan Proposal Skripsi Farrelly Theo Ariela, L0122061_
_Universitas Sebelas Maret, Fakultas Teknologi Informasi dan Sains Data, 2026_
