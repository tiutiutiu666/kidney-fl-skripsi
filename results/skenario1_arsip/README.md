# Arsip Hasil Skenario I — Pembagian Acak per Citra

Rekap angka dari eksperimen tahap pertama, yaitu sebelum pembagian data
group-aware diterapkan. Disimpan di sini agar tetap tersedia untuk penulisan
Bab IV meskipun bobot model dan berkas mentahnya hanya ada di server.

## Isi

| Berkas | Keterangan |
| --- | --- |
| `validasi_per_fold.csv` | Macro F1 validasi tiap fold, kelima nilai alpha |
| `validasi_ringkas.csv` | Rata-rata, standar deviasi, minimum, dan maksimum per alpha |
| `test_metrics_per_alpha.csv` | Metrik test set per alpha |
| `perbandingan_skenario.csv` | Perbandingan langsung Skenario I dan Skenario II |

## Ringkasan

Macro F1 **validasi** (rata-rata lintas 5 fold):

| alpha | Skenario I | Skenario II (group-aware) |
| --- | --- | --- |
| 0.1 | 0,8195 | 0,7159 |
| 0.3 | 0,9934 | 0,8009 |
| 0.5 | 0,9981 | 0,8166 |
| 0.7 | 0,9989 | 0,8209 |
| 1.0 | 0,9997 | 0,8073 |

Macro F1 **test set**:

| alpha | Skenario I | Skenario II | Selisih |
| --- | --- | --- | --- |
| 0.1 | 0,8949 | 0,7567 | −0,1382 |
| 0.3 | 0,9925 | 0,7802 | −0,2123 |
| 0.5 | 0,9994 | 0,7872 | −0,2122 |
| 0.7 | 0,9969 | 0,8436 | −0,1533 |
| 1.0 | 0,9925 | 0,8186 | −0,1739 |

Pada Skenario I, keempat nilai alpha di atas 0.1 menghasilkan macro F1 antara
0,99 dan 1,00 — seluruhnya mentok di batas atas, sehingga pengaruh tingkat
non-IID tidak dapat diamati sama sekali. Perbedaan antar alpha baru terlihat
setelah kebocoran data dikendalikan pada Skenario II.

## Tiga hal yang harus diperhatikan saat mengutip angka ini

**1. Angka Skenario I mengandung kebocoran data.** Pembagian acak per citra
menyebabkan 99,76% citra uji memiliki citra nyaris identik pada data latih.
Angka-angka ini menggambarkan kemampuan mengenali kembali pasien yang sudah
pernah dilihat, bukan kemampuan generalisasi pada pasien baru. Sajikan selalu
berdampingan dengan penjelasan tersebut.

**2. Format kedua skenario tidak sepenuhnya sama.** Kolom test pada Skenario I
berasal dari **satu model per alpha**, yaitu fold dengan macro F1 validasi
tertinggi. Skenario II dilaporkan sebagai **rata-rata seluruh lima fold**
beserta standar deviasinya. Perbandingan pada `perbandingan_skenario.csv`
karenanya bersifat indikatif; selisihnya jauh melampaui sebaran antar fold
sehingga arah kesimpulannya tetap sahih.

**3. Hasil Skenario I tidak dapat dihitung ulang dengan kode saat ini.**
Eksperimen tersebut berjalan sebelum `sorted()` ditambahkan pada
`src/data/data_loader.py`. Ketika itu urutan pemuatan citra mengikuti
`os.listdir()` yang tidak terurut, sedangkan `test_indices.npy` menyimpan
posisi angka, bukan nama berkas. Menjalankan ulang evaluasi dengan kode
sekarang akan memilih kumpulan citra yang berbeda tanpa memunculkan galat apa
pun. Gunakan angka pada arsip ini, jangan menghitung ulang.

## Berkas mentah

Bobot model, `metrics.csv` tiap ronde, confusion matrix, kurva pelatihan, dan
visualisasi Grad-CAM Skenario I tersimpan di server pada
`~/kidneyfl/results_skenario1_split_acak/` beserta salinan terkompresinya.
Berkas `.pth` tidak disertakan di repositori karena batas ukuran GitHub.
