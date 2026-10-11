# TB1-JST-Kelompok_Bebas — Klasifikasi Keterlambatan Pengiriman E-Commerce dengan Perceptron dan ADALINE

Tugas Besar 1 mata kuliah **Jaringan Syaraf Tiruan (IT6053)**, FTI UKDW.

**Kasus.** Memprediksi apakah suatu pesanan e-commerce akan **terlambat dikirim** berdasarkan informasi yang sudah tersedia saat pesanan diproses (berat, nilai, jarak, estimasi hari, biaya kirim, lama proses gudang, segmen pelanggan, kategori produk, metode kirim, kurir, ukuran paket, metode bayar, prioritas, dan cuaca). Masalah ini dirumuskan sebagai **klasifikasi biner terawasi**:

| Label | Arti | Asal |
|:--:|---|---|
| **+1 (kelas positif)** | Terlambat | `late_delivery = "Yes"` |
| −1 | Tepat waktu | `late_delivery = "No"` |

Dua neuron tunggal, **Perceptron** (Rosenblatt, 1958) dan **ADALINE / Delta Rule** (Widrow & Hoff, 1960), diimplementasikan sendiri dengan NumPy dan dibandingkan pada 8 konfigurasi pelatihan.

---

## Anggota Kelompok

| No | Nama | NIM |
|:--:|---|---|
| 1 | Stefanus Westin Giovanno Mursito | 71241073 |
| 2 | Giovanni Albert Harjanto | 71241075 |
| 3 | Natalie Neysa Jessica Soesanto | 71241079 |
| 4 | Fransiska | 71241123 |
| 5 | Kevin Surya Saputra | 71241150 |

---

## Struktur Repository

| Berkas / folder | Isi |
|---|---|
| [TB1_Kelompok_Bebas.ipynb](TB1_Kelompok_Bebas.ipynb) | Notebook utama: data, preprocessing, implementasi Perceptron & ADALINE, perhitungan manual, 8 pelatihan, evaluasi, analisis, pembahasan (sudah dieksekusi, keluaran tersimpan) |
| [TB1_Kelompok_Bebas.pdf](TB1_Kelompok_Bebas.pdf) | Laporan (5–8 halaman isi utama) |
| [Presentasi_TB1.pdf](Presentasi_TB1.pdf) | Slide presentasi |
| [dataset.csv](dataset.csv) | Data yang benar-benar dipakai dalam eksperimen (50.000 baris × 16 kolom) |
| [prepare_dataset.py](prepare_dataset.py) | Skrip pembuat `dataset.csv` dari data mentah Kaggle |
| [data/raw/E-commerce_Delivery_Shipping_Data_2026.csv](data/raw/E-commerce_Delivery_Shipping_Data_2026.csv) | Data mentah Kaggle (CC0), disimpan untuk keterlacakan |
| [requirements.txt](requirements.txt) | Dependensi dan versinya |
| [figures/](figures/) | Semua gambar yang dihasilkan notebook (EDA, kurva, confusion matrix, bobot, decision boundary) |
| [results/](results/) | Tabel hasil dalam CSV (ringkasan 8 konfigurasi, epoch 10/30/50, confusion matrix, bobot & bias, split, validasi) |
| [docs/draft_laporan.md](docs/draft_laporan.md) | Draf isi laporan (bahan untuk PDF) |
| [Tugas_Besar_1_JST_Panduan_dan_Rubrik.pdf](Tugas_Besar_1_JST_Panduan_dan_Rubrik.pdf) | Panduan & rubrik tugas dari dosen |

---

## Data

**Sumber.** *E-commerce Delivery & Shipping Data 2026*, Kaggle: <https://www.kaggle.com/datasets/datascikhan/e-commerce-delivery-and-shipping-data-2026>.
**Lisensi:** CC0 1.0 (domain publik), boleh disalin, dimodifikasi, dan dipublikasikan. Data tidak memuat informasi pribadi (ID pelanggan tidak dipakai).

**Karakteristik.** 50.000 pesanan; 0 missing value; 0 duplikat baris/`order_id`; distribusi kelas: Terlambat 28.162 (56,32%), Tepat waktu 21.838 (43,68%).

### Kolom `dataset.csv`

| Kolom | Tipe | Makna |
|---|---|---|
| `order_id` | identitas | ID pesanan (bukan fitur, hanya pelacak baris) |
| `product_weight_kg` | numerik | Berat produk (kg) |
| `order_value_usd` | numerik | Nilai pesanan (USD) |
| `distance_km` | numerik | Jarak gudang ke pelanggan (km) |
| `promised_delivery_days` | numerik | Estimasi hari pengiriman yang dijanjikan |
| `shipping_cost_usd` | numerik | Biaya pengiriman (USD) |
| `warehouse_processing_hours` | numerik | Lama pemrosesan di gudang (jam) |
| `customer_segment` | nominal (4) | Consumer / Enterprise / Premium / Small Business |
| `product_category` | nominal (12) | Kategori produk |
| `shipping_method` | nominal (5) | Economy / Express / International / Same Day / Standard |
| `carrier` | nominal (8) | Perusahaan kurir |
| `package_size` | ordinal (4) | Small < Medium < Large < Oversized |
| `payment_method` | nominal (6) | Metode pembayaran |
| `order_priority` | ordinal (4) | Low < Normal < High < Urgent |
| `weather_condition` | nominal (6) | Clear / Cloudy / Extreme Heat / Rain / Snow / Storm |
| `late_delivery` | **target** | `Yes` → +1 (Terlambat), `No` → −1 (Tepat waktu) |

### Transformasi data

1. **`prepare_dataset.py`** (data mentah → `dataset.csv`): hanya memilih kolom, tanpa mengubah nilai. Kolom yang baru diketahui saat/sesudah pengiriman dibuang karena **bocor** terhadap target: `actual_delivery_days`, `delivery_delay_days`, `delivery_status`, `tracking_status`, `return_requested`, `return_reason`, `customer_rating`, `delivery_attempts`. Kolom identitas/lokasi/tanggal (`customer_id`, kota, negara, gudang, `order_date`) juga tidak dipakai.
2. **Di notebook**, semuanya dipelajari **hanya dari data latih**:
   - split 80:20 stratified, `random_state = 42` (sama untuk kedua model);
   - imputasi median/modus (disiapkan walaupun data tidak memiliki missing value);
   - ordinal encoding untuk `package_size` dan `order_priority`;
   - one-hot *drop-first* untuk 6 fitur nominal → total **43 fitur**;
   - standardisasi z-score (mean & std dari data latih), dan versi tanpa standardisasi untuk 2 eksperimen pembanding.

---

## Instalasi dan Cara Menjalankan

Diuji dengan **Python 3.13.7** (Windows 11).

```bash
git clone https://github.com/albertharjanto/TB1-JST-Kelompok_Bebas.git
cd TB1-JST-Kelompok_Bebas
python -m venv .venv
# Windows: .venv\Scripts\activate    |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# (opsional) membuat ulang dataset.csv dari data mentah
python prepare_dataset.py

# menjalankan notebook secara interaktif
jupyter notebook TB1_Kelompok_Bebas.ipynb
# atau mengeksekusi seluruh notebook dari awal sampai akhir
jupyter nbconvert --to notebook --execute --inplace TB1_Kelompok_Bebas.ipynb
```

Eksekusi penuh memerlukan sekitar 2–3 menit (setiap Perceptron dilatih per sampel ±8–11 detik). Gambar dan tabel otomatis ditulis ulang ke `figures/` dan `results/`.

**Keterulangan.** `SEED = 42` dipakai untuk split latih/uji, split validasi, dan inisialisasi acak (eksperimen pembanding). Inisialisasi utama adalah nol, dan urutan sampel tetap (urutan hasil split, tersimpan di `results/split_train_test.csv`). Konfigurasi ditetapkan sebelum melihat data uji. Pemilihan model terbaik memakai data validasi (20% dari data latih).

---

## Ringkasan Metode

| Komponen | Perceptron | ADALINE |
|---|---|---|
| Neuron | z = wᵀx + b | z = wᵀx + b |
| Prediksi | ŷ = +1 jika z ≥ 0, selain itu −1 | sama (ambang hanya saat prediksi) |
| Error | y − ŷ (diskret) | e = y − z (keluaran linear) |
| Pembaruan | per sampel: Δ = η(y − ŷ); w ← w + Δx; b ← b + Δ | batch GD: w ← w + ηXᵀe/n; b ← b + η·mean(e) |
| Riwayat | jumlah pembaruan & akurasi latih per epoch | J = (1/2n)Σe² dicatat **sesudah** pembaruan (+ J awal), MSE = 2J |
| Inisialisasi | w = 0, b = 0 | w = 0, b = 0 |

Eksperimen: Perceptron & ADALINE × η ∈ {0,001; 0,01; 0,1} × 50 epoch pada data terstandardisasi, ditambah masing-masing satu pelatihan tanpa standardisasi (η = 0,01). Total **8 pelatihan**.

---

## Ringkasan Hasil (data uji, n = 10.000)

| Model | η | Std | Akurasi latih | Akurasi uji | Precision(+1) | Recall(+1) | F1(+1) | Loss/error akhir |
|---|---|---|---|---|---|---|---|---|
| Perceptron | 0,001 | Ya | 0,7443 | 0,7460 | 0,7874 | 0,7521 | 0,7693 | 10.674 update di epoch 50 |
| Perceptron | 0,01 | Ya | 0,7443 | 0,7460 | 0,7874 | 0,7521 | 0,7693 | 10.674 update di epoch 50 |
| Perceptron | 0,1 | Ya | 0,7443 | 0,7460 | 0,7874 | 0,7521 | 0,7693 | 10.674 update di epoch 50 |
| ADALINE | 0,001 | Ya | 0,7428 | 0,7403 | 0,7958 | 0,7250 | 0,7587 | J = 0,4724 |
| ADALINE | 0,01 | Ya | 0,7683 | 0,7676 | 0,8096 | 0,7679 | 0,7882 | J = 0,3578 |
| **ADALINE** | **0,1** | **Ya** | **0,7959** | **0,7943** | **0,8240** | **0,8072** | **0,8155** | **J = 0,2898** |
| Perceptron | 0,01 | Tidak | 0,4430 | 0,4437 | 0,6971 | 0,0217 | 0,0420 | 17.447 update di epoch 50 |
| ADALINE | 0,01 | Tidak | 0,4368 | 0,4368 | 0 | 0 | 0 | **divergen (J = ∞) pada epoch 31** |

Baseline (selalu menebak "Terlambat"): akurasi 0,5632. Acuan optimum least squares (model linear terbaik menurut J): J = 0,2881, akurasi uji 0,7947.

**Temuan utama**

- **Data tidak terpisah linear.** Perceptron tetap melakukan sekitar 10.600–10.700 pembaruan per epoch hingga epoch 50, akurasi latihnya berosilasi 0,730–0,758, dan distribusi z kedua kelas tumpang-tindih.
- **ADALINE η = 0,1** hampir mencapai optimum least squares (J 0,2898 vs 0,2881; kosinus arah bobot 0,991). Batas ±79% adalah batas model linear, bukan kegagalan optimisasi. Konfigurasi ini juga terpilih lewat data validasi (akurasi validasi 0,7996).
- **Tiga η Perceptron identik**, dan hal ini sah. Dengan inisialisasi nol dan urutan tetap, η hanya menskalakan bobot (‖w‖ = 0,109 / 1,093 / 10,926). Dengan inisialisasi acak, hasilnya berbeda.
- **Standardisasi wajib.** Tanpa standardisasi, λ_max(XᵀX/n) ≈ 9,5×10⁶ sehingga batas stabil η < 2,1×10⁻⁷, dan ADALINE η = 0,01 divergen. Perceptron tanpa standardisasi didominasi `distance_km` dan jatuh di bawah baseline.

Rincian lengkap ada di notebook (bagian 6–9) dan di [results/](results/).

---

## Artikel dan Kode Rujukan

| Sumber | Kontribusi pada tugas |
|---|---|
| Rosenblatt (1958). *Psychological Review* 65(6). [doi:10.1037/h0042519](https://doi.org/10.1037/h0042519) | Dasar aturan pembaruan Perceptron dan argumen konvergensi (pembaruan berhenti jika data terpisah linear), dipakai untuk menafsirkan kurva pembaruan |
| Widrow & Hoff (1960). *IRE WESCON Convention Record*. [PDF](https://isl.stanford.edu/~widrow/papers/c1960adaptiveswitching.pdf) | ADALINE / LMS: error dari keluaran linear dan gradient pada J |
| Pratama, Anrahvi, Tambal, & Singh (2025). *PREDATECS* 3(1). [doi:10.57152/predatecs.v3i1.1855](https://doi.org/10.57152/predatecs.v3i1.1855) | Artikel kasus: klasifikasi ketepatan waktu pengiriman e-commerce dengan supervised learning (akurasi terbaik ±66%), sebagai konteks tingkat kesulitan kasus |
| LeCun, Bottou, Orr, & Müller (1998). *Efficient BackProp*. [doi:10.1007/3-540-49430-8_2](https://doi.org/10.1007/3-540-49430-8_2) | Alasan standardisasi input: kondisi Hessian dan kestabilan gradient descent |
| Gallant (1990). *IEEE Trans. Neural Networks* 1(2). [doi:10.1109/72.80230](https://doi.org/10.1109/72.80230) | *Pocket algorithm* sebagai pengembangan Perceptron untuk data tak terpisah linear |
| Raschka & Mirjalili (2019), kode bab 2: [rasbt/python-machine-learning-book-3rd-edition/ch02](https://github.com/rasbt/python-machine-learning-book-3rd-edition/tree/master/ch02) dan [dokumentasi mlxtend Adaline](https://rasbt.github.io/mlxtend/user_guide/classifier/Adaline/) | **Kode rujukan yang diadaptasi**: struktur kelas `fit`/`net_input`/`predict` serta pencatatan riwayat |
| [Dokumentasi scikit-learn](https://scikit-learn.org/stable/) | `train_test_split`, `StandardScaler` (verifikasi), metrik evaluasi, dan `Perceptron` (hanya pembanding) |

**Perubahan terhadap kode rujukan:** (1) pembaruan Perceptron memakai Δ = η(y − ŷ) dan ambang z ≥ 0 sesuai panduan; (2) label −1/+1 dan inisialisasi nol (rujukan memakai bobot acak kecil); (3) ADALINE memakai J = (1/2n)Σe² yang dicatat sesudah pembaruan, ditambah J awal dan MSE; (4) deteksi divergensi (inf/NaN → berhenti, catat epoch, kembalikan parameter terakhir yang berhingga); (5) pencatatan jumlah pembaruan, akurasi latih, dan riwayat bobot per epoch; (6) preprocessor buatan sendiri yang di-*fit* hanya pada data latih.

---

## Pernyataan Bantuan AI

Asisten AI generatif dipakai untuk membantu menyusun kerangka kode, visualisasi, serta draf narasi notebook dan README. Cara kelompok memverifikasi:

1. Perhitungan manual 2 langkah Perceptron dan 2 iterasi ADALINE dicek dengan `assert` agar identik dengan keluaran kelas yang dipakai dalam eksperimen.
2. Standardisasi buatan sendiri dicek identik dengan `sklearn.preprocessing.StandardScaler` yang di-*fit* pada data latih. Tidak ada irisan indeks latih/uji (`assert`).
3. Invariansi η Perceptron dicek numerik (`w/η` identik, prediksi identik).
4. ADALINE dicek silang dengan solusi least squares tertutup, dan Perceptron dengan `sklearn.linear_model.Perceptron`.
5. Notebook dieksekusi ulang dari awal dengan hasil yang sama, dan setiap angka di pembahasan dicocokkan dengan keluaran sel.

Setiap anggota bertanggung jawab memahami dan mampu menjelaskan seluruh isi repository.

---

## Kontribusi Anggota

Beban kerja dibagi **merata (masing-masing ±20%)**. Setiap anggota memegang satu bagian inti kode, satu bagian analisis, dan satu bagian luaran (laporan/presentasi/repository), sehingga semua anggota memahami data, aturan pembaruan, dan hasil.

| Anggota | Kode / eksperimen | Analisis & pembahasan | Laporan, presentasi & repository |
|---|---|---|---|
| Stefanus Westin Giovanno Mursito (71241073) | `prepare_dataset.py`, pemilihan fitur & pembuangan kolom bocor, kelas `Preprocessor`, split stratified | Kualitas data, EDA, PCA 2D | Laporan bagian 1–2 (masalah, dataset, kaitan Tugas 1); slide kasus & data |
| Giovanni Albert Harjanto (71241075) | Kelas `Perceptron`, perhitungan manual Perceptron + `assert` | Invariansi η, inisialisasi acak, pocket, pembanding scikit-learn | Laporan bagian 4 (Perceptron) & lampiran manual; slide Perceptron |
| Natalie Neysa Jessica Soesanto (71241079) | Kelas `AdalineGD`, perhitungan manual ADALINE + `assert` | Least squares, syarat stabilitas η, analisis divergensi tanpa standardisasi | Laporan bagian 4 (ADALINE) & 5 (rancangan eksperimen); slide ADALINE |
| Fransiska (71241123) | Rancangan & eksekusi 8 pelatihan, tabel epoch 10/30/50, kurva, confusion matrix | Metrik kelas positif, histogram jarak hyperplane, decision boundary 2 fitur | Laporan bagian 6 (hasil); slide hasil & demo notebook |
| Kevin Surya Saputra (71241150) | Seleksi model dengan data validasi, bobot & bias akhir, ekspor `figures/` & `results/` | Penelusuran artikel & kode rujukan, interpretasi bobot, keterbatasan | Laporan bagian 3, 7–9 (rujukan, pembahasan, kesimpulan); README & pengelolaan repository |

---

## Daftar Pustaka

1. Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review, 65*(6), 386–408. https://doi.org/10.1037/h0042519
2. Widrow, B., & Hoff, M. E. (1960). Adaptive switching circuits. *IRE WESCON Convention Record, Part 4*, 96–104. https://isl.stanford.edu/~widrow/papers/c1960adaptiveswitching.pdf
3. Pratama, N., Anrahvi, R., Tambal, A., & Singh, A. (2025). Classification of e-commerce shipping timeliness using supervised learning algorithm. *PREDATECS, 3*(1). https://doi.org/10.57152/predatecs.v3i1.1855
4. LeCun, Y., Bottou, L., Orr, G. B., & Müller, K.-R. (1998). Efficient BackProp. In *Neural Networks: Tricks of the Trade* (LNCS 1524, pp. 9–50). Springer. https://doi.org/10.1007/3-540-49430-8_2
5. Gallant, S. I. (1990). Perceptron-based learning algorithms. *IEEE Transactions on Neural Networks, 1*(2), 179–191. https://doi.org/10.1109/72.80230
6. Raschka, S., & Mirjalili, V. (2019). *Python Machine Learning* (3rd ed.). Packt. https://github.com/rasbt/python-machine-learning-book-3rd-edition
7. Raschka, S. (n.d.). Adaline: Adaptive Linear Neuron Classifier. mlxtend. https://rasbt.github.io/mlxtend/user_guide/classifier/Adaline/ (diakses 6 Oktober 2026)
8. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR, 12*, 2825–2830.
9. E-commerce Delivery and Shipping Data 2026. Kaggle (CC0 1.0). https://www.kaggle.com/datasets/datascikhan/e-commerce-delivery-and-shipping-data-2026
