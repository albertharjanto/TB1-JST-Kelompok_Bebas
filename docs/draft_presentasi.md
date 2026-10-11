# DRAF SLIDE PRESENTASI TUGAS BESAR 1 (IT6053)
## Klasifikasi Keterlambatan Pengiriman E-Commerce dengan Perceptron dan ADALINE

**Durasi Paparan:** ~8 menit + 4 menit tanya jawab  
**Kelompok:** Bebas  
**Anggota:** Stefanus (71241073), Giovanni (71241075), Natalie (71241079), Fransiska (71241123), Kevin (71241150)  

---

### Slide 1: Judul & Anggota (0:00 - 0:45)
- **Judul:** Analisis & Implementasi Klasifikasi Keterlambatan Pengiriman Logistik Menggunakan Perceptron dan ADALINE
- **Mata Kuliah:** Jaringan Syaraf Tiruan (IT6053) — FTI UKDW
- **Anggota & Peran:**
  - Stefanus Westin Giovanno Mursito (71241073) — Data, Pipeline & Preprocessing
  - Giovanni Albert Harjanto (71241075) — Perceptron & Manual Computation
  - Natalie Neysa Jessica Soesanto (71241079) — ADALINE & Delta Rule
  - Fransiska (71241123) — 8 Eksperimen & Evaluasi Metrik
  - Kevin Surya Saputra (71241150) — Analisis Teori, Rujukan & Pelaporan
- **URL Repo:** `https://github.com/albertharjanto/TB1-JST-Kelompok_Bebas`

---

### Slide 2: Latar Belakang Masalah & Formulasi Supervised Learning (0:45 - 1:45)
- **Konteks Bisnis:** Keterlambatan logistik e-commerce menurunkan retensi dan menimbulkan penalti biaya.
- **Formulasi:** Klasifikasi Biner Terawasi (*Supervised Learning*):
  - Label $+1$ (Positif): Terlambat (`late_delivery = Yes`)
  - Label $-1$ (Negatif): Tepat Waktu (`late_delivery = No`)
- **Dataset:** *E-commerce Delivery & Shipping Data 2026* (Kaggle, lisensi CC0 1.0).
- **Integritas Data & Pencegahan Kebocoran (*Leakage*):**
  - Total 50.000 pesanan, 0 missing values, 0 duplikat.
  - Membuang 8 kolom bocor yang baru tercatat pasca-pengiriman (mis. `actual_delivery_days`, `delivery_delay_days`, `delivery_attempts`, rating).
  - 14 fitur awal terpilih $\to$ dienkode menjadi **43 fitur numerik** (ordinal + one-hot drop-first).
- **Distribusi Kelas:** Terlambat 56,32% vs Tepat Waktu 43,68% (Baseline mayoritas: 56,32%).

---

### Slide 3: Model Neuron & Perbedaan Fundamental Pembelajaran (1:45 - 3:00)
- **Neuron Identik:** $z = \mathbf{w}^T \mathbf{x} + b$. Prediksi $\hat{y} = +1$ jika $z \ge 0$, else $-1$.
- **Perbedaan Inti Mekanisme Error:**
  - **Perceptron (Rosenblatt, 1958):**
    - Error Diskret: $\Delta = \eta(y - \hat{y}) \in \{0, \pm 2\eta\}$.
    - Pembaruan *online* per sampel hanya saat salah tebak. Buta terhadap margin/keyakinan.
  - **ADALINE (Widrow & Hoff, 1960):**
    - Error Kontinu Linear: $e = y - z$.
    - Fungsi Biaya Kuadratik: $J = \frac{1}{2n}\sum e_i^2$.
    - Pembaruan *Batch Gradient Descent*: $\mathbf{w} \leftarrow \mathbf{w} + \eta \frac{\mathbf{X}^T \mathbf{e}}{n}$.
    - Ambang kelas hanya diaktifkan saat inferensi/prediksi.
- **Validasi Perhitungan Manual:**
  - 2 sampel Perceptron & 2 iterasi batch ADALINE dihitung manual dan lolos uji `assert` identik dengan kode NumPy.

---

### Slide 4: Rancangan 8 Pelatihan & Pengendalian Eksperimen (3:00 - 4:15)
- **Prinsip Validitas Ilmiah:**
  - Split Stratified 80:20 ($n_{\text{train}} = 40.000, n_{\text{test}} = 10.000$), `random_state = 42`.
  - Imputer & Scaler **HANYA di-fit pada data latih**.
  - Inisialisasi nol ($\mathbf{w}_0 = \mathbf{0}, b_0 = 0$), 50 epoch, urutan data statis.
- **8 Konfigurasi Terkontrol:**
  1. Perceptron terstandardisasi: $\eta \in \{0.001, 0.01, 0.1\}$
  2. ADALINE terstandardisasi: $\eta \in \{0.001, 0.01, 0.1\}$
  3. Kontrol tanpa standardisasi: Perceptron $\eta = 0.01$ & ADALINE $\eta = 0.01$.

---

### Slide 5: Hasil Eksperimen Utama (4:15 - 5:30)
- **Tabel Kinerja Data Uji:**
  - **ADALINE $\eta=0.1$ (Terbaik):** Akurasi **79,43%**, Precision(+1) **82,40%**, Recall(+1) **80,72%**, F1(+1) **81,55%**, Loss $J = 0.2898$.
  - **ADALINE $\eta=0.01$:** Akurasi 76,76%, $J = 0.3578$.
  - **ADALINE $\eta=0.001$:** Akurasi 74,03%, $J = 0.4724$ (belum konvergen di 50 epoch).
  - **Perceptron $\eta \in \{0.001, 0.01, 0.1\}$:** Akurasi identik **74,60%**, F1(+1) 76,93%, 10.674 pembaruan di epoch 50.
  - **Perceptron Tanpa Standardisasi:** Akurasi jatuh ke **44,37%** (Recall 0,0217 — gagal total).
  - **ADALINE Tanpa Standardisasi:** **DIVERGEN NUMERIK ($J \to \infty$)** di epoch 31!

---

### Slide 6: Analisis & Pembahasan Mendalam (5:30 - 6:45)
- **1. Apakah Data Terpisah Linear? TIDAK!**
  - Perceptron tidak pernah konvergen ($\approx 10.670$ pembaruan konstan per epoch).
  - Batas optimum fisik Ordinary Least Squares adalah $J_{\text{LS}} = 0.2881$ (akurasi 79,47%).
  - Histogram $d = z/\|\mathbf{w}\|$ tumpang tindih padat di sekitar 0 (20,6% salah posisi).
- **2. Mengapa 3 $\eta$ Perceptron Identik?**
  - Pada inisialisasi nol & urutan statis, $\mathbf{w} = \eta \mathbf{v}$. Karena $\eta > 0$, $\text{sign}(\mathbf{w}^T\mathbf{x}) = \text{sign}(\mathbf{v}^T\mathbf{x})$. $\eta$ hanya menskalakan norma bobot ($10\times$).
- **3. Mengapa Tanpa Standardisasi ADALINE Meledak?**
  - Hessian $\lambda_{\max} = 9,47 \times 10^6 \implies$ batas stabil $\eta < 2,11 \times 10^{-7}$.
  - Pemakaian $\eta = 0.01$ melanggar batas stabilitas $47.000\times \implies$ faktor pembesaran $(1 - \eta\lambda_{\max})^2 \approx 8,96 \times 10^9$ per epoch $\implies J = 1.4 \times 10^{297}$ di epoch 30!
- **4. Fitur Penentu Keterlambatan:**
  - Positif (Pemicu Terlambat): Cuaca Salju ($+0.32$), Badai ($+0.30$), Hujan ($+0.21$), Jarak ($+0.19$).
  - Negatif (Mencegah Terlambat): Prioritas Pesanan ($-0.26$), Pengiriman Same Day/Express ($-0.08$).

---

### Slide 7: Keterbatasan & Langkah Lanjutan (6:45 - 7:30)
- **Keterbatasan:**
  - Model linear terbatas pada hyperplane tunggal — tidak mampu memodelkan interaksi silang fitur non-linear (mis. cuaca buruk $\times$ jarak jauh).
  - Akurasi 79,4% adalah batas matematis model linear pada data ini.
- **Rekomendasi Lanjutan:**
  - *Pocket Algorithm* (Gallant, 1990) untuk menstabilkan osilasi Perceptron (+1,4% akurasi).
  - *Multi-Layer Perceptron (MLP)* dengan lapisan tersembunyi non-linear (ReLU/Sigmoid) dan Backpropagation.
  - *Regresi Logistik* untuk kalibrasi probabilitas risiko keterlambatan.

---

### Slide 8: Kesimpulan & Penutup (7:30 - 8:00)
- Implementasi mandiri NumPy Perceptron & ADALINE sukses membedakan keterlambatan logistik.
- ADALINE $\eta=0.1$ terbukti paling unggul dan stabil mendekati solusi kuadrat terkecil analitis.
- Standardisasi fitur terbukti krusial secara matematis untuk mencegah divergensi numerik.
- Seluruh kode, gambar, tabel, dan notebook tersedia lengkap & dapat diulang (*reproducible*) di GitHub.
- Sesi Tanya Jawab (Q&A).
