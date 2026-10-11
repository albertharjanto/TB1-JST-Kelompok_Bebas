# LAPORAN TUGAS BESAR 1
## ANALISIS DAN IMPLEMENTASI KLASIFIKASI DATA MENGGUNAKAN PERCEPTRON DAN ADALINE

**Mata Kuliah:** Jaringan Syaraf Tiruan (IT6053) — 3 SKS  
**Program Studi:** Informatika, Fakultas Teknologi Informasi, Universitas Kristen Duta Wacana  
**Semester / Tahun Akademik:** Gasal 2026/2027  
**Nama Kelompok:** Kelompok Bebas  
**Repository GitHub:** `https://github.com/albertharjanto/TB1-JST-Kelompok_Bebas`  

---

### Identitas Anggota Kelompok

| No | Nama Lengkap | NIM | Pembagian Peran Utama |
|:--:|---|:---:|---|
| 1 | Stefanus Westin Giovanno Mursito | 71241073 | Data & Preprocessing Pipeline, EDA, Stratified Split |
| 2 | Giovanni Albert Harjanto | 71241075 | Implementasi Perceptron NumPy, Perhitungan Manual Perceptron |
| 3 | Natalie Neysa Jessica Soesanto | 71241079 | Implementasi ADALINE NumPy, Perhitungan Manual ADALINE |
| 4 | Fransiska | 71241123 | Eksperimen 8 Konfigurasi, Evaluasi Metrik, Visualisasi Grafik |
| 5 | Kevin Surya Saputra | 71241150 | Analisis Rujukan, Interpretasi Bobot, Pembahasan & Pelaporan |

---

## 1. Masalah dan Tujuan

### 1.1 Latar Belakang dan Rumusan Masalah
Dalam ekosistem logistik perdagangan elektronik (*e-commerce*), ketepatan waktu pengiriman merupakan indikator kinerja utama (*Key Performance Indicator*) yang menentukan tingkat kepuasan, loyalitas pelanggan, serta efisiensi operasional perusahaan pengiriman. Keterlambatan pengiriman paket dapat menimbulkan komplain, denda ganti rugi, hingga penurunan retensi konsumen. Mengingat volume pengiriman yang mencapai puluhan ribu paket per hari, proses pemilahan pesanan yang rentan mengalami keterlambatan secara manual menjadi tidak realistis.

Oleh karena itu, diperlukan pendekatan otomatisasi berbasis komputasi cerdas. Masalah ini dirumuskan sebagai **pembelajaran terawasi (*supervised learning*) klasifikasi biner**, di mana model bertugas memetakan atribut-atribut awal paket $\mathbf{x} \in \mathbb{R}^d$ ke dalam label keputusan diskret $y \in \{-1, +1\}$:
- **Kelas Positif ($y = +1$): Terlambat (*Late Delivery = Yes*)**
- **Kelas Negatif ($y = -1$): Tepat Waktu (*Late Delivery = No*)**

### 1.2 Tujuan Eksperimen
1. Membangun representasi matematis dari neuron tunggal berbobot dan berbias untuk kasus keterlambatan pengiriman logistik *e-commerce*.
2. Mengimplementasikan algoritma pembelajaran **Perceptron** (Rosenblatt, 1958) dan **ADALINE** (*Adaptive Linear Neuron* / Delta Rule; Widrow & Hoff, 1960) secara mandiri dari dasar menggunakan pustaka matriks NumPy tanpa memanfaatkan kelas estimator siap pakai.
3. Membuktikan secara analitis dan manual mekanisme adaptasi bobot untuk kedua model, lalu memvalidasi kesesuaiannya terhadap implementasi kode program.
4. Menjalankan 8 konfigurasi eksperimen terkontrol (variasi *learning rate* $\eta \in \{0.001, 0.01, 0.1\}$ serta pengaruh standardisasi fitur) untuk mengamati konvergensi, osilasi, maupun divergensi numerik.
5. Membuktikan separabilitas linear data berdasarkan perilaku riwayat pelatihan dan karakteristik geometris ruang fitur, serta menganalisis batas performa dari arsitektur *single-layer linear neuron*.

---

## 2. Dataset dan Karakteristik Data

### 2.1 Sumber Data dan Kesinambungan Tugas 1
Dataset yang digunakan merupakan kelanjutan dari analisis awal pada Tugas 1, yaitu **E-commerce Delivery and Shipping Data 2026** yang diperoleh dari repositori publik Kaggle (lisensi *Creative Commons CC0 1.0 Universal - Public Domain Dedication*). Data ini bebas dari batasan hak cipta dan tidak mengandung data identitas pribadi (*Personally Identifiable Information* / PII) pelanggan.

### 2.2 Jumlah Observasi dan Fitur Masukan
Dataset mentah memuat 50.000 observasi dan 30 kolom atribut. Untuk menghindari fenomena **kebocoran data (*data leakage*)**, seluruh fitur yang baru tercatat atau diketahui setelah proses pengiriman berlangsung disaring dan dibuang, meliputi: `actual_delivery_days`, `delivery_delay_days`, `delivery_status`, `tracking_status`, `return_requested`, `return_reason`, `customer_rating`, dan `delivery_attempts`.

Data bersih disimpan pada `dataset.csv` dengan ukuran 50.000 baris $\times$ 16 kolom (1 identifier `order_id`, 14 fitur operasional awal, dan 1 kolom target). Rincian fitur masukan meliputi:
1. **Fitur Numerik Kontinu (6 fitur):**
   - `product_weight_kg`: Berat fisik paket produk (0,10 kg – 52,19 kg; rata-rata 2,26 kg).
   - `order_value_usd`: Total nilai moneter pesanan ($10,00 – $3.530,73; rata-rata $177,92).
   - `distance_km`: Estimasi jarak transportasi gudang ke alamat penerima (40,0 km – 9.553,6 km; rata-rata 2.496,8 km).
   - `promised_delivery_days`: Janji durasi estimasi pengiriman saat checkout (0 – 15 hari; rata-rata 7,2 hari).
   - `shipping_cost_usd`: Tarif ongkos kirim ($4,05 – $500,00; rata-rata $120,52).
   - `warehouse_processing_hours`: Durasi penyortiran dan pengemasan di fasilitas pemenuhan gudang (1,0 – 48,0 jam; rata-rata 14,4 jam).
2. **Fitur Kategorikal Ordinal (2 fitur):**
   - `package_size`: Dimensi bungkusan (`Small` < `Medium` < `Large` < `Oversized`).
   - `order_priority`: Tingkat urgensi pesanan (`Low` < `Normal` < `High` < `Urgent`).
3. **Fitur Kategorikal Nominal (6 fitur):**
   - `customer_segment` (4 level: *Consumer, Enterprise, Premium, Small Business*).
   - `product_category` (12 level: *Automotive, Beauty, Books, Electronics, Fashion, Grocery, Health & Personal Care, Home & Kitchen, Office Supplies, Pet Supplies, Sports & Fitness, Toys*).
   - `shipping_method` (5 level: *Economy, Express, International, Same Day, Standard*).
   - `carrier` (8 vendor kurir: *BlueRoute, EagleCourier, FastTrack Logistics, GlobalExpress, ParcelPro, PrimeDelivery, SpeedyCargo, SwiftShip*).
   - `payment_method` (6 level: *Bank Transfer, Cash on Delivery, Credit Card, Debit Card, Digital Wallet, PayPal*).
   - `weather_condition` (6 kondisi atmosfer saat ekspedisi: *Clear, Cloudy, Extreme Heat, Rain, Snow, Storm*).

### 2.3 Kualitas Data dan Distribusi Kelas
- **Missing Value:** 0 nilai hilang (lengkap 100%).
- **Duplikasi:** 0 baris duplikat penuh, 0 duplikasi `order_id`. Tidak ditemukan konflik data (*fitur identik namun target berlawanan*).
- **Distribusi Kelas:**
  - Kelas Terlambat ($+1$): 28.162 observasi (**56,32%**)
  - Kelas Tepat Waktu ($-1$): 21.838 observasi (**43,68%**)
  - Proporsi menunjukkan kelas relatif seimbang dengan sedikit mayoritas pada kelas terlambat. Oleh karena itu, performa model dievaluasi tidak hanya menggunakan akurasi keseluruhan, tetapi juga metrik *Precision*, *Recall*, dan *F1-score* untuk kelas positif $+1$, serta dibandingkan dengan garis dasar probabilitas mayoritas (*Zero-R Baseline* = 56,32%).

---

## 3. Artikel Ilmiah dan Kode Pendukung

### 3.1 Penelusuran Artikel Ilmiah
1. **Rosenblatt, F. (1958).** *"The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"*, *Psychological Review*, 65(6), 386–408.  
   *Kontribusi:* Menjadi landasan matematis aturan pembaruan berbasis kesalahan klasifikasi biner dan pembuktian *Perceptron Convergence Theorem* (teorema yang menyatakan bahwa jika data terpisah linear, Perceptron pasti konvergen dalam jumlah iterasi berhingga).
2. **Widrow, B., & Hoff, M. E. (1960).** *"Adaptive Switching Circuits"*, *IRE WESCON Convention Record*, Part 4, 96–104.  
   *Kontribusi:* Merumuskan prinsip kuadrat terkecil (*Least Mean Squares* / LMS) dan Delta Rule pada arsitektur ADALINE, di mana fungsi objektif kontinu $J(w, b)$ diturunkan terhadap vektor bobot untuk menghasilkan pembaruan berbasis gradien keluaran linear.
3. **Pratama, N., Anrahvi, R., Tambal, A., & Singh, A. (2025).** *"Classification of E-Commerce Shipping Timeliness Using Supervised Learning Algorithm"*, *PREDATECS*, 3(1), DOI: `10.57152/predatecs.v3i1.1855`.  
   *Kontribusi:* Menjadi rujukan domain masalah ketepatan waktu ekspedisi belanja daring. Eksperimen mereka menunjukkan model non-linear Support Vector Machine (SVM) menghasilkan akurasi berkisar 66,35%, mengonfirmasi bahwa kasus logistik memiliki derau lingkungan dan non-linearitas intrinsik yang menantang.
4. **LeCun, Y., Bottou, L., Orr, G. B., & Müller, K.-R. (1998).** *"Efficient BackProp"*, Springer LNCS 1524.  
   *Kontribusi:* Menjelaskan urgensi normalisasi/standardisasi fitur input agar matriks Hessian memiliki rasio nilai eigen (*condition number*) yang seimbang sehingga gradien tidak mengalami osilasi liar atau divergensi numerik.
5. **Gallant, S. I. (1990).** *"Perceptron-based Learning Algorithms"*, *IEEE Transactions on Neural Networks*, 1(2), 179–191.  
   *Kontribusi:* Menjadi dasar usulan perbaikan arsitektur (*Pocket Algorithm*) saat menghadapi dataset yang tidak dapat dipisahkan secara linear (*non-linearly separable*).

### 3.2 Sumber Kode dan Dokumentasi Teknis
- **Kode Rujukan:** Bab 2 repositori publik buku *Python Machine Learning* (Raschka & Mirjalili, 2019) dan dokumentasi pustaka *mlxtend* (`rasbt.github.io/mlxtend`).
- **Modifikasi dan Adaptasi:**
  1. Aturan pembaruan Perceptron dimodifikasi menjadi formula resmi panduan IT6053: $\Delta = \eta (y - \hat{y})$ sehingga besar pembaruan bernilai $\pm 2\eta$ saat terjadi kesalahan klasifikasi.
  2. Fungsi batas keputusan disesuaikan menjadi fungsi tanda non-negatif: $\hat{y} = +1$ jika $z \ge 0$, dan $-1$ jika $z < 0$.
  3. ADALINE diimplementasikan menggunakan metode *Full-Batch Gradient Descent*, di mana fungsi biaya kuadratik rata-rata $J = \frac{1}{2n}\sum e_i^2$ dicatat secara konsisten tepat **setelah** parameter diperbarui tiap epoch.
  4. Penambahan mekanisme *early termination* deteksi divergensi numerik (`NaN` / `Inf`) untuk menghentikan loop pelatihan dan mencatat epoch kejadian secara persis.
  5. Seluruh modul dipaketkan dalam kelas mandiri berstandar OOP Python yang hanya bergantung pada array NumPy.

---

## 4. Model Neuron dan Aturan Pembelajaran

### 4.1 Persamaan Neuron Tunggal
Baik Perceptron maupun ADALINE menggunakan komponen komputasi linear neuron yang identik:
$$z = \mathbf{w}^T \mathbf{x} + b = \sum_{j=1}^{d} w_j x_j + b$$
di mana $\mathbf{w} = [w_1, w_2, \dots, w_d]^T$ adalah vektor bobot sinaptik, $b$ adalah skalar bias, $\mathbf{x}$ adalah vektor fitur input, dan $z$ adalah net input (aktivasi linear). Keputusan akhir prediksi biner $\hat{y}$ ditentukan oleh fungsi ambang (*Heaviside step / sign*):
$$\hat{y} = \begin{cases} +1, & \text{jika } z \ge 0 \\ -1, & \text{jika } z < 0 \end{cases}$$

### 4.2 Perbedaan Aturan Pembelajaran dan Fungsi Objektif

```
Alur Perceptron:
x ---> [ w^T x + b = z ] ---> [ Threshold sign(z) ] ---> y_hat ---> Error: (y - y_hat) ---> Update w, b
                                                                 (diskret: 0, +2, -2)

Alur ADALINE:
x ---> [ w^T x + b = z ] -----------------------------------------> Error: e = (y - z)  ---> Update w, b
             |                                                   (kontinu linear)
             v
      [ Threshold sign(z) ] ---> y_hat (hanya saat prediksi / evaluasi akhir)
```

1. **Perceptron (Rosenblatt Rule):**
   - Pembelajaran bersifat stokastik per sampel (*online learning*).
   - Sinyal galat dihitung dari selisih keluaran terkuantisasi diskret:
     $$\Delta = \eta (y_i - \hat{y}_i) \in \{0, +2\eta, -2\eta\}$$
   - Pembaruan bobot dan bias:
     $$\mathbf{w} \leftarrow \mathbf{w} + \Delta \mathbf{x}_i, \quad b \leftarrow b + \Delta$$
   - Jika $\hat{y}_i = y_i$, nilai $\Delta = 0$ (tidak ada perubahan). Model hanya bereaksi terhadap misklasifikasi tanpa mempedulikan margin keyakinan.
2. **ADALINE (Delta Rule / LMS):**
   - Pembelajaran berbasis optimisasi fungsi kerugian kontinu:
     $$J(\mathbf{w}, b) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - z_i)^2 = \frac{1}{2n} \|\mathbf{y} - (\mathbf{X}\mathbf{w} + b\mathbf{1})\|^2$$
   - Error $e_i = y_i - z_i$ diukur langsung dari keluaran aktivasi linear $z_i$, bukan label ambang.
   - Menggunakan *Full-Batch Gradient Descent*:
     $$\nabla_{\mathbf{w}} J = -\frac{1}{n}\mathbf{X}^T \mathbf{e}, \quad \frac{\partial J}{\partial b} = -\frac{1}{n}\sum_{i=1}^{n} e_i = -\overline{\mathbf{e}}$$
     $$\mathbf{w} \leftarrow \mathbf{w} + \eta \frac{\mathbf{X}^T \mathbf{e}}{n}, \quad b \leftarrow b + \eta \, \text{mean}(\mathbf{e})$$

### 4.3 Verifikasi Perhitungan Manual

Untuk memastikan ketepatan algoritma, dilakukan perhitungan manual langkah demi langkah pada subset mini 4 sampel data latih terstandardisasi dengan 3 fitur kontinu: $x_1$ (`distance_km`), $x_2$ (`promised_delivery_days`), $x_3$ (`warehouse_processing_hours`), serta laju pembelajaran $\eta = 0.1$, inisialisasi awal $\mathbf{w}_0 = [0, 0, 0]^T$ dan $b_0 = 0$.

#### Data Sampel Mini
- **$S_1$ (urutan latih 5):** $\mathbf{x} = [-0.1654, +1.0307, -0.7766]^T$, target $y = -1$
- **$S_2$ (urutan latih 0):** $\mathbf{x} = [-0.4735, -1.1432, -0.8371]^T$, target $y = +1$
- **$S_3$ (urutan latih 6):** $\mathbf{x} = [-0.0029, -0.3280, -0.3228]^T$, target $y = -1$
- **$S_4$ (urutan latih 1):** $\mathbf{x} = [+0.9708, -0.0562, +0.8472]^T$, target $y = +1$

#### 1. Perhitungan Manual Perceptron (2 Sampel Pertama)
- **Langkah 1 (Sampel $S_1$, $y = -1$):**
  $$z = 0(-0.1654) + 0(1.0307) + 0(-0.7766) + 0 = 0.0000 \ge 0 \implies \hat{y} = +1$$
  $$\text{Terjadi salah klasifikasi! } \Delta = 0.1 \times (-1 - (+1)) = -0.2000$$
  $$\mathbf{w}^{(1)} = [0, 0, 0]^T + (-0.2000)[-0.1654, 1.0307, -0.7766]^T = [+0.0331, -0.2061, +0.1553]^T$$
  $$b^{(1)} = 0 + (-0.2000) = -0.2000$$
- **Langkah 2 (Sampel $S_2$, $y = +1$):**
  $$z = (0.0331)(-0.4735) + (-0.2061)(-1.1432) + (0.1553)(-0.8371) - 0.2000 = -0.1100 < 0 \implies \hat{y} = -1$$
  $$\text{Terjadi salah klasifikasi! } \Delta = 0.1 \times (+1 - (-1)) = +0.2000$$
  $$\mathbf{w}^{(2)} = \mathbf{w}^{(1)} + (+0.2000)[-0.4735, -1.1432, -0.8371]^T = [-0.0616, -0.4348, -0.0121]^T$$
  $$b^{(2)} = -0.2000 + 0.2000 = +0.0000$$
- *Pengecekan Kode:* Pemanggilan `Perceptron(eta=0.1, n_epochs=1).fit(X[:2], y[:2])` menghasilkan $\mathbf{w} = [-0.0616, -0.4348, -0.0121]^T$ dan $b = 0.0000$ (identik sempurna, terverifikasi via `assert`).

#### 2. Perhitungan Manual ADALINE (2 Iterasi Batch, $n = 4$)
- **Kondisi Awal (Epoch 0):** $\mathbf{w}^{(0)} = [0, 0, 0]^T, b^{(0)} = 0$.  
  $z_i = 0 \implies e_i = y_i - 0 = y_i$.  
  $J^{(0)} = \frac{1}{2(4)}[(-1)^2 + (1)^2 + (-1)^2 + (1)^2] = \frac{4}{8} = 0.5000$.
- **Iterasi 1:**
  - Galat: $\mathbf{e} = [-1.0, +1.0, -1.0, +1.0]^T$.
  - Gradien bobot: $\frac{1}{4}\mathbf{X}^T\mathbf{e} = [+0.1664, -0.4755, +0.2774]^T$.
  - Rata-rata galat: $\text{mean}(\mathbf{e}) = \frac{-1 + 1 - 1 + 1}{4} = 0.0000$.
  - Pembaruan:  
    $\mathbf{w}^{(1)} = \mathbf{w}^{(0)} + 0.1 \times [+0.1664, -0.4755, +0.2774]^T = [+0.0166, -0.0476, +0.0277]^T$  
    $b^{(1)} = 0 + 0.1(0.0000) = 0.0000$
  - Biaya sesudah pembaruan: $J^{(1)} = 0.4679$ (MSE = $0.9358$).
- **Iterasi 2:**
  - Net input baru: $z = [-0.0733, +0.0233, +0.0066, +0.0423]^T$.
  - Galat baru: $\mathbf{e} = [-0.9267, +0.9767, -1.0066, +0.9577]^T$.
  - Gradien: $\frac{1}{4}\mathbf{X}^T\mathbf{e} = [+0.1558, -0.4489, +0.2596]^T$; $\text{mean}(\mathbf{e}) = +0.0003$.
  - Pembaruan:  
    $\mathbf{w}^{(2)} = [+0.0322, -0.0924, +0.0537]^T$  
    $b^{(2)} = +0.0000$
  - Biaya sesudah pembaruan: $J^{(2)} = 0.4394$ (MSE = $0.8789$).
- *Pengecekan Kode:* Pemanggilan `AdalineGD(eta=0.1, n_epochs=2).fit(X, y)` menghasilkan bobot, bias, dan riwayat biaya `cost_ = [0.5000, 0.4679, 0.4394]` persis identik dengan kalkulasi analitis (terverifikasi via `assert`).

---

## 5. Penyiapan Data dan Rancangan Eksperimen

### 5.1 Pembagian Data Bebas Kebocoran
Data dibagi dengan rasio **80% data latih (40.000 sampel)** dan **20% data uji (10.000 sampel)** menggunakan metode *Stratified Train-Test Split* (`random_state = 42`). Stratifikasi memastikan distribusi label pada kedua subset persis 56,32% kelas positif ($+1$) dan 43,68% kelas negatif ($-1$). Irisan indeks antara data latih dan data uji dipastikan 0 (`len(intersect) == 0`).

### 5.2 Skema Preprocessing
Dibangun kelas `Preprocessor` mandiri dengan kaidah kepatuhan ketat:
1. **Fit hanya pada Data Latih:** Nilai median numerik, modus kategorikal, kamus kategori *one-hot*, serta nilai mean ($\mu_j$) dan standar deviasi ($\sigma_j$) dipelajari secara eksklusif hanya dari `df_train`.
2. **Transformasi pada Data Latih & Uji:** Nilai statistik tersebut diterapkan secara deterministik ke `df_test` tanpa melakukan kalkulasi ulang pada data uji. Terbukti mean data uji tidak persis nol ($|\mu_{\text{test}}| \le 0.025$), membuktikan scaler tidak mengalami kebocoran.
3. **Encoding:**
   - 2 fitur ordinal dipetakan ke urutan numerik diskret $\{0, 1, 2, 3\}$.
   - 6 fitur nominal diubah melalui *One-Hot Encoding* dengan skema *Drop-First* (kategori pertama secara leksikografis dijadikan basis acuan). Total dimensi fitur akhir adalah **43 fitur**.
   - Standardisasi $z$-score dilakukan pada data terstandardisasi, sedangkan pada data tanpa standardisasi fitur dibiarkan pada skala aslinya.

### 5.3 Rancangan Eksperimen 8 Konfigurasi
Sesuai panduan tugas, ditetapkan 8 konfigurasi pelatihan sebelum menguji data uji:
- **Konfigurasi 1–3:** Perceptron terstandardisasi ($\eta \in \{0.001, 0.01, 0.1\}$, 50 epoch, inisialisasi nol $\mathbf{w}_0 = \mathbf{0}, b_0 = 0$).
- **Konfigurasi 4–6:** ADALINE terstandardisasi ($\eta \in \{0.001, 0.01, 0.1\}$, 50 epoch, inisialisasi nol $\mathbf{w}_0 = \mathbf{0}, b_0 = 0$).
- **Konfigurasi 7:** Perceptron tanpa standardisasi ($\eta = 0.01$, 50 epoch).
- **Konfigurasi 8:** ADALINE tanpa standardisasi ($\eta = 0.01$, 50 epoch).

---

## 6. Hasil Eksperimen

### 6.1 Ringkasan Kinerja Seluruh Konfigurasi
Evaluasi kinerja model pada data lati ($n = 40.000$) dan data uji ($n = 10.000$) disajikan pada tabel berikut:

| Konfigurasi Model | $\eta$ | Standardisasi | Epoch Selesai | Akurasi Latih | Akurasi Uji | Precision (+1) | Recall (+1) | F1-Score (+1) | Salah Latih Akhir | Pembaruan Ep. 50 | Loss Akhir ($J$) | MSE Akhir |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Perceptron** | 0.001 | Ya | 50 | 0.7443 | 0.7460 | 0.7874 | 0.7521 | 0.7693 | 10.229 | 10.674 | — | — |
| **Perceptron** | 0.010 | Ya | 50 | 0.7443 | 0.7460 | 0.7874 | 0.7521 | 0.7693 | 10.229 | 10.674 | — | — |
| **Perceptron** | 0.100 | Ya | 50 | 0.7443 | 0.7460 | 0.7874 | 0.7521 | 0.7693 | 10.229 | 10.674 | — | — |
| **ADALINE** | 0.001 | Ya | 50 | 0.7428 | 0.7403 | 0.7958 | 0.7250 | 0.7587 | 10.286 | — | 0.4724 | 0.9448 |
| **ADALINE** | 0.010 | Ya | 50 | 0.7683 | 0.7676 | 0.8096 | 0.7679 | 0.7882 | 9.268 | — | 0.3578 | 0.7157 |
| **ADALINE (Terbaik)** | **0.100** | **Ya** | **50** | **0.7959** | **0.7943** | **0.8240** | **0.8072** | **0.8155** | **8.163** | — | **0.2898** | **0.5797** |
| **Perceptron** | 0.010 | Tidak | 50 | 0.4430 | 0.4437 | 0.6971 | 0.0217 | 0.0420 | 22.281 | 17.447 | — | — |
| **ADALINE** | 0.010 | Tidak | **30\*** | 0.4368 | 0.4368 | 0.0000 | 0.0000 | 0.0000 | 22.530 | — | $1.43 \times 10^{297}$ | $2.85 \times 10^{297}$ |

*\*Keterangan ADALINE tanpa standardisasi: Divergen numerik ($J = \infty$) pada epoch 31. Pelatihan dihentikan otomatis dan bobot epoch 30 disimpan.*  
*Garis Dasar Mayoritas (Zero-R Baseline): Akurasi = 0.5632, Recall(+1) = 1.0000, Precision(+1) = 0.5632.*

### 6.2 Riwayat Pelatihan pada Epoch ke-10, ke-30, dan ke-50

| Konfigurasi | Jumlah Pembaruan (Perceptron) | | | Fungsi Kerugian $J$ (ADALINE) | | | Akurasi Data Latih | | |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | **Ep. 10** | **Ep. 30** | **Ep. 50** | **Ep. 10** | **Ep. 30** | **Ep. 50** | **Ep. 10** | **Ep. 30** | **Ep. 50** |
| Perceptron $\eta=0.001$ (std) | 10.642 | 10.673 | 10.674 | — | — | — | 0.7531 | 0.7501 | 0.7443 |
| Perceptron $\eta=0.01$ (std) | 10.642 | 10.673 | 10.674 | — | — | — | 0.7531 | 0.7501 | 0.7443 |
| Perceptron $\eta=0.1$ (std) | 10.642 | 10.673 | 10.674 | — | — | — | 0.7531 | 0.7501 | 0.7443 |
| ADALINE $\eta=0.001$ (std) | — | — | — | 0.4940 | 0.4828 | 0.4724 | 0.7400 | 0.7415 | 0.7428 |
| ADALINE $\eta=0.01$ (std) | — | — | — | 0.4493 | 0.3897 | 0.3578 | 0.7460 | 0.7588 | 0.7683 |
| ADALINE $\eta=0.1$ (std) | — | — | — | 0.3210 | 0.2945 | 0.2898 | 0.7807 | 0.7938 | 0.7959 |
| Perceptron $\eta=0.01$ (raw) | 18.295 | 17.774 | 17.447 | — | — | — | 0.4356 | 0.4374 | 0.4430 |
| ADALINE $\eta=0.01$ (raw) | — | — | — | $1.28 \times 10^{98}$ | $1.43 \times 10^{297}$ | Divergen | 0.4368 | 0.4368 | Divergen |

### 6.3 Matriks Kebingungan (*Confusion Matrix*) Data Uji
Matriks kebingungan disusun dengan urutan label baris aktual dan kolom prediksi: `[Index 0: Tepat Waktu (-1), Index 1: Terlambat (+1)]`.

| Konfigurasi | True Negative (TN) | False Positive (FP) | False Negative (FN) | True Positive (TP) | Total Sampel |
|---|:---:|:---:|:---:|:---:|:---:|
| Perceptron $\eta \in \{0.001, 0.01, 0.1\}$ (std) | 3.224 | 1.144 | 1.396 | 4.236 | 10.000 |
| ADALINE $\eta=0.001$ (std) | 3.320 | 1.048 | 1.549 | 4.083 | 10.000 |
| ADALINE $\eta=0.01$ (std) | 3.351 | 1.017 | 1.307 | 4.325 | 10.000 |
| **ADALINE $\eta=0.1$ (std)** | **3.397** | **971** | **1.086** | **4.546** | **10.000** |
| Perceptron $\eta=0.01$ (tanpa std) | 4.315 | 53 | 5.510 | 122 | 10.000 |
| ADALINE $\eta=0.01$ (tanpa std) | 4.368 | 0 | 5.632 | 0 | 10.000 |

### 6.4 Analisis Bobot dan Bias Akhir
Perbandingan parameter bobot pada dua konfigurasi terstandardisasi paling representatif (`results/bobot_bias_akhir.csv`):
- Bias akhir Perceptron ($\eta = 0.01$): $b = +0.2000$; norma bobot $\|\mathbf{w}\| = 1.0926$.
- Bias akhir ADALINE ($\eta = 0.1$): $b = +0.1258$; norma bobot $\|\mathbf{w}\| = 0.6558$.
- **Kemiripan Kosinus Arah Bobot:** $\cos(\mathbf{w}_{\text{Perc}}, \mathbf{w}_{\text{ADALINE}}) = 0.8591$.
- **Fitur Paling Berpengaruh (Urutan Magnitudo $|w_j|$ ADALINE):**
  1. `weather_condition=Snow` ($w = +0.3179$): Kondisi salju merupakan pemicu keterlambatan terkuat.
  2. `weather_condition=Storm` ($w = +0.2990$): Cuaca badai meningkatkan risiko keterlambatan secara tajam.
  3. `order_priority` ($w = -0.2584$): Semakin tinggi prioritas pesanan, semakin kecil peluang keterlambatan (bobot negatif mendorong ke kelas $-1$).
  4. `weather_condition=Rain` ($w = +0.2062$): Hujan lebat menghambat distribusi logistik.
  5. `distance_km` ($w = +0.1860$): Semakin jauh jarak gudang ke pelanggan, peluang keterlambatan semakin membesar.
  6. `shipping_method=Same Day` ($w = -0.0816$) & `Express` ($w = -0.0648$): Mengurangi peluang keterlambatan secara konsisten.

---

## 7. Pembahasan dan Keterbatasan

### 7.1 Pembuktian Separabilitas Linear Data
Berdasarkan bukti empiris dan analitis, **data pengiriman e-commerce ini TIDAK DAPAT dipisahkan secara linear**:
1. **Kegagalan Konvergensi Perceptron:** Sesuai *Perceptron Convergence Theorem*, jika suatu dataset *linearly separable*, Perceptron bergaransi mencapai pembaruan 0 dalam epoch terhingga. Pada eksperimen ini, kurva pembaruan Perceptron berfluktuasi datar di angka $\approx 10.670$ pembaruan per epoch (26,7% data latih salah diklasifikasi di tiap epoch) hingga epoch ke-50 tanpa ada tren menuju nol (`figures/04_kurva_perceptron.png`).
2. **Residual Least Squares Positif Jauh di Atas Nol:** Solusi analitis *Ordinary Least Squares* (optimum global hyperplane linear) memiliki batas bawah kerugian $J_{\text{LS}} = 0.2881$ dengan akurasi 0.7947. Nilai residual kuadratik sebesar 0.2881 membuktikan bahwa tidak ada bidang datar linear dalam $\mathbb{R}^{43}$ yang dapat memisahkan kedua kelas tanpa kesalahan.
3. **Tumpang Tindih Margin Keputusan:** Histogram proyeksi jarak bertanda $d = \frac{\mathbf{w}^T\mathbf{x} + b}{\|\mathbf{w}\|}$ (`figures/09_histogram_jarak_hyperplane.png`) menunjukkan kedua kelas saling bertumpuk tebal di sekitar $d = 0$. Pada model ADALINE terbaik, 20,6% sampel data uji tetap berada di sisi yang salah dari batas keputusan. Visualisasi batas 2 fitur (`figures/10_decision_boundary_2fitur.png`) mempertegas bercampurnya sebaran titik kedua kelas.

### 7.2 Perbedaan Mekanisme Pembelajaran Error
- **Perceptron** bersifat buta terhadap margin. Evaluasi diskret $y_i - \hat{y}_i$ menyebabkan pembaruan bernilai konstan $\pm 2\eta$ tidak peduli apakah sampel berada sangat dekat di dekat batas keputusan ($z = +0.01$) atau berada jauh di pedalaman ($z = +10.0$). Akibatnya, pada data tak terpisah, vektor bobot Perceptron berosilasi liar ditarik bolak-balik oleh pencilan (*outliers*).
- **ADALINE** meminimalkan permukaan kesalahan mangkuk kuadratik kontinu. Sinyal galat $e_i = y_i - z_i$ memperhitungkan besaran deviasi metrik. Pembaruan secara *full-batch* merata-ratakan gradien seluruh 40.000 sampel, sehingga menghasilkan lintasan parameter yang mulus dan konvergen stabil menuju minimum kuadrat terkecil global.

### 7.3 Pengaruh Learning Rate, Epoch, dan Standardisasi
1. **Invariansi Learning Rate pada Perceptron:**
   Pada inisialisasi nol $\mathbf{w}_0 = \mathbf{0}, b_0 = 0$ dan urutan data statis, perubahan $\eta$ **hanya mengubah magnitudo bobot tanpa mengubah arah hyperplane**:
   $$\mathbf{w}_t(\eta) = \eta \sum_{k \in \mathcal{M}_t} (y_k - \hat{y}_k)\mathbf{x}_k = \eta \mathbf{v}_t$$
   Karena $z = \eta(\mathbf{v}_t^T\mathbf{x} + c_t)$ dan $\eta > 0$, maka $\text{sign}(z) = \text{sign}(\mathbf{v}_t^T\mathbf{x} + c_t)$. Akibatnya urutan misklasifikasi identik 100%. Terbukti norma bobot $\|\mathbf{w}\|$ tepat membesar $10\times$ (0.1093 $\to$ 1.0926 $\to$ 10.9263) dengan rasio $\mathbf{w}/\eta$ identik sempurna (`assert np.allclose`). Namun, ketika diuji dengan inisialisasi acak kecil $\mathcal{N}(0, 0.01)$, invariansi ini gugur dan masing-masing $\eta$ menghasilkan akurasi latih berbeda (0.7474, 0.7370, 0.7477) karena rasio $\frac{2\eta}{\|\mathbf{w}_0\|}$ berubah.
2. **Efektivitas Learning Rate pada ADALINE:**
   Laju konvergensi ADALINE diatur oleh kondisi nilai eigen matriks Hessian $\mathbf{H} = \frac{1}{n}\mathbf{A}^T\mathbf{A}$. Pada data terstandardisasi, $\lambda_{\max}(\mathbf{H}) = 3.081$, memberikan batas kestabilan teoritis $\eta < \frac{2}{\lambda_{\max}} = 0.6491$. Nilai $\eta = 0.1$ ($\eta \lambda_{\max} \approx 0.308$) memberikan laju penurunan tercepat dan paling efisien, mencapai $J = 0.2898$ (hampir menyentuh optimum analitis $J_{\text{LS}} = 0.2881$). Sementara $\eta = 0.001$ terlalu lambat dan belum konvergen pada epoch 50.
3. **Bencana Tanpa Standardisasi:**
   - **ADALINE Divergen:** Tanpa standardisasi, fitur `distance_km` (skala ribuan) mendominasi variansi data, menyebabkan $\lambda_{\max}(\mathbf{H}) = 9.467 \times 10^6$. Batas kestabilan turun drastis menjadi $\eta < 2.113 \times 10^{-7}$. Pemakaian $\eta = 0.01$ melanggar batas stabilitas sebesar $47.000\times$, mengakibatkan kuadrat faktor pengali $(1 - \eta \lambda_{\max})^2 \approx 8.96 \times 10^9$ per epoch. Nilai $J$ meledak eksponensial dari $1.28 \times 10^{98}$ di epoch 10 menjadi $1.43 \times 10^{297}$ di epoch 30, hingga `OverflowError` ($J = \infty$) di epoch 31.
   - **Perceptron Kolaps:** Skala `distance_km` mendominasi nilai $z$ (kontribusi rata-rata $|w_j x_j| \approx 39.087$), menenggelamkan kontribusi fitur biner cuaca dan metode pengiriman. Model memprediksi hampir seluruh sampel sebagai $-1$ (hanya memprediksi 175 sampel positif dari 10.000 sampel uji), menghasilkan recall katastropik 0.0217 dan akurasi 0.4437 (di bawah tebakan mayoritas).

### 7.4 Analisis Kesamaan Akurasi
Akurasi yang bernilai serupa **TIDAK MENUNJUKKAN** proses pembelajaran yang sama:
- Perceptron $\eta = 0.01$ (akurasi uji 0.7460) dan ADALINE $\eta = 0.001$ (akurasi uji 0.7403) memiliki nilai akurasi berdekatan, namun ADALINE mencapai titik tersebut melalui pergerakan gradien mulus dengan *Precision* lebih tinggi (0.7958 vs 0.7874), sedangkan Perceptron mencapainya melalui lebih dari 530.000 kali lompatan parameter yang fluktuatif.
- Pada pengujian seleksi data validasi independen (20% dari data latih), Perceptron mengalami penurunan performa ke akurasi 0.6759, sedangkan ADALINE tetap stabil unggul di angka 0.7996.

### 7.5 Keterbatasan dan Langkah Pengembangan
- **Keterbatasan:**
  1. Arsitektur linear berdimensi tunggal tidak mampu menangkap korelasi silang interaksi non-linear (misalnya: kombinasi badai ekstrem bersamaan dengan pengiriman jarak jauh antar pulau).
  2. Batas akurasi $79.43\%$ menyiratkan sekitar $1$ dari $5$ pengiriman tetap salah diprediksi.
  3. Perceptron rentan terhadap urutan penghentian epoch (*stopping epoch*). Bobot di epoch 18 sebenarnya memiliki performa latih lebih tinggi (0.7575) daripada epoch 50 (0.7443).
- **Langkah Pengembangan:**
  1. **Pocket Algorithm (Gallant, 1990):** Menyimpan bobot terbaik sepanjang iterasi latih untuk menstabilkan Perceptron (+1,4% peningkatan akurasi pada data uji).
  2. **Multi-Layer Perceptron (MLP):** Menambahkan neuron lapisan tersembunyi (*hidden layer*) dengan fungsi aktivasi non-linear (ReLU/Sigmoid) dan algoritma *Backpropagation* untuk membentuk batas keputusan non-linear berlapis.
  3. **Regresi Logistik:** Mengganti fungsi identitas ADALINE dengan fungsi Sigmoid dan *Binary Cross-Entropy Loss* untuk mengestimasi peluang keterlambatan secara probabilistik terkalibrasi.

---

## 8. Kesimpulan

Berdasarkan keseluruhan implementasi mandiri, verifikasi analitis, dan eksekusi 8 eksperimen pelatihan klasifikasi keterlambatan pengiriman e-commerce:
1. Model neuron tunggal berhasil mempelajari pola keterlambatan paket logistik, di mana konfigurasi **ADALINE terstandardisasi dengan $\eta = 0.1$** terpilih sebagai model terbaik dengan performa data uji: **Akurasi 79,43%**, **Precision(+1) 82,40%**, **Recall(+1) 80,72%**, dan **F1-Score(+1) 81,55%**, secara signifikan mengungguli garis dasar mayoritas (56,32%).
2. Data secara definitif **tidak terpisah linear**, dibuktikan oleh kegagalan konvergensi pembaruan Perceptron (tetap $\approx 10.670$ pembaruan/epoch), residual kuadrat terkecil $J_{\text{LS}} = 0.2881$, serta tumpang tindih sebaran margin keputusan. Akurasi $\approx 79,4\%$ merupakan batas optimum fisik dari representasi linear (*linear hyperplane bound*).
3. Sinyal galat kontinu ADALINE terbukti jauh lebih superior dan stabil dibandingkan galat diskret Perceptron dalam memandu arah bobot menuju optimum kuadrat terkecil global ($\cos(\mathbf{w}, \mathbf{w}_{\text{LS}}) = 0.9913$).
4. Invariansi $\eta$ pada Perceptron berinisialisasi nol terbukti secara matematis dan numerik hanya mengubah norma skala bobot tanpa menggeser orientasi bidang keputusan.
5. Standardisasi fitur merupakan prasyarat mutlak: tanpanya, ADALINE mengalami ledakan numerik ($J \to \infty$) akibat pelanggaran batas nilai eigen Hessian, sedangkan Perceptron terdistorsi oleh fitur berskala besar hingga kehilangan daya diskriminasi.

---

## 9. Daftar Pustaka

1. Gallant, S. I. (1990). Perceptron-based learning algorithms. *IEEE Transactions on Neural Networks*, 1(2), 179–191. https://doi.org/10.1109/72.80230
2. LeCun, Y., Bottou, L., Orr, G. B., & Müller, K.-R. (1998). Efficient BackProp. Dalam *Neural Networks: Tricks of the Trade* (LNCS 1524, hlm. 9–50). Springer. https://doi.org/10.1007/3-540-49430-8_2
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
4. Pratama, N., Anrahvi, R., Tambal, A., & Singh, A. (2025). Classification of E-Commerce Shipping Timeliness Using Supervised Learning Algorithm. *Public Research Journal of Engineering, Data Technology and Computer Science (PREDATECS)*, 3(1). https://doi.org/10.57152/predatecs.v3i1.1855
5. Raschka, S., & Mirjalili, V. (2019). *Python Machine Learning* (Edisi ke-3). Packt Publishing.
6. Raschka, S. (tanpa tahun). *Adaline: Adaptive Linear Neuron Classifier*. Dokumentasi mlxtend. https://rasbt.github.io/mlxtend/user_guide/classifier/Adaline/
7. Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review*, 65(6), 386–408. https://doi.org/10.1037/h0042519
8. Widrow, B., & Hoff, M. E. (1960). Adaptive switching circuits. *IRE WESCON Convention Record*, Part 4, 96–104.
