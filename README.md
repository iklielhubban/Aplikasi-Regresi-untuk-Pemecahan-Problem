# Aplikasi Regresi untuk Pemecahan Problem

Repositori ini berisi implementasi model regresi linear dan polynomial untuk menganalisis hubungan antara durasi waktu belajar dan nilai ujian siswa. Data yang digunakan dalam proyek ini diambil dari [dataset Kaggle](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression).

## Struktur Folder

Aplikasi-Regresi-untuk-Pemecahan-Problem/
│
├── data/
│   └── student_performance.csv    # Dataset dari Kaggle
│
├── src/
│   └── regression_analysis.py      # Kode sumber Python untuk analisis regresi
│
├── testing/
│   └── test_regression_analysis.py # Kode sumber Python untuk testing
│
├── results/                        # Folder untuk menyimpan hasil plot
│
├── README.md                       # Deskripsi singkat tentang proyek
│
└── LICENSE                         # Lisensi proyek (opsional)

## Cara Menjalankan

1. **Pastikan telah menginstal pustaka yang diperlukan**:
    ```bash
    pip install pandas numpy scikit-learn matplotlib
    ```

2. **Jalankan script analisis regresi**:
    ```bash
    python src/regression_analysis.py
    ```

3. **Lihat hasil plot di folder `results`**:
    - `linear_regression_plot.png`: Hasil plot regresi linear
    - `polynomial_regression_plot.png`: Hasil plot regresi polynomial

## Menjalankan Pengujian

Untuk menjalankan pengujian, gunakan perintah berikut:
```bash
python -m unittest testing/test_regression_analysis.py
