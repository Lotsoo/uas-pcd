# 🚗 Klasifikasi Mobil vs Motor 🚲

Aplikasi web klasifikasi gambar sederhana namun kuat yang dibangun dengan **Streamlit**. Proyek ini mengidentifikasi apakah sebuah gambar berisi **Mobil (Car)** atau **Motor (Bike)** menggunakan berbagai metode ekstraksi fitur dan algoritma pembelajaran mesin (machine learning).

Proyek ini dikembangkan untuk tugas akhir mata kuliah **Pengolahan Citra Digital (PCD)**.

## 🌟 Fitur

- **Tampilan Dataset**: Visualisasikan kelas dataset dan contoh gambar secara langsung.
- **Berbagai Metode Ekstraksi Fitur**:
  - **Color Histogram**: Menganalisis distribusi warna dalam ruang warna HSV.
  - **Edge Features**: Menggunakan Canny Edge Detection untuk menganalisis kepadatan tepi bentuk.
  - **HOG (Histogram of Oriented Gradients)**: Menangkap struktur dan bentuk objek.
- **Algoritma Pembelajaran Mesin**:
  - **K-Nearest Neighbors (KNN)**: Dengan nilai K yang dapat disesuaikan.
  - **Support Vector Machine (SVM)**: Kernel linear untuk klasifikasi yang tangguh.
- **Prediksi Real-time**: Unggah gambar dan dapatkan hasil instan beserta skor kepercayaan (confidence score).
- **Evaluasi Performa**: Visualisasi akurasi dan confusion matrix setelah proses pelatihan (training).

## 📁 Struktur Proyek

```text
PCD-UAS/
├── dataset/            # Gambar Training/Testing (terorganisir dalam subfolder)
├── src/
│   ├── app.py          # Aplikasi utama Streamlit
│   ├── processing.py   # Preprocessing gambar & ekstraksi fitur
│   └── classifier.py   # Logika pelatihan model & prediksi
├── requirements.txt    # Daftar dependensi
└── README.md           # File ini
```

## 🛠️ Persyaratan Sistem

- Python 3.8+
- OpenCV
- NumPy
- Streamlit
- Scikit-learn
- Scikit-image (untuk HOG)
- Matplotlib

## 🚀 Instalasi & Persiapan

1. **Clone repositori** (atau unduh filenya):
   ```bash
   git clone <url-repositori>
   cd PCD-UAS
   ```

2. **Instal dependensi**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Siapkan dataset**:
   Pastikan folder `dataset/` memiliki subfolder untuk setiap kelas (misalnya, `dataset/car/` dan `dataset/bike/`).

## 🎮 Cara Penggunaan

1. **Jalankan aplikasi**:
   ```bash
   streamlit run src/app.py
   ```

2. **Konfigurasi opsi** di bilah sisi (sidebar):
   - Pilih Jalur Dataset (Dataset Path).
   - Pilih metode Ekstraksi Fitur (Feature Extraction).
   - Pilih Algoritma Klasifikasi (Classification Algorithm).

3. **Latih model**:
   - Buka tab **Training** dan klik tombol **"Train Model"**.
   - Tinjau akurasi dan confusion matrix yang muncul.

4. **Uji gambar baru**:
   - Buka tab **Testing**.
   - Unggah gambar baru (JPG/PNG).
   - Lihat hasil prediksi dan skor kepercayaannya!

## 📝 Lisensi

Proyek ini dibuat untuk tujuan pendidikan sebagai bagian dari kuliah Pengolahan Citra Digital.
