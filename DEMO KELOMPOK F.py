# ==============================================================================
# KODE DEMO MACHINE LEARNING: KLASIFIKASI BUNGA IRIS MENGGUNAKAN KNN
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. IMPORT LIBRARY (Memanggil pustaka/tools yang dibutuhkan)
# ------------------------------------------------------------------------------
from sklearn.datasets import load_iris             # Mengambil dataset Iris bawaan Scikit-Learn
from sklearn.model_selection import train_test_split # Fungsi untuk membagi data latih & uji
from sklearn.neighbors import KNeighborsClassifier   # Algoritma klasifikasi K-Nearest Neighbors
from sklearn.metrics import accuracy_score          # Metrik untuk mengukur persentase akurasi

# ------------------------------------------------------------------------------
# 2. LOAD DATASET (Memuat data bunga Iris ke dalam variabel)
# ------------------------------------------------------------------------------
iris = load_iris()

# X = Features (Input): 4 kolom ukuran fisik (Panjang Sepal, Lebar Sepal, Panjang Petal, Lebar Petal)
X = iris.data

# y = Target (Output): Label kelas spesies (0 = Setosa, 1 = Versicolor, 2 = Virginica)
y = iris.target

# ------------------------------------------------------------------------------
# 3. SPLIT DATA (Membagi data latih dan data uji)
# ------------------------------------------------------------------------------
# Total 150 baris data dipisah menjadi:
# - 80% Data Latih (X_train, y_train) -> Digunakan untuk melatih/mengajari model
# - 20% Data Uji (X_test, y_test)     -> Digunakan khusus untuk menguji kepintaran model
# - test_size=0.2                     -> Porsi data uji sebesar 20% (30 sampel)
# - stratify=y                        -> Memastikan proporsi 3 jenis bunga seimbang di train & test
# - random_state=42                   -> Pengunci acak agar pembagian data selalu sama tiap kali di-run
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------------------------
# 4. TRAIN MODEL (Membuat dan melatih model KNN)
# ------------------------------------------------------------------------------
# Inisialisasi model KNN dengan n_neighbors=3
# (Model akan menebak jenis bunga berdasarkan 3 data bunga terdekat yang ukurannya paling mirip)
model = KNeighborsClassifier(n_neighbors=3)

# fit() = Proses melatih model untuk mempelajari pola hubungan antara fitur (X) dan target (y)
model.fit(X_train, y_train)

# ------------------------------------------------------------------------------
# 5. EVALUASI MODEL (Menguji akurasi tebakan model)
# ------------------------------------------------------------------------------
# predict() = Model diminta menebak jenis bunga dari data uji yang belum pernah ia lihat (X_test)
y_pred = model.predict(X_test)

# Membandingkan tebakan model (y_pred) dengan kunci jawaban asli (y_test)
akurasi = accuracy_score(y_test, y_pred)

# Cetak hasil akurasi ke terminal (dikali 100 agar jadi persen)
print(f"=== HASIL EVALUASI MODEL ===")
print(f"Akurasi Model pada Data Uji: {akurasi * 100:.2f}%\n")

# ------------------------------------------------------------------------------
# 6. LIVE DEMO (Simulasi memprediksi sampel bunga baru di lapangan)
# ------------------------------------------------------------------------------
# Data masukan bunga baru: [Panjang Sepal, Lebar Sepal, Panjang Petal, Lebar Petal] dalam cm
bunga_baru = [[5.1, 3.5, 1.4, 0.2]]

# Model memproses 4 angka masukan di atas lalu menentukan prediksi kelasnya
hasil_prediksi = model.predict(bunga_baru)

# Mengubah angka hasil prediksi (0, 1, atau 2) menjadi nama asli spesiesnya (Setosa/Versicolor/Virginica)
nama_spesies = iris.target_names[hasil_prediksi[0]]

# Cetak hasil prediksi ke terminal
print(f"=== SIMULASI PREDIKSI BUNGA BARU ===")
print(f"Input Ukuran Bunga : {bunga_baru[0]}")
print(f"Hasil Prediksi ML  : Bunga {nama_spesies.capitalize()}")