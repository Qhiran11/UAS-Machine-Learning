# =============================================================================
# CAPSTONE PROJECT AKHIR PRAKTIKUM: KLASIFIKASI MAKANAN KHAS PALU
# =============================================================================
# Deskripsi:
# Proyek ini bertujuan untuk membangun model machine learning yang dapat
# mengklasifikasikan gambar makanan khas Palu.
#
# Dataset:
# Dataset yang digunakan adalah 'data_makanan_palu_processed.npz', yang berisi
# fitur-fitur yang telah diekstraksi dari gambar makanan. Dataset ini dipilih
# karena relevansinya dengan tujuan proyek untuk mengenali kuliner lokal.
#
# Pendekatan Model:
# Model yang digunakan adalah Klasifikasi dengan algoritma K-Nearest Neighbors (KNN).
# Alasan pemilihan KNN adalah karena sederhana, mudah diinterpretasikan, dan
# efektif untuk dataset dengan fitur yang jelas.
# =============================================================================

# --- 1. IMPORT LIBRARY YANG DIBUTUHKAN ---
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# --- 2. MEMUAT DATASET ---
# Data yang dimuat adalah data yang sudah melalui tahap feature extraction.
# File .npz ini berisi array X (fitur), y (label), dan nama kelas.
try:
    processed_data_file = r'C:\Users\YOGA\Documents\!Muh. Qhiran. N F55123021 (S1 - Teknik Informatika)\Folder Semester 4\Machine Learning\Preprocessing\data_makanan_palu_processed.npz'
    loaded_data = np.load(processed_data_file)
    X = loaded_data['X']
    y = loaded_data['y']
    class_names = loaded_data['classes']
    print("✅ Data berhasil dimuat.")
    print(f"   Bentuk data (fitur) X: {X.shape}")
    print(f"   Bentuk data (label) y: {y.shape}")
    print(f"   Nama kelas: {class_names}")
except FileNotFoundError:
    print("❌ ERROR: File tidak ditemukan. Pastikan path file sudah benar.")
    exit()

# --- 3. EXPLORATORY DATA ANALYSIS (EDA) ---
# EDA bertujuan untuk memahami distribusi dan karakteristik data.
print("\n--- 3. Melakukan Exploratory Data Analysis (EDA) ---")

# Visualisasi distribusi kelas untuk melihat apakah datanya seimbang
plt.figure(figsize=(10, 6))
sns.countplot(x=y)
plt.title('Distribusi Jumlah Sampel per Kelas Makanan')
plt.xlabel('Kelas (Label)')
plt.ylabel('Jumlah Sampel')
plt.xticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=45)
plt.tight_layout()
plt.show()

# --- 4. PREPROCESSING & FEATURE EXTRACTION ---
# Karena feature extraction sudah dilakukan, fokus preprocessing adalah standarisasi.
# Standarisasi (mengubah skala fitur) sangat penting untuk algoritma berbasis jarak seperti KNN.
print("\n--- 4. Melakukan Preprocessing Data (Standarisasi) ---")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("   Data fitur telah berhasil distandarisasi.")

# --- 5. MEMBAGI DATA MENJADI TRAINING DAN TESTING SET ---
# Data dibagi menjadi 80% data latih dan 20% data uji.
# 'stratify=y' digunakan untuk memastikan proporsi kelas di data latih dan uji sama.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("\n--- 5. Data telah dibagi menjadi data latih dan data uji ---")
print(f"   Bentuk X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"   Bentuk X_test: {X_test.shape}, y_test: {y_test.shape}")

# --- 6. HYPERPARAMETER TUNING UNTUK KNN ---
# Mencari nilai 'k' (n_neighbors) terbaik untuk model KNN.
# Kita akan menguji nilai k dari 1 hingga 20.
print("\n--- 6. Melakukan Hyperparameter Tuning untuk mencari 'k' terbaik ---")
param_grid = {'n_neighbors': np.arange(1, 21)}
knn = KNeighborsClassifier()

# GridSearchCV akan mencoba semua nilai 'k' dan menemukan yang terbaik
# menggunakan cross-validation (cv=5).
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_k = grid_search.best_params_['n_neighbors']
print(f"   Nilai 'k' terbaik yang ditemukan adalah: {best_k}")

# --- 7. MELATIH MODEL KNN DENGAN PARAMETER TERBAIK ---
print(f"\n--- 7. Melatih model KNN dengan k={best_k} ---")
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train, y_train)
print("   Model berhasil dilatih.")

# --- 8. EVALUASI MODEL ---
# Mengevaluasi performa model pada data uji yang belum pernah dilihat sebelumnya.
print("\n--- 8. Mengevaluasi performa model pada data uji ---")
y_pred = final_model.predict(X_test)

# Menghitung Akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f"   Akurasi Model: {accuracy:.4f}")

# Menampilkan Laporan Klasifikasi (Precision, Recall, F1-Score)
print("\n   Laporan Klasifikasi:")
report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
print(report)

# --- 9. HASIL OUTPUT DAN INFERENCE (KESIMPULAN) ---
# Menyajikan output secara visual dan memberikan kesimpulan.
print("\n--- 9. Hasil Output dan Inference ---")

# Visualisasi Confusion Matrix
# Confusion matrix menunjukkan seberapa baik model mengklasifikasikan setiap kelas.
# Sumbu diagonal menunjukkan prediksi yang benar.
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix Hasil Klasifikasi')
plt.xlabel('Label Prediksi')
plt.ylabel('Label Sebenarnya')
plt.show()

print("\n   Kesimpulan (Inference):")
print(f"   - Model KNN yang telah dioptimalkan (dengan k={best_k}) berhasil mencapai akurasi sebesar {accuracy:.2%} pada data uji.")
print("   - Berdasarkan laporan klasifikasi dan confusion matrix, model menunjukkan performa yang baik dalam membedakan sebagian besar kelas makanan.")
print("   - Adanya beberapa kesalahan klasifikasi (nilai di luar diagonal utama pada confusion matrix) menunjukkan bahwa beberapa jenis makanan mungkin memiliki fitur visual yang mirip.")
print("   - Rekomendasi untuk Aplikasi Nyata: Model ini dapat dikembangkan menjadi aplikasi mobile untuk turis atau warga lokal yang ingin mengidentifikasi makanan khas Palu secara otomatis melalui kamera smartphone.")