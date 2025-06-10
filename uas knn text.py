# =============================================================================
# CAPSTONE PROJECT AKHIR PRAKTIKUM: KLASIFIKASI TEKS KULINER
# =============================================================================
# Deskripsi:
# Proyek ini bertujuan untuk membangun model machine learning yang dapat
# mengklasifikasikan kategori kuliner berdasarkan deskripsi teks.
#
# Dataset:
# Dataset diambil dari file 'archive/RAW/Data_Kalimat.xlsx'. Dataset ini berisi
# kolom 'Kalimat' (sebagai fitur) dan 'Label' (sebagai target).
#
# Pendekatan Model:
# Model yang digunakan adalah Klasifikasi dengan algoritma K-Nearest Neighbors (KNN).
# Alasan pemilihan KNN adalah untuk menguji kemampuannya dalam mengelompokkan
# teks dengan deskripsi serupa setelah melalui proses feature extraction.
# =============================================================================

# --- 1. IMPORT LIBRARY YANG DIBUTUHKAN ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Library untuk pemrosesan teks dan machine learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 2. MEMUAT DATASET DARI FILE EXCEL ---
# Menggunakan pandas untuk membaca file .xlsx
try:
    # Ganti dengan path yang sesuai jika perlu
    excel_file_path = 'archive/RAW/Data_Kalimat.xlsx'
    df = pd.read_excel(excel_file_path)
    print("✅ Data berhasil dimuat dari file Excel.")
    print("   Berikut 5 baris pertama dari data:")
    print(df.head())
    print(f"\n   Jumlah data: {len(df)} baris")
    print(f"   Nama kolom: {df.columns.tolist()}")

except FileNotFoundError:
    print(f"❌ ERROR: File tidak ditemukan di '{excel_file_path}'. Pastikan path sudah benar.")
    exit()
except Exception as e:
    print(f"❌ ERROR saat membaca file Excel: {e}")
    exit()


# --- 3. EXPLORATORY DATA ANALYSIS (EDA) ---
# Memahami distribusi dan karakteristik data.
print("\n--- 3. Melakukan Exploratory Data Analysis (EDA) ---")

# Cek apakah ada data yang kosong/hilang
if df.isnull().sum().sum() > 0:
    print("\n   Menangani nilai yang hilang...")
    df.dropna(inplace=True)
    print("   Baris dengan nilai kosong telah dihapus.")

# Asumsikan kolom fitur adalah 'Kalimat' dan kolom target adalah 'Label'
# Jika nama kolomnya berbeda, silakan sesuaikan di sini
FEATURE_COLUMN = 'Kalimat'
LABEL_COLUMN = 'Label'

if FEATURE_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
    print(f"❌ ERROR: Kolom '{FEATURE_COLUMN}' atau '{LABEL_COLUMN}' tidak ditemukan di file Excel.")
    exit()

# Visualisasi distribusi kelas untuk melihat apakah datanya seimbang
plt.figure(figsize=(10, 6))
sns.countplot(x=LABEL_COLUMN, data=df, order=df[LABEL_COLUMN].value_counts().index)
plt.title('Distribusi Jumlah Sampel per Kelas (Label)')
plt.xlabel('Kelas (Label)')
plt.ylabel('Jumlah Sampel')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# --- 4. FEATURE EXTRACTION (Mengubah Teks menjadi Angka) ---
# Menggunakan TF-IDF untuk mengubah kolom 'Kalimat' menjadi vektor numerik.
print("\n--- 4. Melakukan Feature Extraction dengan TF-IDF ---")
vectorizer = TfidfVectorizer(max_features=1000) # Batasi hingga 1000 fitur paling top
X = vectorizer.fit_transform(df[FEATURE_COLUMN])
y = df[LABEL_COLUMN]

print("   Teks berhasil diubah menjadi matriks fitur.")
print(f"   Bentuk matriks fitur X (setelah TF-IDF): {X.shape}")


# --- 5. MEMBAGI DATA MENJADI TRAINING DAN TESTING SET ---
# Data dibagi menjadi 80% data latih dan 20% data uji.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y # 'stratify' penting untuk data yang tidak seimbang
)
print("\n--- 5. Data telah dibagi menjadi data latih dan data uji ---")


# --- 6. HYPERPARAMETER TUNING UNTUK KNN ---
# Mencari nilai 'k' (n_neighbors) terbaik untuk model KNN.
print("\n--- 6. Melakukan Hyperparameter Tuning untuk mencari 'k' terbaik ---")
param_grid = {'n_neighbors': np.arange(1, 11)} # Menguji k dari 1 hingga 10
knn = KNeighborsClassifier()

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
print("\n--- 8. Mengevaluasi performa model pada data uji ---")
y_pred = final_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"   Akurasi Model: {accuracy:.4f}")

print("\n   Laporan Klasifikasi:")
report = classification_report(y_test, y_pred, zero_division=0)
print(report)


# --- 9. HASIL OUTPUT DAN INFERENCE (KESIMPULAN) ---
print("\n--- 9. Hasil Output dan Inference ---")
cm = confusion_matrix(y_test, y_pred)
class_names = np.unique(y) # Dapatkan nama kelas unik dari data

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix Hasil Klasifikasi')
plt.xlabel('Label Prediksi')
plt.ylabel('Label Sebenarnya')
plt.show()

print("\n   Kesimpulan (Inference):")
print(f"   - Dengan mengubah data teks menjadi fitur numerik menggunakan TF-IDF, model KNN berhasil mencapai akurasi sebesar {accuracy:.2%}.")
print("   - Model ini menunjukkan kemampuannya untuk mempelajari pola dari data teks dan melakukan klasifikasi.")
print("   - Untuk peningkatan di masa depan, bisa dicoba algoritma lain yang lebih canggih untuk pemrosesan teks seperti Naive Bayes, SVM, atau bahkan model berbasis Transformer.")