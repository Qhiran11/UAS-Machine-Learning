import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

all_models = {
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=3), 
    "Random Forest": RandomForestClassifier(random_state=60, n_estimators=100)
}

processed_data_file = r'C:\Users\YOGA\Documents\!Muh. Qhiran. N F55123021 (S1 - Teknik Informatika)\Folder Semester 4\Machine Learning\Preprocessing\data_makanan_palu_processed.npz'

try:
    loaded_data = np.load(processed_data_file)
    X_all = loaded_data['X']
    y_all = loaded_data['y']
    class_names_loaded = loaded_data['classes']

    print(f"Data berhasil dimuat. Bentuk X: {X_all.shape}, Bentuk y: {y_all.shape}")
    print(f"Nama kelas: {class_names_loaded}")

except FileNotFoundError:
    print(f"ERROR: File data yang diproses tidak ditemukan di {processed_data_file}")
    print("Pastikan path file sudah benar dan skrip preprocessing sudah dijalankan.")
    exit()
except KeyError as e:
    print(f"ERROR: Kunci {e} tidak ditemukan dalam file .npz. Pastikan file .npz berisi 'X', 'y', dan 'classes'.")
    exit()



X_train, X_test, y_train, y_test = train_test_split(
    X_all,
    y_all,
    test_size=0.2,
    random_state=42,
    stratify=y_all
)

print(f"\nData telah di-split ulang:")
print(f"Bentuk X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Bentuk X_test: {X_test.shape}, y_test: {y_test.shape}")

results = {}

for model_name, model in all_models.items():
    print(f"\n--- Melatih Model: {model_name} ---")
    try:
    
        model.fit(X_train, y_train)

    
        y_pred = model.predict(X_test)

    
        accuracy = accuracy_score(y_test, y_pred)
    
        if isinstance(class_names_loaded, np.ndarray) and class_names_loaded.ndim == 0:
            target_names_list = class_names_loaded.tolist() if isinstance(class_names_loaded, np.ndarray) else class_names_loaded
        elif isinstance(class_names_loaded, np.ndarray) and class_names_loaded.ndim > 0 :
             target_names_list = class_names_loaded.tolist()
        else: 
            target_names_list = class_names_loaded

        report = classification_report(y_test, y_pred, target_names=target_names_list, zero_division=0)


        results[model_name] = {
            "accuracy": accuracy,
            "classification_report": report
        }

        print(f"Akurasi: {accuracy:.4f}")
        print("Laporan Klasifikasi:")
        print(report)

    except Exception as e:
        print(f"   Error saat melatih atau mengevaluasi {model_name}: {e}")
        results[model_name] = {
            "accuracy": "Error",
            "classification_report": str(e)
        }

print("\n--- Ringkasan Akurasi ---")
for model_name, result_data in results.items():
    print(f"{model_name}: {result_data['accuracy']}")