from duckduckgo_search import DDGS
import requests
import os

def download_images(query, folder, max_results=20):
    """
    Fungsi untuk mencari dan mengunduh gambar menggunakan DuckDuckGo.
    """
    # Membuat folder jika belum ada
    os.makedirs(folder, exist_ok=True)
    print(f"Memulai unduhan untuk query: '{query}'...")

    # Menggunakan context manager 'with' adalah praktik terbaik
    with DDGS(client_config={"verify": False}) as ddgs:
        # Panggil metode .images() untuk mencari gambar
        # Gunakan parameter 'keywords' untuk query pencarian
        results = ddgs.images(
            keywords=query,
            max_results=max_results
        )

        if not results:
            print(f"Tidak ada gambar yang ditemukan untuk query: '{query}'")
            return

        # Iterasi melalui hasil pencarian
        for i, result in enumerate(results):
            try:
                # URL gambar ada di dalam key 'image'
                img_url = result['image']
                
                # Mengunduh konten gambar
                img_data = requests.get(img_url).content
                
                # Menyimpan gambar ke file
                file_path = os.path.join(folder, f"{query.replace(' ', '_')}_{i+1}.jpg")
                with open(file_path, 'wb') as handler:
                    handler.write(img_data)
                
                print(f"[✓] Berhasil diunduh: {file_path}")

            # Menggunakan 'Exception as e' lebih baik karena menampilkan detail error
            except Exception as e:
                print(f"[x] Gagal mengunduh gambar {i+1} untuk '{query}'. Error: {e}")

# Contoh penggunaan:
download_images("sport car", "dataset/sport", max_results=30)
download_images("sedan car", "dataset/sedan", max_results=30)

print("\nProses selesai.")