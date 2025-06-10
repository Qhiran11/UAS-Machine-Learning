import requests
from bs4 import BeautifulSoup
import os
import json

def scrape_bing_images(query: str, limit: int = 10):
    """
    Mencari dan mengunduh gambar dari Bing berdasarkan query.

    :param query: Kata kunci untuk pencarian gambar (e.g., "kucing oren lucu").
    :param limit: Jumlah maksimum gambar yang ingin diunduh.
    """
    
    # Membuat nama folder yang valid dari query
    folder_name = "".join(c for c in query if c.isalnum() or c in (' ', '_')).rstrip()
    if not os.path.exists(folder_name):
        print(f"Membuat folder: '{folder_name}'")
        os.makedirs(folder_name)
    else:
        print(f"Folder '{folder_name}' sudah ada.")

    # Header User-Agent agar request terlihat seperti dari browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # URL pencarian gambar di Bing
    url = f"https://www.bing.com/images/search?q={query}"
    
    print(f"Mengambil halaman untuk query: '{query}'...")
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Gagal mengambil halaman. Status: {response.status_code}")
        return

    # Parsing HTML dengan BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Menemukan semua link gambar. Class 'iusc' adalah container untuk thumbnail dan link.
    # PENTING: Class ini bisa berubah sewaktu-waktu oleh Bing.
    image_links = soup.find_all("a", class_="iusc")
    
    print(f"Ditemukan {len(image_links)} potensi gambar. Mengunduh {limit} gambar pertama...")
    
    downloaded_count = 0
    for i, link in enumerate(image_links):
        if downloaded_count >= limit:
            break
            
        try:
            # Data gambar (termasuk URL resolusi tinggi) disimpan dalam atribut 'm' sebagai JSON string
            m_json = link.get("m")
            if not m_json:
                continue

            m_data = json.loads(m_json)
            image_url = m_data.get("murl") # 'murl' adalah URL media
            
            if not image_url:
                continue

            # Mengunduh gambar
            print(f"({downloaded_count + 1}/{limit}) Mengunduh dari: {image_url}")
            image_response = requests.get(image_url, headers=headers, timeout=10)
            
            if image_response.status_code == 200:
                # Menentukan nama file dan menyimpannya
                file_extension = image_url.split(".")[-1]
                if len(file_extension) > 4: file_extension = 'jpg' # Default jika ekstensi tidak jelas
                
                file_path = os.path.join(folder_name, f"{folder_name}_{downloaded_count + 1}.{file_extension}")
                
                with open(file_path, "wb") as f:
                    f.write(image_response.content)
                
                downloaded_count += 1

        except Exception as e:
            print(f"Gagal mengunduh gambar ke-{i+1}: {e}")

    print(f"\nSelesai! {downloaded_count} gambar berhasil diunduh ke folder '{folder_name}'.")


if __name__ == "__main__":
    # --- GANTI BAGIAN INI SESUAI KEBUTUHAN ANDA ---
    search_query = "white rose"
    image_limit =1000
    # ---------------------------------------------
    
    scrape_bing_images(query=search_query, limit=image_limit)