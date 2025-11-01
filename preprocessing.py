import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import time

print("Ön işleme betiği başlatıldı...")

# 1️⃣ Tür (genre) JSON'ını işlemek için yardımcı fonksiyon
def get_genre_names(json_string):
    try:
        genres_list = json.loads(json_string) 
        if not genres_list: return ""
        names = [item['name'] for item in genres_list]
        return " ".join(names)
    except:
        return ""

# 2️⃣ Veri setini yükle
print("tmdb_5000_movies.csv yükleniyor...")
films = pd.read_csv("tmdb_5000_movies.csv")

# 3️⃣ Gerekli sütunları al ve temizle
films = films[["title", "overview", "genres"]]
films = films.dropna(subset=["overview"]).reset_index(drop=True)
films.rename(columns={"overview": "description"}, inplace=True)

# 4️⃣ Türleri işle ve açıklama ile birleştir
print("Film türleri (genres) işleniyor...")
films["genre_text"] = films["genres"].apply(get_genre_names)
films["combined_features"] = films["description"] + (" " + films["genre_text"]) * 3

# 5️⃣ API için sadeleştirilmiş film verisini hazırla
# Sadece bu iki sütuna ihtiyacımız var
films_data_api = films[["title", "genre_text"]]

# 6️⃣ Embedding modelini yükle
print("SentenceTransformer modeli yükleniyor...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 7️⃣ Embedding'leri hesapla (EN YAVAŞ ADIM)
print("Embedding'ler hesaplanıyor... Bu işlem birkaç dakika sürebilir.")
start_time = time.time()
film_embeddings = model.encode(films["combined_features"].tolist(), show_progress_bar=True)
end_time = time.time()
print(f"Embedding hesaplaması {end_time - start_time:.2f} saniyede tamamlandı.")

# 8️⃣ Hesaplanan verileri diske kaydet
np.save("film_embeddings.npy", film_embeddings)
print("✅ 'film_embeddings.npy' dosyası başarıyla kaydedildi.")

films_data_api.to_csv("films_data.csv")
print("✅ 'films_data.csv' dosyası başarıyla kaydedildi.")
print("\nÖn işleme tamamlandı!")