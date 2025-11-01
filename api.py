import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
import uvicorn

# --- YENİ YARDIMCI FONKSİYON ---
def clean_text(text):
    """Metni arama için temizler (küçük harf, boşluk/tire yok)."""
    return str(text).lower().replace('-', '').replace(' ', '')

# --- API UYGULAMASINI BAŞLAT ---
app = FastAPI(
    title="Film Öneri API",
    description="Benzer filmleri bulmak için bir backend sunucusu."
)

# --- GLOBAL DEĞİŞKENLER (SUNUCU BAŞLARKEN BİR KEZ YÜKLENİR) ---
print("API sunucusu başlatılıyor...")
try:
    print("Veriler hafızaya yükleniyor...")
    films_df = pd.read_csv("films_data.csv", index_col=0)
    film_embeddings = np.load("film_embeddings.npy")
    
    # --- GÜNCELLENDİ ---
    # Arama için TEMİZLENMİŞ bir sütun oluşturuyoruz
    print("Arama için başlıklar temizleniyor...")
    films_df['search_title'] = films_df['title'].apply(clean_text)
    
    print("✅ Veriler başarıyla yüklendi. Sunucu hazır.")
except FileNotFoundError:
    print("HATA: 'films_data.csv' veya 'film_embeddings.npy' bulunamadı.")
    print("Lütfen önce 'preprocessing.py' betiğini çalıştırdığınızdan emin olun.")
    exit()

# --- YARDIMCI FONKSİYONLAR ---
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- API ENDPOINT'LERİ ---

@app.get("/")
def read_root():
    return {"message": "Film Öneri API'sine Hoş Geldiniz!"}


@app.get("/search")
def search_movies(query: str = Query(..., min_length=2)):
    """
    Kullanıcının girdiği metne göre filmleri arar.
    """
    
    # --- GÜNCELLENDİ ---
    # Kullanıcının aramasını da aynı temizleme fonksiyonundan geçir
    query_clean = clean_text(query)
    
    # Temizlenmiş 'search_title' sütununda ara
    partial_matches = films_df[films_df['search_title'].str.contains(query_clean)]
    
    if partial_matches.empty:
        # Hiçbir şey bulunamazsa
        return {"results": []} # Mobil uygulama bu yüzden boş ekran gösteriyordu
    
    results = partial_matches
    
    # Sonuçları mobil uygulamanın anlayacağı formata (JSON) çevir
    return_data = results[['title']].reset_index().rename(columns={'index': 'id'}).to_dict('records')
    return {"results": return_data}


@app.get("/recommendations/{movie_id}")
def get_recommendations(movie_id: int, top_n: int = 5):
    """
    Belirli bir film ID'sine göre en benzer filmleri döndürür.
    """
    if movie_id not in films_df.index:
        raise HTTPException(status_code=404, detail="Film ID'si bulunamadı.")

    try:
        movie_vector = film_embeddings[movie_id]
        original_movie_title = films_df.loc[movie_id]['title']
        
        similarities = []
        for i, vec in enumerate(film_embeddings):
            if i != movie_id:
                sim = cosine_similarity(movie_vector, vec)
                similarities.append((i, films_df.iloc[i]['title'], sim))
        
        similarities = sorted(similarities, key=lambda x: x[2], reverse=True)
        
        recommendations = []
        for rec_id, title, score in similarities[:top_n]:
            recommendations.append({
                "id": rec_id, 
                "title": title, 
                "score": float(score)
            })
            
        return {"query_movie": original_movie_title, "recommendations": recommendations}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sunucu hatası: {str(e)}")


# --- SUNUCUYU ÇALIŞTIRMAK İÇİN ---
if __name__ == "__main__":
    print("Sunucu http://127.0.0.1:8000 adresinde başlatılıyor...")
    uvicorn.run(app, host="127.0.0.1", port=8000)