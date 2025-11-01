# 🎬 Film Öneri API (Backend)

Bu proje, bir film öneri sistemi için **Python (FastAPI)** ile geliştirilmiş backend sunucusudur. Kaggle'ın "TMDB 5000 Movie Dataset" verilerini kullanarak, film açıklamalarına ve türlerine (genres) dayalı içerik tabanlı (content-based) öneriler sunar.

---

> ### 🔗 İLİŞKİLİ PROJE: ANDROID (FRONTEND)
>
> Bu API sunucusunun verilerini kullanan **Kotlin / Jetpack Compose** ile yazılmış Android mobil uygulamasına aşağıdaki linkten ulaşabilirsiniz:
>
> **[➡️ film-oneri-android (Kotlin) Reposu](https://github.com/ogulcannarin/film-oneri-android.git)**


---

## 🛠️ Neler Yapıldı?

Bu projede iki ana Python betiği (script) bulunmaktadır:

### 1. `preprocessing.py` (Ön İşleme)
Bu betik, sunucunun hızlı çalışması için tüm ağır işi önceden yapar:
* `tmdb_5000_movies.csv` dosyasını **Pandas** ile okur.
* Gerekli sütunları (`title`, `overview`, `genres`) temizler.
* Film türlerini (JSON) metne dönüştürür ve öneri kalitesini artırmak için açıklama metniyle birleştirir.
* **Sentence Transformers (`all-MiniLM-L6-v2`)** modelini kullanarak 4800 filmin birleştirilmiş metinlerinden vektör embedding'leri (vektör gömüleri) oluşturur.
* Bu embedding'leri `film_embeddings.npy` dosyasına, temiz film listesini ise `films_data.csv` dosyasına kaydeder.

### 2. `api.py` (API Sunucusu)
Bu betik, **FastAPI** kullanarak mobil uygulamanın bağlanacağı sunucuyu başlatır:
* Sunucu başlarken `film_embeddings.npy` ve `films_data.csv` dosyalarını hafızaya yükler (böylece her istekte hesaplama yapılmaz).
* Metinleri (`spider-man` vs `spiderman` gibi) arayabilmek için temiz bir arama sütunu oluşturur.
* **İki adet "endpoint" (uç nokta) sağlar:**
    * `GET /search?query=...`: Kullanıcının arama terimine uyan filmleri JSON listesi olarak döndürür.
    * `GET /recommendations/{movie_id}`: Belirli bir filmin ID'sine göre **Numpy** ile kosinüs benzerliği (cosine similarity) hesaplar ve en benzer 5 filmi JSON olarak döndürür.

## 🚀 Nasıl Çalıştırılır?

1.  Gerekli kütüphaneleri yükle:
    ```bash
    pip install pandas numpy sentence-transformers fastapi "uvicorn[standard]"
    ```
2.  (Sadece bir kez) Ön işleme betiğini çalıştırarak `.npy` ve `.csv` dosyalarını oluştur:
    ```bash
    python preprocessing.py
    ```
3.  API sunucusunu başlat:
    ```bash
    python api.py
    ```
4.  Sunucu artık `http://127.0.0.1:8000` adresinde çalışıyor olacaktır.
