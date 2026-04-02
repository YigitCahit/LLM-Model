# Transformer Tabanlı Türkçe Dil Modeli

Bu proje, Transformer mimarisine dayalı, saf Python ile yazılmış bir yapay sinir ağı kullanan Türkçe metin üreticisidir. PyTorch veya NumPy dahil hiçbir kütüphane kullanılmamıştır. Model, verilen başlangıç kelimesinden başlayarak yeni metinler üretebilir. Sürekli güncellenecek ve geliştirilecektir.

## Proje Özellikleri

- **Transformer Mimarisi**: Multi-head attention mekanizması ile gelişmiş metin işleme
- **Pozisyonel Kodlama**: Token pozisyonlarını modellemek için positional encoding
- **Layer Normalization**: Ağ katmanlarında normalizasyon
- **Geri Yayılım Algoritması**: Backpropagation ile eğitim
- **Model Kaydı/Yükleme**: Eğitilmiş modeli kaydetme ve yükleme özelliği
- **Metin Üretimi**: Eğitilmiş model ile dinamik metin üretme

## Dosya Yapısı

```
├── app.py                   # Ana uygulama - model eğitimi ve metin üretimi
├── sinir_agi.py             # Sinir ağı mimarisi ve yapı tanımları
├── model.py                 # İleri geçiş (forward pass) işlemleri
├── egitim.py                # Eğitim algoritmaları ve loss fonksiyonları
├── egitim_dongusu.py        # Eğitim döngüsü ve metin üretim fonksiyonları
├── islemler.py              # Matematiksel işlemler (çarpım, softmax, vb.)
├── kaydet.py                # Model kaydı ve yükleme işlemleri
├── rastgele.py              # Rastgele sayı üretimi ve matematiksel fonksiyonlar
├── veri.txt                 # Eğitim verisi (Türkçe metinler)
├── model.json               # Kaydedilmiş model ağırlıkları
├── teste.py                 # Test kodları
└── README.md                # Bu dosya
```

## Nasıl Kullanılır

### Gereksinimleri
Python 3.7 veya daha yüksek bir sürüm gereklidir. Harici kütüphane bağımlılığı yoktur.

### Eğitim ve Metin Üretimi

Ana programı çalıştırın:

```bash
python app.py
```

`app.py` şunları yapar:
1. Mevcut bir model varsa (`model.json`) yükler
2. Model yoksa yeni bir model eğitir
3. Eğitilmiş model ile metin üretimi yapar

### Örnek Kullanım

```python
from egitim_dongusu import egit, uret
from kaydet import modeli_kaydet, modeli_yukle

# Yeni model eğitimi
ag, kelime_sayi, sayi_kelime = egit(
    dosya_adi="veri.txt",
    epochs=100,
    ogrenme_hizi=0.001,
    pencere=4
)

# Model kaydı
modeli_kaydet(ag, kelime_sayi, sayi_kelime, "model.json")

# Metin üretimi
uret(ag, kelime_sayi, sayi_kelime, 
     baslangic_kelime="merhaba", 
     uzunluk=10)
```

## Ana Bileşenler

### `SinirAgi` Sınıfı
Transformer modelinin ana yapısını tanımlar:
- Embedding tablosu
- Pozisyonel encoding
- Transformer katmanları
- Çıkış doğrusal katmanı

### Transformer Bloğu
- **Multi-Head Attention**: Birden fazla dikkat başı ile gelişmiş ilişki modelleme
- **Feed-Forward Network**: Non-lineer dönüşümler
- **Layer Normalization**: Katman çıktılarının normalizasyonu

### Eğitim
- **Cross-Entropy Loss**: Sınıflandırma kaybı hesaplaması
- **Backpropagation**: Gradyan hesaplaması ve parametre güncelleme
- **Softmax**: Çıkış olasılıkları

## Eğitim Parametreleri

Default parametreler:
- **Epochs**: 100 - Eğitim iterasyonu sayısı
- **Learning Rate**: 0.001 - Öğrenme hızı
- **Window Size**: 4 - Giriş token sayısı
- **Hidden Dimension**: 64 - Gizli katman boyutu
- **Feedforward Dimension**: 256 - Feedforward katman boyutu
- **Number of Heads**: 4 - Attention başlarının sayısı
- **Number of Layers**: 2 - Transformer katman sayısı

Bu parametreler `app.py` dosyasında değiştirilebilir.

## Veri Formatı

`veri.txt` dosyası, her satırın bir cümle olduğu Türkçe metinler içerir:

```
merhaba bugün nasılsın umarım iyisindir
merhaba bugün hava çok güzel
selam bugün neler yapacaksın
...
```

## Metin Üretimi

Eğitilmiş model, verilen başlangıç kelimesinden başlayarak ardışık tokenlar predikt eder:

```python
uret(ag, kelime_sayi, sayi_kelime, 
     baslangic_kelime="merhaba", 
     uzunluk=20)
```

Bu, "merhaba" ile başlayan ve 20 kelime uzunluğunda bir metin üretir.

## Notlar

- Model tamamıyla sıfırdan geliştirilmiştir (harici ML kütüphaneleri kullanılmamıştır)
- Tüm matematiksel işlemler NumPy veya benzeri kütüphaneler olmadan uygulanmıştır
- Model yapısı ve eğitim algoritması tamamen Python ile kodlanmıştır
- Proje eğitsel amaçlı geliştirilmiştir ve aktif geliştirme aşamasındadır

## Lisans

MIT
