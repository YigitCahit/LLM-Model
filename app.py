from egitim_dongusu import egit, uret
from kaydet import modeli_kaydet, modeli_yukle
import os

MODEL_DOSYASI = "model.json"

if os.path.exists(MODEL_DOSYASI):
    print("Model bulundu, yükleniyor...")
    ag, kelime_sayi, sayi_kelime = modeli_yukle(MODEL_DOSYASI)
else:
    print("Yeni model egitiliyor...")
    ag, kelime_sayi, sayi_kelime = egit(
        dosya_adi="veri.txt",
        epochs=100,
        ogrenme_hizi=0.001,
        pencere=4
    )
    modeli_kaydet(ag, kelime_sayi, sayi_kelime, MODEL_DOSYASI)

uret(ag, kelime_sayi, sayi_kelime, baslangic_kelime="merhaba", uzunluk=10)