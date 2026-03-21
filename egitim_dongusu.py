# file name 'egitim_dongusu.py'
from sinir_agi import SinirAgi
from egitim import egitim_adimi
from veri import dosyadan_yukle, sozluk_olustur, egitim_ornekleri_olustur
from model import ileri_gecis

def egit(dosya_adi="veri.txt", epochs=10, ogrenme_hizi=0.001, pencere=4):
    cumleler = dosyadan_yukle(dosya_adi)
    kelime_sayi, sayi_kelime = sozluk_olustur(cumleler)
    ornekler = egitim_ornekleri_olustur(cumleler, kelime_sayi, pencere)

    sozluk_boyutu = len(kelime_sayi)
    print(f"Sozcuk sayisi: {sozluk_boyutu}")
    print(f"Egitim ornegi: {len(ornekler)}")

    ag = SinirAgi(
        sozluk_boyutu=sozluk_boyutu,
        model_boyutu=64,
        ff_boyutu=256,
        kafa_sayisi=4,
        katman_sayisi=2
    )

    for epoch in range(epochs):
        toplam_kayip = 0.0
        for i, (girdi, hedef) in enumerate(ornekler):
            if len(girdi) == 0:
                continue
            kayip = egitim_adimi(ag, girdi, hedef, ogrenme_hizi)
            toplam_kayip += kayip

            if i % 100 == 0:
                print(f"Adim {epoch+1} | Ornek {i}/{len(ornekler)} | Kayip: {kayip:.4f}")

        ort_kayip = toplam_kayip / len(ornekler)
        print(f"\nAdim {epoch+1} tamamlandi | Ortalama Kayip: {ort_kayip:.4f}\n")

    return ag, kelime_sayi, sayi_kelime

def uret(ag, kelime_sayi, sayi_kelime, baslangic_kelime, uzunluk=10, pencere=4, temperature=0.8):
    if baslangic_kelime not in kelime_sayi:
        print("Kelime bulunamadı.")
        return

    from rastgele import rastgele, tohum_uret

    indexler = [kelime_sayi[baslangic_kelime]]
    for _ in range(uzunluk):
        girdi = indexler[-pencere:]
        olasiliklar = ileri_gecis(ag, girdi)

        olasiliklar = [p ** (1 / temperature) for p in olasiliklar]
        toplam = sum(olasiliklar)
        olasiliklar = [p / toplam for p in olasiliklar]

        r = rastgele(tohum_uret(), 1)[0]
        kumulatif = 0.0
        tahmin = 0
        for j, p in enumerate(olasiliklar):
            kumulatif += p
            if r < kumulatif:
                tahmin = j
                break

        indexler.append(tahmin)

    print(" ".join(sayi_kelime[i] for i in indexler))