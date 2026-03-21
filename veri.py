# file name 'veri.py'
def dosyadan_yukle(dosya_adi="veri.txt"):
    bitis = ".!?\n"
    noktalamalar = ",;:\"'()[]{}-_/#*"

    with open(dosya_adi, "r", encoding="utf-8") as f:
        metin = f.read()

    metin = metin.replace("İ", "i").replace("I", "ı").replace("Ş", "ş") \
                 .replace("Ğ", "ğ").replace("Ü", "ü").replace("Ö", "ö") \
                 .replace("Ç", "ç").lower()

    for isaret in bitis:
        metin = metin.replace(isaret, "|")
    for isaret in noktalamalar:
        metin = metin.replace(isaret, " ")

    cumleler = [c.split() for c in metin.split("|") if c.split()]
    return cumleler

def sozluk_olustur(cumleler):
    kelime_sayi = {}
    sayi_kelime = {}
    sayac = 0
    for cumle in cumleler:
        for kelime in cumle:
            if kelime not in kelime_sayi:
                kelime_sayi[kelime] = sayac
                sayi_kelime[sayac] = kelime
                sayac += 1
    return kelime_sayi, sayi_kelime

def egitim_ornekleri_olustur(cumleler, kelime_sayi, pencere=4):
    ornekler = []
    for cumle in cumleler:
        indexler = [kelime_sayi[k] for k in cumle if k in kelime_sayi]
        for i in range(1, len(indexler)):
            girdi = indexler[max(0, i - pencere):i]
            hedef = indexler[i]
            ornekler.append((girdi, hedef))
    return ornekler