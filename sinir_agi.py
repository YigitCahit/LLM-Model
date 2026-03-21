# file name 'sinir_agi.py'
from rastgele import gauss_rastgele, tohum_uret

def agirlik_olustur(satirlar, sutunlar):
    degerler = gauss_rastgele(tohum_uret(), satirlar * sutunlar)
    return [
        [degerler[i * sutunlar + j] * 0.02 for j in range(sutunlar)]
        for i in range(satirlar)
    ]

def sifir_vektor(boyut):
    return [0.0] * boyut

def bir_vektor(boyut):
    return [1.0] * boyut

class KafaAgirliklari:
    def __init__(self, model_boyutu, kafa_boyutu):
        self.WQ = agirlik_olustur(model_boyutu, kafa_boyutu)
        self.WK = agirlik_olustur(model_boyutu, kafa_boyutu)
        self.WV = agirlik_olustur(model_boyutu, kafa_boyutu)

class AttentionAgirliklari:
    def __init__(self, kafa_sayisi, model_boyutu):
        kafa_boyutu = model_boyutu // kafa_sayisi
        self.kafalar = [KafaAgirliklari(model_boyutu, kafa_boyutu) for _ in range(kafa_sayisi)]
        self.WO = agirlik_olustur(model_boyutu, model_boyutu)
        self.bO = sifir_vektor(model_boyutu)

class FFAgirliklari:
    def __init__(self, model_boyutu, ff_boyutu):
        self.W1 = agirlik_olustur(model_boyutu, ff_boyutu)
        self.b1 = sifir_vektor(ff_boyutu)
        self.W2 = agirlik_olustur(ff_boyutu, model_boyutu)
        self.b2 = sifir_vektor(model_boyutu)

class KatmanAgirliklari:
    def __init__(self, kafa_sayisi, model_boyutu, ff_boyutu):
        self.attention = AttentionAgirliklari(kafa_sayisi, model_boyutu)
        self.ff        = FFAgirliklari(model_boyutu, ff_boyutu)
        self.gamma1    = bir_vektor(model_boyutu)
        self.beta1     = sifir_vektor(model_boyutu)
        self.gamma2    = bir_vektor(model_boyutu)
        self.beta2     = sifir_vektor(model_boyutu)

class SinirAgi:
    def __init__(self, sozluk_boyutu, model_boyutu, ff_boyutu, kafa_sayisi, katman_sayisi):
        self.sozluk_boyutu = sozluk_boyutu
        self.model_boyutu  = model_boyutu
        self.ff_boyutu     = ff_boyutu
        self.kafa_sayisi   = kafa_sayisi
        self.katman_sayisi = katman_sayisi

        self.embedding = agirlik_olustur(sozluk_boyutu, model_boyutu)
        self.katmanlar = [
            KatmanAgirliklari(kafa_sayisi, model_boyutu, ff_boyutu)
            for _ in range(katman_sayisi)
        ]
        self.cikti_W = agirlik_olustur(model_boyutu, sozluk_boyutu)
        self.cikti_b = sifir_vektor(sozluk_boyutu)