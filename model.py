# file name 'model.py'
from islemler import embed, pozisyonel_encoding, topla, transformer_blok, dogrusal, softmax
from sinir_agi import SinirAgi

def ileri_gecis(ag, token_indexleri):
    vektorler = embed(token_indexleri, ag.embedding)
    pe = pozisyonel_encoding(len(token_indexleri), ag.model_boyutu)
    x = topla(vektorler, pe)

    for katman in ag.katmanlar:
        x = transformer_blok(x, katman)

    son_token = [x[-1]]
    logitler = dogrusal(son_token, ag.cikti_W, ag.cikti_b)
    olasiliklar = softmax(logitler[0])
    return olasiliklar

def tahmin_et(ag, token_indexleri):
    olasiliklar = ileri_gecis(ag, token_indexleri)
    return olasiliklar.index(max(olasiliklar))