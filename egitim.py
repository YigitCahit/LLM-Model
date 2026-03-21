# file name 'egitim.py'
from model import ileri_gecis
from islemler import carpim, transpoze, embed, pozisyonel_encoding, transformer_blok, dogrusal, topla, katman_norm, cok_baslikli_attention

def cross_entropy_kayip(olasiliklar, hedef_index):
    p = max(olasiliklar[hedef_index], 1e-10)
    y = (p - 1) / (p + 1)
    y2 = y * y
    sonuc = 0.0
    for i in range(100):
        n = 2 * i + 1
        sonuc += (y ** n) / n
    return -2 * sonuc

def softmax_cross_entropy_gradyan(olasiliklar, hedef_index):
    grad = olasiliklar[:]
    grad[hedef_index] -= 1.0
    return grad

def dogrusal_gradyan(grad_cikti, x, W):
    grad_x = carpim(grad_cikti, transpoze(W))
    grad_W = carpim(transpoze(x), grad_cikti)
    grad_b = [sum(grad_cikti[i][j] for i in range(len(grad_cikti)))
              for j in range(len(grad_cikti[0]))]
    return grad_x, grad_W, grad_b

def guncelle(agirlik, grad, ogrenme_hizi):
    return [
        [agirlik[i][j] - ogrenme_hizi * grad[i][j] for j in range(len(agirlik[0]))]
        for i in range(len(agirlik))
    ]

def vektor_guncelle(vektor, grad, ogrenme_hizi):
    return [vektor[i] - ogrenme_hizi * grad[i] for i in range(len(vektor))]

def egitim_adimi(ag, token_indexleri, hedef_index, ogrenme_hizi=0.001):
    from islemler import embed, pozisyonel_encoding, topla, transformer_blok

    olasiliklar = ileri_gecis(ag, token_indexleri)
    kayip = cross_entropy_kayip(olasiliklar, hedef_index)

    grad = softmax_cross_entropy_gradyan(olasiliklar, hedef_index)
    grad_matris = [grad]

    vektorler = embed(token_indexleri, ag.embedding)
    pe = pozisyonel_encoding(len(token_indexleri), ag.model_boyutu)
    x = topla(vektorler, pe)

    ara_x = [x]
    for katman in ag.katmanlar:
        x = transformer_blok(x, katman)
        ara_x.append(x)

    son_token = [ara_x[-1][-1]]
    grad_x, grad_W, grad_b = dogrusal_gradyan(grad_matris, son_token, ag.cikti_W)
    ag.cikti_W = guncelle(ag.cikti_W, grad_W, ogrenme_hizi)
    ag.cikti_b = vektor_guncelle(ag.cikti_b, grad_b, ogrenme_hizi)
    son_token_grad = grad_x[0]
    grad_katman = [son_token_grad for _ in range(len(token_indexleri))]

    for i in reversed(range(len(ag.katmanlar))):
        grad_katman = transformer_blok_gradyan(
            grad_katman,
            ara_x[i],
            ag.katmanlar[i]
        )

    return kayip

def gelu_gradyan(x):
    PI = 3.141592653589793
    k = (2 / PI) ** 0.5
    ic = k * (x + 0.044715 * x**3)
    ic = max(-20.0, min(20.0, ic))
    e = 2.718281828459
    exp2 = e ** (2 * ic)
    tanh_ic = (exp2 - 1) / (exp2 + 1)
    
    dtanh = 1 - tanh_ic ** 2
    dic = k * (1 + 3 * 0.044715 * x**2)
    return 0.5 * (1 + tanh_ic) + 0.5 * x * dtanh * dic

def ileri_besleme_gradyan(grad_cikti, x, ff_agirliklar):
    ara = [[0.0] * len(ff_agirliklar.b1) for _ in range(len(x))]
    ara_aktif = dogrusal(x, ff_agirliklar.W1, ff_agirliklar.b1)
    aktivasyon = [[max(0.0, v) for v in satir] for satir in ara_aktif]  

    grad_x2, grad_W2, grad_b2 = dogrusal_gradyan(grad_cikti, aktivasyon, ff_agirliklar.W2)

    grad_aktif = [
        [grad_x2[i][j] * gelu_gradyan(ara_aktif[i][j]) for j in range(len(ara_aktif[0]))]
        for i in range(len(ara_aktif))
    ]

    grad_x1, grad_W1, grad_b1 = dogrusal_gradyan(grad_aktif, x, ff_agirliklar.W1)

    return grad_x1, grad_W1, grad_b1, grad_W2, grad_b2

def transformer_blok_gradyan(grad_cikti, x, katman_agirliklar):
    if isinstance(grad_cikti[0], (int, float)):
        grad_cikti = [grad_cikti]

    att_cikti = cok_baslikli_attention(x, katman_agirliklar.attention)
    x_att = katman_norm(topla(x, att_cikti), katman_agirliklar.gamma1, katman_agirliklar.beta1)

    grad_ln2 = []
    grad_gamma2_toplam = [0.0] * len(katman_agirliklar.gamma2)
    grad_beta2_toplam  = [0.0] * len(katman_agirliklar.beta2)

    for i in range(len(grad_cikti)):
        gx, gg, gb = katman_norm_gradyan(grad_cikti[i], x_att[i], katman_agirliklar.gamma2)
        grad_ln2.append(gx)
        for j in range(len(gg)):
            grad_gamma2_toplam[j] += gg[j]
            grad_beta2_toplam[j]  += gb[j]

    katman_agirliklar.gamma2 = vektor_guncelle(katman_agirliklar.gamma2, grad_gamma2_toplam, 0.001)
    katman_agirliklar.beta2  = vektor_guncelle(katman_agirliklar.beta2,  grad_beta2_toplam,  0.001)

    grad_ff_x, gW1, gb1, gW2, gb2 = ileri_besleme_gradyan(grad_ln2, x_att, katman_agirliklar.ff)
    katman_agirliklar.ff.W1 = guncelle(katman_agirliklar.ff.W1, gW1, 0.001)
    katman_agirliklar.ff.b1 = vektor_guncelle(katman_agirliklar.ff.b1, gb1, 0.001)
    katman_agirliklar.ff.W2 = guncelle(katman_agirliklar.ff.W2, gW2, 0.001)
    katman_agirliklar.ff.b2 = vektor_guncelle(katman_agirliklar.ff.b2, gb2, 0.001)

    grad_x_att = [
        [grad_ff_x[i][j] + grad_ln2[i][j] for j in range(len(grad_ff_x[0]))]
        for i in range(len(grad_ff_x))
    ]

    grad_ln1 = []
    grad_gamma1_toplam = [0.0] * len(katman_agirliklar.gamma1)
    grad_beta1_toplam  = [0.0] * len(katman_agirliklar.beta1)

    for i in range(len(grad_x_att)):
        gx, gg, gb = katman_norm_gradyan(grad_x_att[i], x[i], katman_agirliklar.gamma1)
        grad_ln1.append(gx)
        for j in range(len(gg)):
            grad_gamma1_toplam[j] += gg[j]
            grad_beta1_toplam[j]  += gb[j]

    katman_agirliklar.gamma1 = vektor_guncelle(katman_agirliklar.gamma1, grad_gamma1_toplam, 0.001)
    katman_agirliklar.beta1  = vektor_guncelle(katman_agirliklar.beta1,  grad_beta1_toplam,  0.001)

    return grad_ln1

def katman_norm_gradyan(grad_cikti, x, gamma, eps=1e-5):
    n = len(x)
    ortalama = sum(x) / n
    varyans = sum((v - ortalama) ** 2 for v in x) / n
    std = (varyans + eps) ** 0.5

    x_norm = [(v - ortalama) / std for v in x]

    grad_gamma = [grad_cikti[i] * x_norm[i] for i in range(n)]
    grad_beta  = grad_cikti[:]

    grad_x_norm = [grad_cikti[i] * gamma[i] for i in range(n)]

    grad_varyans  = sum(grad_x_norm[i] * (x[i] - ortalama) for i in range(n)) * (-0.5) * (varyans + eps) ** (-1.5)
    grad_ortalama = sum(-grad_x_norm[i] / std for i in range(n)) + grad_varyans * sum(-2 * (x[i] - ortalama) for i in range(n)) / n

    grad_x = [
        grad_x_norm[i] / std +
        grad_varyans * 2 * (x[i] - ortalama) / n +
        grad_ortalama / n
        for i in range(n)
    ]

    return grad_x, grad_gamma, grad_beta