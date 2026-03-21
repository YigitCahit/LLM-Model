# file name 'islemler.py'
from rastgele import cos, PI

def carpim(A, B):
    satir_A = len(A)
    sutun_A = len(A[0])
    satir_B = len(B)
    sutun_B = len(B[0])

    if sutun_A != satir_B:
        raise ValueError("A'nin sutun sayisi B'nin satir sayisina esit olmali!")
    
    sonuc = [[0 for _ in range(sutun_B)] for _ in range(satir_A)]

    for i in range(satir_A):
        for j in range(sutun_B):
            for k in range(sutun_A):
                sonuc[i][j] += A[i][k] * B[k][j]
    
    return sonuc

def softmax(vektor):
    e = 2.718281828459
    maks = max(vektor)
    ustel_degerler = [e ** (x - maks) for x in vektor]
    toplam = sum(ustel_degerler)
    return [deger / toplam for deger in ustel_degerler]

def transpoze(matris):
    satir_sayisi = len(matris)
    sutun_sayisi = len(matris[0])
    yeni_matris = [[0 for _ in range(satir_sayisi)] for _ in range(sutun_sayisi)]
    for i in range(satir_sayisi):
        for j in range(sutun_sayisi):
            yeni_matris[j][i] = matris[i][j]
    return yeni_matris

def sin(x):
    return cos(x - PI / 2)

def topla(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def embed(token_indexleri, embedding_tablosu):
    return [embedding_tablosu[i] for i in token_indexleri]

def pozisyonel_encoding(uzunluk, boyut):
    pe = [[0.0] * boyut for _ in range(uzunluk)]
    for pos in range(uzunluk):
        for i in range(0, boyut, 2):
            pe[pos][i] = sin(pos / (10000 ** (i / boyut)))
            if i + 1 < boyut:
                pe[pos][i+1] = cos(pos / (10000 ** (i / boyut)))
    return pe

def dogrusal(x, W, b):
    sonuc = carpim(x, W)
    return [[sonuc[i][j] + b[j] for j in range(len(b))] for i in range(len(sonuc))]

def gelu(x):
    PI = 3.141592653589793
    k = (2 / PI) ** 0.5
    ic = k * (x + 0.044715 * x**3)
    ic = max(-20.0, min(20.0, ic))
    e = 2.718281828459
    exp2 = e ** (2 * ic)
    tanh_ic = (exp2 - 1) / (exp2 + 1)
    return 0.5 * x * (1 + tanh_ic)

def katman_norm(x, gamma, beta, eps=1e-5):
    sonuc = []
    for satir in x:
        ortalama = sum(satir) / len(satir)
        varyans = sum((v - ortalama) ** 2 for v in satir) / len(satir)
        normalize = [(v - ortalama) / ((varyans + eps) ** 0.5) for v in satir]
        sonuc.append([gamma[i] * normalize[i] + beta[i] for i in range(len(satir))])
    return sonuc

def attention(Q, K, V):
    d = len(K[0])
    olcek = d ** 0.5
    kt = transpoze(K)
    skorlar = carpim(Q, kt)
    olceklenmis = [[s / olcek for s in satir] for satir in skorlar]
    agirliklar = [softmax(satir) for satir in olceklenmis]
    return carpim(agirliklar, V)

def cok_baslikli_attention(x, att_agirliklar):
    kafalar = []
    for kafa in att_agirliklar.kafalar:
        Q = carpim(x, kafa.WQ)
        K = carpim(x, kafa.WK)
        V = carpim(x, kafa.WV)
        kafalar.append(attention(Q, K, V))

    birlesmis = [[] for _ in range(len(x))]
    for kafa in kafalar:
        for i in range(len(x)):
            birlesmis[i].extend(kafa[i])

    return dogrusal(birlesmis, att_agirliklar.WO, att_agirliklar.bO)

def ileri_besleme(x, ff_agirliklar):
    ara = dogrusal(x, ff_agirliklar.W1, ff_agirliklar.b1)
    aktivasyon = [[gelu(d) for d in satir] for satir in ara]
    return dogrusal(aktivasyon, ff_agirliklar.W2, ff_agirliklar.b2)

def transformer_blok(x, katman_agirliklar):
    att_cikti = cok_baslikli_attention(x, katman_agirliklar.attention)
    x = katman_norm(topla(x, att_cikti), katman_agirliklar.gamma1, katman_agirliklar.beta1)

    ff_cikti = ileri_besleme(x, katman_agirliklar.ff)
    x = katman_norm(topla(x, ff_cikti), katman_agirliklar.gamma2, katman_agirliklar.beta2)
    return x