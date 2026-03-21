# file name 'rastgele.py'
import time

PI = 3.141592653589793
COS_TABLOSU = []
HASSASIYET = 2000

def log(x, iterasyon=100):
    if x <= 0:
        raise ValueError("log tanimsiz")
    y = (x - 1) / (x + 1)
    y2 = y * y
    sonuc = 0.0
    for i in range(iterasyon):
        n = 2 * i + 1
        sonuc += (y ** n) / n
    return 2 * sonuc

def cos(x):
    x = x % (2 * PI)
    if x > PI:
        x -= 2 * PI
    
    sonuc = 0.0
    terim = 1.0
    for i in range(1, 20):
        sonuc += terim
        terim *= -x * x / ((2 * i - 1) * (2 * i))
    return sonuc

def tohum_uret():
    t = time.time()
    mikrosaniye = int(t * 1_000_000)
    
    a = mikrosaniye & 0xFFFF
    b = (mikrosaniye >> 16) & 0xFFFF
    c = (mikrosaniye >> 32) & 0xFFFF
    
    karisik = (a * b * c) ^ (a << 3) ^ (b >> 2) ^ c
    return karisik

def rastgele(tohum, n):
    A = 1664525
    C = 1013904223
    M = 2**32
    
    sonuclar = []
    durum = tohum
    for _ in range(n):
        durum = (A * durum + C) % M
        sonuclar.append(durum / M)
    return sonuclar

def gauss_rastgele(tohum, n):
    tekil = rastgele(tohum, n * 2 + 2)
    sonuclar = []
    for i in range(0, n * 2, 2):
        u1 = max(tekil[i], 1e-10)
        u2 = tekil[i+1]
        ln = -2 * log(u1)
        kok = ln ** 0.5
        cos_deg = cos(2 * PI * u2)
        sonuclar.append(kok * cos_deg)
    return sonuclar[:n]