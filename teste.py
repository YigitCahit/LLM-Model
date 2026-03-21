from sinir_agi import SinirAgi
from islemler import embed, dogrusal, softmax
from egitim import dogrusal_gradyan, guncelle, vektor_guncelle, cross_entropy_kayip

def basit_ileri(ag, token_idx):
    x = [ag.embedding[token_idx]]  # [1 x model_boyutu]
    logitler = dogrusal(x, ag.cikti_W, ag.cikti_b)
    return softmax(logitler[0])

def basit_egitim(ag, girdi_idx, hedef_idx, ogrenme_hizi=0.01):
    olasiliklar = basit_ileri(ag, girdi_idx)
    kayip = cross_entropy_kayip(olasiliklar, hedef_idx)

    grad = olasiliklar[:]
    grad[hedef_idx] -= 1.0
    grad_matris = [grad]

    x = [ag.embedding[girdi_idx]]
    grad_x, grad_W, grad_b = dogrusal_gradyan(grad_matris, x, ag.cikti_W)

    ag.cikti_W = guncelle(ag.cikti_W, grad_W, ogrenme_hizi)
    ag.cikti_b = vektor_guncelle(ag.cikti_b, grad_b, ogrenme_hizi)

    # Embedding güncelle
    for j in range(len(ag.embedding[girdi_idx])):
        ag.embedding[girdi_idx][j] -= ogrenme_hizi * grad_x[0][j]

    return kayip

# Test
ag = SinirAgi(sozluk_boyutu=10, model_boyutu=16, ff_boyutu=32, kafa_sayisi=2, katman_sayisi=1)

MERHABA = 0
BUGÜN = 1

print("Egitim başlıyor: 'merhaba → bugün'")
for epoch in range(200):
    kayip = basit_egitim(ag, MERHABA, BUGÜN)
    if epoch % 20 == 0:
        olasiliklar = basit_ileri(ag, MERHABA)
        print(f"Epoch {epoch:3d} | Kayıp: {kayip:.4f} | P(bugün|merhaba): {olasiliklar[BUGÜN]:.4f}")