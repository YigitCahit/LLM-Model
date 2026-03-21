# file name 'kaydet.py'
def modeli_kaydet(ag, kelime_sayi, sayi_kelime, dosya_adi="model.json"):
    import json
    
    veri = {
        "sozluk_boyutu": ag.sozluk_boyutu,
        "model_boyutu": ag.model_boyutu,
        "ff_boyutu": ag.ff_boyutu,
        "kafa_sayisi": ag.kafa_sayisi,
        "katman_sayisi": ag.katman_sayisi,
        "kelime_sayi": kelime_sayi,
        "sayi_kelime": {str(k): v for k, v in sayi_kelime.items()},
        "embedding": ag.embedding,
        "cikti_W": ag.cikti_W,
        "cikti_b": ag.cikti_b,
        "katmanlar": []
    }

    for katman in ag.katmanlar:
        katman_veri = {
            "gamma1": katman.gamma1,
            "beta1": katman.beta1,
            "gamma2": katman.gamma2,
            "beta2": katman.beta2,
            "ff": {
                "W1": katman.ff.W1,
                "b1": katman.ff.b1,
                "W2": katman.ff.W2,
                "b2": katman.ff.b2,
            },
            "attention": {
                "WO": katman.attention.WO,
                "bO": katman.attention.bO,
                "kafalar": [
                    {
                        "WQ": kafa.WQ,
                        "WK": kafa.WK,
                        "WV": kafa.WV,
                    }
                    for kafa in katman.attention.kafalar
                ]
            }
        }
        veri["katmanlar"].append(katman_veri)

    with open(dosya_adi, "w", encoding="utf-8") as f:
        json.dump(veri, f, ensure_ascii=False)

    print(f"Model kaydedildi: {dosya_adi}")


def modeli_yukle(dosya_adi="model.json"):
    import json
    from sinir_agi import SinirAgi, KatmanAgirliklari, FFAgirliklari, AttentionAgirliklari, KafaAgirliklari

    with open(dosya_adi, "r", encoding="utf-8") as f:
        veri = json.load(f)

    ag = SinirAgi(
        sozluk_boyutu=veri["sozluk_boyutu"],
        model_boyutu=veri["model_boyutu"],
        ff_boyutu=veri["ff_boyutu"],
        kafa_sayisi=veri["kafa_sayisi"],
        katman_sayisi=veri["katman_sayisi"]
    )

    ag.embedding = veri["embedding"]
    ag.cikti_W   = veri["cikti_W"]
    ag.cikti_b   = veri["cikti_b"]

    for i, katman_veri in enumerate(veri["katmanlar"]):
        ag.katmanlar[i].gamma1 = katman_veri["gamma1"]
        ag.katmanlar[i].beta1  = katman_veri["beta1"]
        ag.katmanlar[i].gamma2 = katman_veri["gamma2"]
        ag.katmanlar[i].beta2  = katman_veri["beta2"]

        ag.katmanlar[i].ff.W1 = katman_veri["ff"]["W1"]
        ag.katmanlar[i].ff.b1 = katman_veri["ff"]["b1"]
        ag.katmanlar[i].ff.W2 = katman_veri["ff"]["W2"]
        ag.katmanlar[i].ff.b2 = katman_veri["ff"]["b2"]

        for j, kafa_veri in enumerate(katman_veri["attention"]["kafalar"]):
            ag.katmanlar[i].attention.kafalar[j].WQ = kafa_veri["WQ"]
            ag.katmanlar[i].attention.kafalar[j].WK = kafa_veri["WK"]
            ag.katmanlar[i].attention.kafalar[j].WV = kafa_veri["WV"]

        ag.katmanlar[i].attention.WO = katman_veri["attention"]["WO"]
        ag.katmanlar[i].attention.bO = katman_veri["attention"]["bO"]

    kelime_sayi  = veri["kelime_sayi"]
    sayi_kelime  = {int(k): v for k, v in veri["sayi_kelime"].items()}

    print(f"Model yüklendi: {dosya_adi}")
    return ag, kelime_sayi, sayi_kelime