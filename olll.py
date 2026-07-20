# -*- coding: utf-8 -*-
"""
Ollama (yerel LLM) ile arıza açıklamalarını Mekanik/Otomasyon olarak
sınıflandıran script. Ara-kayıt (checkpoint) ve devam etme (resume)
özellikli, 29k gibi büyük dosyalar için güvenli hale getirilmiştir.

Kullanım:
    1) Ollama'nın çalıştığından emin ol (ollama serve) ve modeli çekmiş ol
       (ollama pull qwen2.5:7b).
    2) CSV_DOSYASI, METIN_KOLONU değerlerini kontrol et.
    3) İlk çalıştırmadan önce TEST_MODU=True ile küçük bir örnekte dene.
    4) python ollama_kategorize.py
    5) Script yarıda kesilirse TEKRAR çalıştır — otomatik kaldığı yerden
       devam eder (aynı OUTPUT_DOSYASI'nı okur).
"""

import os
import time
import pandas as pd
import requests
from tqdm import tqdm

# ------------------- AYARLAR -------------------
CSV_DOSYASI = "converted_43.csv"
OUTPUT_DOSYASI = "sonuc_arizalar.csv"
METIN_KOLONU = "Talep Açıklaması"
LABEL_KOLONU = "İş Tipi"          # varsa: zaten MEK/OTO dolu satırlar atlanır
KATEGORI_KOLONU = "Kategori"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_ADI = "qwen2.5:7b"

TEST_MODU = True        # True: sadece ilk TEST_SATIR_SAYISI satırı işler
TEST_SATIR_SAYISI = 10
KAYIT_ARALIGI = 100      # her N satırda bir diske yaz (çökme durumunda kayıp azalır)
TIMEOUT_SANIYE = 60
MAX_DENEME = 3
# ------------------------------------------------

PROMPT_TEMPLATE = """Sen bir endüstriyel bakım ve otomasyon uzmanısın. Görevin, sana verilen arıza/bakım metinlerini "Otomasyon (Elektrik/Elektronik)" veya "Mekanik" olarak sınıflandırmaktır.

Sınıflandırma Kuralları:
1. OTOMASYON (ELEKTRİK/ELEKTRONİK):
   - Sinyal, veri, kontrol, yazılım, parametre ve pano içi elektriksel bileşenler (PLC, SCADA, sensör, inverter, röle, sigorta, MCC, haberleşme vb.).
   - Elektrikle çalışan cihazların kendisi veya iç arızaları (Elektrik motoru sargı/besleme arızası, fan motoru arızası, aydınlatma, switch, klemens vb.).
   - Yazım hataları veya kısaltmalar olsa dahi ana bileşene odaklan.

2. MEKANİK:
   - Fiziksel aşınma, kırılma, hizalama, montaj, demontaj, kaynak, torna, yağlama ve sızdırmazlık işlemleri.
   - Güç aktarım organları (Redüktör, rulman, dişli, kayış, kasnak, şaft, zincir, kaplin).
   - Akışkan gücü ve tesisat elemanları (Pnomatik/hidrolik silindirler, valfler, vanalar, boru, hortum, keçe, conta).
   - Yapısal parçalar (Şasi, kapak, cıvata, platform, tank, bıçak, muhafaza).
   NOT = Nadir de olsa ikisine de uygun olmadığını düşünürsen "MEK" olarak sınıflandır.

Yanıt Formatı:
Sadece şu iki değerden birini döndür (ekstra açıklama yapma):
- OTO
- MEK

Arıza Açıklaması: {aciklama}
Kategori:"""


def csv_oku(dosya_yolu: str) -> pd.DataFrame:
    for encoding in ["utf-8-sig", "cp1254", "iso-8859-9", "latin1"]:
        for sep in [";", ",", "\t"]:
            try:
                df = pd.read_csv(
                    dosya_yolu, sep=sep, encoding=encoding,
                    low_memory=False, on_bad_lines="skip",
                )
                if len(df.columns) > 3:  # gerçekçi bir eşik: tek kolonla yanlış ayraç ele
                    df.columns = [c.strip() for c in df.columns]
                    print(f"CSV okundu -> ayraç: '{sep}', kodlama: '{encoding}', "
                          f"{df.shape[0]} satır, {df.shape[1]} sütun")
                    return df
            except Exception:
                continue
    raise RuntimeError("CSV okunamadı. Dosya yolunu/yapısını kontrol et.")


def ollama_sor(aciklama: str) -> str:
    payload = {
        "model": MODEL_ADI,
        "prompt": PROMPT_TEMPLATE.format(aciklama=aciklama),
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 10,
        },
    }
    for attempt in range(1, MAX_DENEME + 1):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_SANIYE)
            if r.status_code == 200:
                cevap = r.json().get("response", "").strip().upper()
                
                # Modelden ne gelirse gelsin tek bir standarta eşle:
                if any(k in cevap for k in ["OTO", "OTOMASYON", "ELEKTRİK", "ELEKTRONİK"]):
                    return "OTO"
                elif any(k in cevap for k in ["MEK", "MEKANİK"]):
                    return "MEK"
                else:
                    return "MEK" # Prompt'taki kurala paralel olarak belirsizse MEK varsayıyoruz
                    
        except requests.exceptions.RequestException:
            pass
        time.sleep(1.5 * attempt)
        
    return "Bağlantı Hatası"


def main():
    if os.path.exists(OUTPUT_DOSYASI):
        print(f"Devam ediliyor: mevcut '{OUTPUT_DOSYASI}' bulundu, kaldığı yerden sürecek.")
        df = pd.read_csv(OUTPUT_DOSYASI, sep=";", encoding="utf-8-sig")
    else:
        df = csv_oku(CSV_DOSYASI)
        if KATEGORI_KOLONU not in df.columns:
            df[KATEGORI_KOLONU] = ""

    if METIN_KOLONU not in df.columns:
        raise ValueError(
            f"'{METIN_KOLONU}' sütunu bulunamadı. Mevcut sütunlar: {list(df.columns)}"
        )

    if TEST_MODU:
        print(f"*** TEST MODU: sadece ilk {TEST_SATIR_SAYISI} satır işlenecek ***")
        calisilacak = df.head(TEST_SATIR_SAYISI).index
    else:
        calisilacak = df.index

    zaten_var = LABEL_KOLONU in df.columns
    islenen_sayaci = 0

    for idx in tqdm(calisilacak, total=len(calisilacak)):
        mevcut_kategori = str(df.at[idx, KATEGORI_KOLONU]).strip()
        if mevcut_kategori not in ("", "nan"):
            continue  # bu satır zaten işlenmiş (resume)

        if zaten_var:
            mevcut_label = str(df.at[idx, LABEL_KOLONU]).strip().upper()
            if mevcut_label == "MEK":
                df.at[idx, KATEGORI_KOLONU] = "Mekanik"
                continue
            if mevcut_label == "OTO":
                df.at[idx, KATEGORI_KOLONU] = "Otomasyon"
                continue

        aciklama = str(df.at[idx, METIN_KOLONU])
        if not aciklama.strip() or aciklama.strip().lower() == "nan":
            df.at[idx, KATEGORI_KOLONU] = "Belirsiz"
            continue

        df.at[idx, KATEGORI_KOLONU] = ollama_sor(aciklama)
        islenen_sayaci += 1

        if islenen_sayaci % KAYIT_ARALIGI == 0:
            df.to_csv(OUTPUT_DOSYASI, index=False, encoding="utf-8-sig", sep=";")

    df.to_csv(OUTPUT_DOSYASI, index=False, encoding="utf-8-sig", sep=";")
    print(f"\nİşlem tamamlandı! '{OUTPUT_DOSYASI}' dosyası güncellendi.")
    print(df[KATEGORI_KOLONU].value_counts())


if __name__ == "__main__":
    main()