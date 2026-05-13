import pandas as pd
import numpy as np

print("🔍 FORMÜL AVCISI 4.0: 0.22 ŞİFRESİNİ KIRMA OPERASYONU 🔍\n")

# 1. HAM VERİYİ OKU VE TEMİZLE
train = pd.read_csv('train.csv')
hedef_kolon = 'bilissel_performans_skoru'

# Sayısal sütunları seç ve zorla sayıya çevir (Hata buradan geliyordu)
num_cols = ['yas', 'vucut_kitle_indeksi', 'rem_yuzdesi', 'derin_uyku_yuzdesi', 
            'uykuya_dalma_suresi_dk', 'gecelik_uyanma_sayisi', 'uyku_oncesi_kafein_mg', 
            'uyku_oncesi_ekran_suresi_dk', 'gunluk_adim_sayisi', 'sekerleme_suresi_dk', 
            'stres_skoru', 'gunluk_calisma_saati', 'dinlenik_nabiz_bpm', 'oda_sicakligi_celsius']

# Veriyi temizle (NaN olan satırları bu işlem için geçici olarak atalım)
df = train[num_cols + [hedef_kolon]].copy()
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna().reset_index(drop=True)
y = df[hedef_kolon]

results = []

print(f"-> {len(df)} temiz satır üzerinde binlerce matematiksel olasılık taranıyor...")

# 2. BRUTE-FORCE DENKLEM ARAMA
for c1 in num_cols:
    for c2 in num_cols:
        if c1 == c2: continue
        
        # Temel Oranlar (A / B)
        f1 = df[c1] / (df[c2] + 1e-6)
        # Temel Çarpımlar (A * B)
        f2 = df[c1] * df[c2]
        
        results.append((f"{c1} / {c2}", np.abs(np.corrcoef(f1, y)[0,1])))
        results.append((f"{c1} * {c2}", np.abs(np.corrcoef(f2, y)[0,1])))
        
        # 3'lü Kombinasyonlar (Zirve Taktiği: (A * B) / C)
        for c3 in num_cols:
            if c3 in [c1, c2]: continue
            f3 = (df[c1] * df[c2]) / (df[c3] + 1e-6)
            results.append((f"({c1} * {c2}) / {c3}", np.abs(np.corrcoef(f3, y)[0,1])))

# Korelasyonları temizle ve sırala
results = [r for r in results if not np.isnan(r[1])]
results.sort(key=lambda x: x[1], reverse=True)

print("\n--- 0.22 SKORUNU GETİRECEK MUHTEMEL GİZLİ DENKLEMLER ---")
for formula, score in results[:20]:
    if score > 0.85:
        print(f"💎 BULDUM! -> {formula:<45} | GÜÇ: %{score*100:.2f}")
    elif score > 0.70:
        print(f"🔥 ÇOK GÜÇLÜ -> {formula:<45} | GÜÇ: %{score*100:.2f}")
    else:
        print(f"Formül: {formula:<45} | Güç: %{score*100:.2f}")

print("\n==================================================")