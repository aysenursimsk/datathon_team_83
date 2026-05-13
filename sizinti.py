import pandas as pd
import numpy as np

print("🔍 0.81'İN KAYNAĞI ARANIYOR: SIZINTI DEDEKTİFİ 2.0 🔍\n")

# 1. HAM VERİLERİ OKU (Temizlenmiş değil, orijinal dosyalar)
train = pd.read_csv('train.csv')
test = pd.read_csv('test_x.csv')

# 2. ID SIZINTISI KONTROLÜ (En muhtemel sebep)
# ID ile Hedef arasındaki matematiksel ilişkiyi ölçüyoruz
id_corr = train['id'].corr(train['bilissel_performans_skoru'])
print(f"-> ID ile Hedef arasındaki korelasyon: {id_corr:.4f}")

# Eğer bu sayı 0.10'dan büyükse, ID rastgele değildir, sızıntı vardır!
if abs(id_corr) > 0.10:
    print("🚨 KRİTİK: ID sütunu hedefle ilişkili! 0.81'in sırrı bu olabilir.")

# 3. SIRALAMA ANALİZİ
# Veri bilişsel skora göre mi dizilmiş?
train['skor_degisimi'] = train['bilissel_performans_skoru'].diff().abs()
print(f"-> Ortalama satırlar arası skor farkı: {train['skor_degisimi'].mean():.4f}")

# 4. DUPLICATE (TEKRARLANAN) VERİ ANALİZİ
# Acaba aynı özelliklere sahip satırlar mı var?
ozellikler = ['yas', 'cinsiyet', 'meslek', 'ulke', 'stres_skoru']
tekrarlar = train.duplicated(subset=ozellikler).sum()
print(f"-> Eğitim setindeki mükemmel tekrar sayısı: {tekrarlar}")

# 5. TEST SETİNDE 'KOPYA' VAR MI?
# Test setindeki bazı satırlar aslında Train setinde var mı?
print("\n-> Test setinin Train içinde gizli kopyaları aranıyor...")
merged = pd.merge(test, train, on=ozellikler, how='inner')
print(f"-> Test setinde olup Train setinde de AYNI OLAN satır sayısı: {len(merged)}")

if len(merged) > 0:
    print(f"💡 BULDUM! Test setindeki {len(merged)} satırın cevabı zaten Train setinde var.")
    print("Bu kişilerin skorlarını direkt kopyalayanlar 0.81'e iner.")

# 6. GİZLİ SİNYAL: KATEGORİK ORTALAMALAR
# Bazı meslekler veya ülkeler uçuk puanlar mı alıyor?
top_meslekler = train.groupby('meslek')['bilissel_performans_skoru'].mean().sort_values(ascending=False)
print("\nEn yüksek puanlı 3 meslek:")
print(top_meslekler.head(3))

print("\n==================================================")