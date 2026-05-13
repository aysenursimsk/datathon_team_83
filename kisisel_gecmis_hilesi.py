import pandas as pd
import numpy as np

print("🧬 KİŞİSEL GEÇMİŞ (REPEATED USERS) HİLESİ BAŞLATILDI 🧬\n")

# 1. Verileri Okuma
train = pd.read_csv('train_temiz.csv')
test = pd.read_csv('test_temiz.csv')

# En iyi modelimizin tahminleri (Ana Kasa)
try:
    sub = pd.read_csv('submission_ULTIMATE_GRANDMASTER_MLP.csv')
except FileNotFoundError:
    print("HATA: 'submission_ULTIMATE_GRANDMASTER_MLP.csv' bulunamadı!")
    exit()

# 2. Değişmez Kimlik Sütunları (İnsanın Parmak İzi)
kimlik_sutunlari = ['yas', 'cinsiyet', 'meslek', 'ulke', 'vucut_kitle_indeksi']

# 3. Eğitim Setinden Her "Kişinin" Kendi Ortalama Puanını Bulma
kisi_ortalama = train.groupby(kimlik_sutunlari)['bilissel_performans_skoru'].agg(['mean', 'count']).reset_index()
kisi_ortalama.rename(columns={'mean': 'KISI_GECMIS_ORTALAMASI', 'count': 'TRAIN_GUN_SAYISI'}, inplace=True)

# Sadece Train'de en az 2 günü olan kişileri güvenilir kabul edelim
kisi_ortalama = kisi_ortalama[kisi_ortalama['TRAIN_GUN_SAYISI'] >= 2]

# 4. Test Setindeki Tanıdık Kişileri Bulma
test_kopya = test.copy()
test_kopya = pd.merge(test_kopya, kisi_ortalama, on=kimlik_sutunlari, how='left')

# 5. Hack İşlemi: Tahminleri Kişisel Geçmişle Harmanlama
# Tamamen silmek yerine, %80 kişinin kendi geçmişi + %20 Yapay Zeka yapalım
eslesen_indeksler = test_kopya['KISI_GECMIS_ORTALAMASI'].notnull()
eslesme_sayisi = eslesen_indeksler.sum()

eski_tahminler = sub['bilissel_performans_skoru'].copy()
yeni_tahminler = eski_tahminler.copy()

gecmis_skorlar = test_kopya.loc[eslesen_indeksler, 'KISI_GECMIS_ORTALAMASI']
ai_skorlar = eski_tahminler[eslesen_indeksler]

# Büyü burada gerçekleşiyor:
yeni_tahminler[eslesen_indeksler] = (gecmis_skorlar * 0.85) + (ai_skorlar * 0.15)

print(f"🔥 BİNGÖ! Test setinde Train'den tanıdığımız tam {eslesme_sayisi} satır (gün) yakalandı!")
print(f"-> Bu {eslesme_sayisi} test satırı için AI tahminleri %85 oranında kişinin 'Tarihi Ortalaması' ile değiştirildi.")

# 6. Kaydetme
sub['bilissel_performans_skoru'] = np.clip(yeni_tahminler, 0.0, 10.0)
dosya_adi = 'submission_REPEATED_USERS_HACK.csv'
sub.to_csv(dosya_adi, index=False)

print(f"\n🚀 SİLAHLANDIRILMIŞ DOSYA HAZIR: {dosya_adi}")