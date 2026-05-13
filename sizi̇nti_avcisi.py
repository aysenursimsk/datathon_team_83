import pandas as pd
import numpy as np
import itertools

print("🕵️‍♂️ KAGGLE SIZINTI VE MAGIC FEATURE AVCI BOTU BAŞLATILDI 🕵️‍♂️\n")

train = pd.read_csv('train_temiz.csv')
test = pd.read_csv('test_temiz.csv')

# 1. KİMLİK (IDENTITY) SIZINTISI AVI
print("1. KİMLİK SIZINTISI KONTROLÜ (Test setindekiler aslında Train'dekiler mi?)")
kimlik_sutunlari = ['yas', 'cinsiyet', 'meslek', 'ulke', 'vucut_kitle_indeksi']

train_kimlik = train.groupby(kimlik_sutunlari).size().reset_index(name='train_sayi')
test_kimlik = test.groupby(kimlik_sutunlari).size().reset_index(name='test_sayi')

ortak_kisiler = pd.merge(train_kimlik, test_kimlik, on=kimlik_sutunlari, how='inner')
print(f"Train ve Test setinde TAMAMEN AYNI profile sahip eşsiz grup sayısı: {len(ortak_kisiler)}")
if len(ortak_kisiler) > 0:
    print("🚨 TEHLİKE: Test setindeki bazı kişiler Train'de zaten var! Skoru direkt kopyalayabiliriz.")

# 2. SIFIR VARYANS (KESİN KURAL) AVI
print("\n2. SIFIR VARYANS AVI (Team 136'nın bulduğu Magic Feature aranıyor...)")
# Hangi kategorik kombinasyonlar performansı %100 kesinlikle belirliyor?
kategorikler = ['meslek', 'ulke', 'kronotip', 'ruh_sagligi_durumu', 'gun_tipi', 'mevsim']

bulunan_kural_sayisi = 0
for r in range(2, 4): # İkili ve Üçlü kombinasyonları dene
    for combo in itertools.combinations(kategorikler, r):
        grup = train.groupby(list(combo))['bilissel_performans_skoru']
        
        # Grubun standart sapması 0 ise (yani gruptaki herkesin puanı virgüline kadar aynıysa)
        sifir_sapma = grup.std()[grup.std() == 0]
        
        # Grupta en az 3 kişi olsun ki tesadüf olmasın
        kisi_sayisi = grup.count()
        gecerli_kurallar = sifir_sapma.index[kisi_sayisi[sifir_sapma.index] > 3]
        
        if len(gecerli_kurallar) > 0:
            bulunan_kural_sayisi += 1
            print(f"\n🔥 MAGIC SİNYAL BULUNDU! Kombinasyon: {combo}")
            print(f"{len(gecerli_kurallar)} farklı alt grupta skor %100 KESİN.")
            break # Konsolu boğmamak için ilk bulduğunda dur

if bulunan_kural_sayisi == 0:
    print("-> Basit kategorik kombinasyonlarda sıfır sapmalı bir kural bulunamadı.")

# 3. KÜSURAT HİLESİ (FRACTIONAL LEAKAGE)
print("\n3. HEDEF DEĞİŞKEN KÜSURAT ANALİZİ")
# Performans skorlarının virgülden sonraki kısımlarında bir şifre var mı?
train['kusurat'] = train['bilissel_performans_skoru'] % 1
kusurat_farkli = train['kusurat'].nunique()
print(f"Performans skorlarında {kusurat_farkli} farklı küsurat var.")
if kusurat_farkli < 100:
    print("🚨 TEHLİKE: Skorlar hesaplanmış bir formülden geliyor! (Örn: Sadece 0.25'in katları)")
else:
    print("-> Küsuratlar sürekli, basit bir yuvarlama hilesi yok.")
    
print("\n==================================================")