import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OrdinalEncoder
import warnings
warnings.filterwarnings('ignore')

print("🧬 SIFIRDAN VERİ İNŞASI VE HIZLI CERRAHİ TEMİZLİK BAŞLIYOR 🧬\n")

# 1. HAM VERİLERİ OKU
train = pd.read_csv('train.csv')
test = pd.read_csv('test_x.csv')

hedef_kolon = 'bilissel_performans_skoru'
test_id = test['id'].copy() if 'id' in test.columns else None

if 'id' in train.columns: train = train.drop(columns=['id'])
if 'id' in test.columns: test = test.drop(columns=['id'])

# Ülke ve Meslek isimlerindeki tutarsızlıkları düzelt (Veri standartlaştırma)
ulke_mapping = {'Spain': 'Ispanya', 'South Korea': 'Guney Kore', 'Sweden': 'Isvec'}
meslek_mapping = {'Lawyer': 'Avukat'}
for df in [train, test]:
    df['ulke'] = df['ulke'].replace(ulke_mapping)
    df['meslek'] = df['meslek'].replace(meslek_mapping)

y_train = train[hedef_kolon]
train = train.drop(columns=[hedef_kolon])

# 2. KATEGORİK VERİLERİ NUMARALANDIRMA (Ordinal Encoding)
# Target Encoding kullanmıyoruz (Sızıntı yapmamak için). Güvenli olan Ordinal kullanıyoruz.
kategorik_kolonlar = train.select_dtypes(include=['object']).columns.tolist()

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train[kategorik_kolonlar] = encoder.fit_transform(train[kategorik_kolonlar])
test[kategorik_kolonlar] = encoder.transform(test[kategorik_kolonlar])

# 3. MICE İLE HIZLI VE AKILLI EKSİK VERİ DOLDURMA (BayesianRidge)
print("-> Eksik veriler Makine Öğrenmesi (MICE - BayesianRidge) ile saniyeler içinde tahmin ediliyor...")
imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)

# Train ve Test'i birleştirip ortak uzayda dolduralım ki daha tutarlı olsun
tum_veri = pd.concat([train, test], axis=0)
tum_veri_dolu = pd.DataFrame(imputer.fit_transform(tum_veri), columns=tum_veri.columns)

train_dolu = tum_veri_dolu.iloc[:len(train)].reset_index(drop=True)
test_dolu = tum_veri_dolu.iloc[len(train):].reset_index(drop=True)
train_dolu[hedef_kolon] = y_train.values

# 4. YENİ NESİL BİYOLOJİK ÖZELLİK MÜHENDİSLİĞİ
print("-> Nörolojik ve Fizyolojik sinyaller üretiliyor...")
for df in [train_dolu, test_dolu]:
    # 1. Uyku Kalitesi Metrikleri
    df['toplam_kaliteli_uyku_yuzdesi'] = df['rem_yuzdesi'] + df['derin_uyku_yuzdesi']
    df['kalitesiz_uyku_yuzdesi'] = 100 - df['toplam_kaliteli_uyku_yuzdesi']
    df['uyku_verimsizligi'] = df['kalitesiz_uyku_yuzdesi'] * df['gecelik_uyanma_sayisi']
    
    # 2. Fiziksel ve Zihinsel Stres Çarpanları
    df['fiziksel_stres_endeksi'] = df['stres_skoru'] * df['dinlenik_nabiz_bpm']
    df['kronik_yorgunluk_riski'] = df['gunluk_calisma_saati'] * df['stres_skoru']
    
    # 3. Uyku Hijyeni Toksisitesi
    df['uyku_hijyeni_toksisitesi'] = np.log1p(df['uyku_oncesi_kafein_mg']) * df['uyku_oncesi_ekran_suresi_dk']
    
    # 4. Telafi ve Dinlenme Oranı
    df['net_dinlenme_orani'] = (df['sekerleme_suresi_dk'] + 1) / (df['gunluk_calisma_saati'] + 1)

# 5. İZOLASYON ORMANI İLE ZEHİRLİ SATIR (OUTLIER) SİLME
print("-> Eğitim setindeki modeli bozan zehirli aykırı değerler tespit ediliyor...")
# Sadece özelliklere bakarak anomali tespiti yapıyoruz (hedef kolonu dahil etmeden)
X_train_for_iso = train_dolu.drop(columns=[hedef_kolon])
iso_forest = IsolationForest(contamination=0.03, random_state=42) # Verinin en tuhaf %3'ünü bul
outlier_labels = iso_forest.fit_predict(X_train_for_iso)

# -1 olanlar outlier (aykırı), 1 olanlar inlier (normal)
train_ultra_temiz = train_dolu[outlier_labels == 1].reset_index(drop=True)
silinen_satir_sayisi = len(train_dolu) - len(train_ultra_temiz)

test_dolu['id'] = test_id.values

print(f"🔥 İŞLEM TAMAM: Modeli kör eden tam {silinen_satir_sayisi} zehirli satır eğitim setinden silindi!")

# 6. YENİ ALTIN VERİ SETLERİNİ KAYDET
train_ultra_temiz.to_csv('train_ULTRA_temiz.csv', index=False)
test_dolu.to_csv('test_ULTRA_temiz.csv', index=False)

print("\n🚀 ŞAMPİYONLUK VERİ SETLERİNİZ HAZIR: 'train_ULTRA_temiz.csv' ve 'test_ULTRA_temiz.csv'")