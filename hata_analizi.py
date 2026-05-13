import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("🔍 MODELİN EN ÇOK HATA YAPTIĞI KİŞİLER ARANIYOR...\n")

# Sadece temel veriyi okuyoruz
train = pd.read_csv('train.csv')
if 'id' in train.columns: train = train.drop(columns=['id'])

# Kategorik değişkenleri ayarla
kategorik = train.select_dtypes(include=['object']).columns.tolist()
for col in kategorik: train[col] = train[col].fillna('Bilinmiyor')
for col in train.select_dtypes(exclude=['object']).columns:
    if col != 'bilissel_performans_skoru': train[col] = train[col].fillna(train[col].median())

X = train.drop(columns=['bilissel_performans_skoru'])
y = train['bilissel_performans_skoru']

# Hızlı bir model eğit
model = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=5, cat_features=kategorik, verbose=0, random_seed=42)
model.fit(X, y)

# Kendi eğittiği veriyi tahmin et
tahminler = model.predict(X)

# Hataları hesapla (Gerçek Skor - Tahmin)
train['tahmin'] = tahminler
train['hata_miktari'] = np.abs(train['bilissel_performans_skoru'] - train['tahmin'])

# EN ÇOK HATA YAPILAN İLK 500 KİŞİYİ GETİR
en_kotuler = train.sort_values(by='hata_miktari', ascending=False).head(500)

print("--- EN ÇOK YANILDIĞIMIZ 500 KİŞİNİN PROFİLİ ---")
print("\n1. En Çok Hata Yapılan Meslekler:")
print(en_kotuler['meslek'].value_counts().head(5))

print("\n2. En Çok Hata Yapılan Gün Tipleri:")
print(en_kotuler['gun_tipi'].value_counts().head(3))

print("\n3. En Çok Hata Yapılan Yaş Ortalaması vs Genel Yaş Ortalaması:")
print(f"Genel Yaş Ortalaması: {train['yas'].mean():.1f}")
print(f"Hatalı Grubun Yaş Ortalaması: {en_kotuler['yas'].mean():.1f}")

print("\n4. Ortalama Hata Miktarı (RMSE formatında değil, direkt sapma):")
print(f"{en_kotuler['hata_miktari'].mean():.2f} puanlık devasa sapmalar var!")