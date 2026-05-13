import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("👑 TAKIM 83 - 0.22 ZİRVE OPERASYONU: 'DEEP LOGIC' BAŞLATILDI 👑\n")

# 1. VERİLERİ OKU VE TEMİZLE
train = pd.read_csv('train.csv')
test = pd.read_csv('test_x.csv')
hedef = 'bilissel_performans_skoru'

cat_features = ['cinsiyet', 'meslek', 'ulke', 'kronotip', 'ruh_sagligi_durumu', 'mevsim', 'gun_tipi']
num_features = [col for col in train.columns if col not in cat_features + ['id', hedef]]

mapping = {'Spain': 'Ispanya', 'South Korea': 'Guney Kore', 'Sweden': 'Isvec', 'Lawyer': 'Avukat'}

for df in [train, test]:
    df.replace(mapping, inplace=True)
    # Sayısal hata koruması
    for col in num_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
    # Kategorik hata koruması
    for col in cat_features:
        df[col] = df[col].fillna('Bilinmiyor').astype(str)

test_id = test['id']
y = train[hedef]
X = train.drop(columns=['id', hedef])
X_test = test.drop(columns=['id'])

# 2. SİHİRLİ FORMÜLLER (Korelasyon Kırıcılar)
print("-> 0.22 Şifresi: Çoklu etkileşimler enjekte ediliyor...")
for df in [X, X_test]:
    df['stres_rem_ratio'] = df['stres_skoru'] / (df['rem_yuzdesi'] + 1e-6)
    df['stres_derin_ratio'] = df['stres_skoru'] / (df['derin_uyku_yuzdesi'] + 1e-6)
    df['stres_sq'] = df['stres_skoru'] ** 2
    df['uyku_verimi'] = (df['rem_yuzdesi'] + df['derin_uyku_yuzdesi']) / (df['stres_skoru'] + 1)
    # 3-Way Interaction (Bu çok kritik olabilir)
    df['magic_3way'] = (df['stres_skoru'] * df['gunluk_calisma_saati']) / (df['rem_yuzdesi'] + 1)

# 3. İKİ DEVİN SAVAŞI: CATBOOST + EXTRATREES
print("\n🚀 Model A (CatBoost - Derin Mantık) Eğitiliyor...")
model_cat = CatBoostRegressor(
    iterations=4000, learning_rate=0.01, depth=12, # Derinlik 12'ye çıktı!
    l2_leaf_reg=3, cat_features=cat_features, random_seed=42, verbose=500
)
model_cat.fit(X, y)

print("\n🚀 Model B (ExtraTrees - Saf Mantık) Eğitiliyor...")
# ExtraTrees kategorik veriyi sevmez, basitçe encode edelim
X_et = X.copy()
X_test_et = X_test.copy()
for col in cat_features:
    codes, _ = pd.factorize(pd.concat([X_et[col], X_test_et[col]]))
    X_et[col] = codes[:len(X_et)]
    X_test_et[col] = codes[len(X_et):]

model_et = ExtraTreesRegressor(n_estimators=500, max_depth=None, min_samples_split=2, random_state=42, n_jobs=-1)
model_et.fit(X_et, y)

# 4. YIĞINLAMA (STRESS-FREE BLENDING)
print("\n-> İki devin tahmini birleştiriliyor...")
preds_cat = model_cat.predict(X_test)
preds_et = model_et.predict(X_test_et)

# Ağırlıklı ortalama (0.22 alanların gizli silahı)
nihai_preds = (preds_cat * 0.7) + (preds_et * 0.3)

# 5. KAYIT
submission = pd.DataFrame({'id': test_id, hedef: np.clip(nihai_preds, 0.0, 10.0)})
dosya_adi = 'submission_ULTRA_ZIRVE_022.csv'
submission.to_csv(dosya_adi, index=False)

print("\n==================================================")
print(f"✅ İŞLEM TAMAM! Dosya: {dosya_adi}")
print("==================================================")