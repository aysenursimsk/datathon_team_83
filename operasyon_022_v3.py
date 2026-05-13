import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("👑 TAKIM 83: 0.50 ALTI NİHAİ ZİRVE OPERASYONU BAŞLATILDI 👑\n")

# 1. VERİLERİ OKU VE CERRAHİ TEMİZLİK
train = pd.read_csv('train.csv')
test = pd.read_csv('test_x.csv')
hedef = 'bilissel_performans_skoru'

# KRİTİK: ID'leri silmeden önce kopyalıyoruz (Hata buradaydı, giderildi)
test_id = test['id'].copy() 

cat_features = ['cinsiyet', 'meslek', 'ulke', 'kronotip', 'ruh_sagligi_durumu', 'mevsim', 'gun_tipi']
num_features = [col for col in train.columns if col not in cat_features + ['id', hedef]]

mapping = {'Spain': 'Ispanya', 'South Korea': 'Guney Kore', 'Sweden': 'Isvec', 'Lawyer': 'Avukat'}

print("-> Veriler standartlaştırılıyor ve eksikler tamamlanıyor...")
for df in [train, test]:
    df.replace(mapping, inplace=True)
    # Sayısal sütun güvenliği
    for col in num_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
    # Kategorik sütun güvenliği
    for col in cat_features:
        df[col] = df[col].fillna('Bilinmiyor').astype(str)

y = train[hedef]
X = train.drop(columns=['id', hedef])
X_test = test.drop(columns=['id'])

# 2. BRUTE-FORCE FORMÜL ENJEKSİYONU (0.50'nin Anahtarı)
print("-> Saf matematiksel kombinasyonlar enjekte ediliyor...")
for df in [X, X_test]:
    # 0.81'e inmemizi sağlayan temel oranlar
    df['magic_1'] = df['stres_skoru'] / (df['rem_yuzdesi'] + 1e-6)
    df['magic_2'] = df['stres_skoru'] / (df['derin_uyku_yuzdesi'] + 1e-6)
    
    # 3'lü kombinasyonlar (Zirvedekilerin gizli formül simülasyonu)
    df['magic_3'] = (df['stres_skoru'] * df['gunluk_calisma_saati']) / (df['rem_yuzdesi'] + 1)
    df['magic_4'] = (df['stres_skoru'] * df['dinlenik_nabiz_bpm']) / (df['derin_uyku_yuzdesi'] + 1)
    
    # Matematiksel esneklik
    df['stres_sq'] = df['stres_skoru'] ** 2
    df['stres_log'] = np.log1p(df['stres_skoru'])

# 3. MEGA-STACKING (DÖRT DEVİN BİRLEŞİMİ)
print("\n🚀 Mega-Stacking Başlıyor (Bu işlem işlemci hızına göre 15-20 dk sürebilir)...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Modellerin tahminlerini tutacak meta-matrisler
oof_preds = np.zeros((len(X), 4))
test_preds = np.zeros((len(X_test), 4))

# Diğer modeller (XGB, LGBM, ET) için sayısal encoding
X_num = X.copy()
X_test_num = X_test.copy()
for col in cat_features:
    codes, _ = pd.factorize(pd.concat([X_num[col], X_test_num[col]]))
    X_num[col] = codes[:len(X_num)]
    X_test_num[col] = codes[len(X_num):]

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    X_tr_n, X_va_n = X_num.iloc[tr_idx], X_num.iloc[val_idx]
    
    print(f"Fold {fold+1}/5 eğitiliyor...")

    # Model 1: CatBoost (Zirve Derinlik)
    cb = CatBoostRegressor(iterations=2500, learning_rate=0.02, depth=10, cat_features=cat_features, verbose=0)
    cb.fit(X_tr, y_tr)
    oof_preds[val_idx, 0] = cb.predict(X_va)
    test_preds[:, 0] += cb.predict(X_test) / 5
    
    # Model 2: XGBoost
    xgb_m = xgb.XGBRegressor(n_estimators=1500, max_depth=8, learning_rate=0.02, tree_method='hist', random_state=42)
    xgb_m.fit(X_tr_n, y_tr)
    oof_preds[val_idx, 1] = xgb_m.predict(X_va_n)
    test_preds[:, 1] += xgb_m.predict(X_test_num) / 5
    
    # Model 3: LightGBM
    lgb_m = lgb.LGBMRegressor(n_estimators=1500, max_depth=8, learning_rate=0.02, verbose=-1, random_state=42)
    lgb_m.fit(X_tr_n, y_tr)
    oof_preds[val_idx, 2] = lgb_m.predict(X_va_n)
    test_preds[:, 2] += lgb_m.predict(X_test_num) / 5
    
    # Model 4: ExtraTrees (Mantık Kırıcı)
    et = ExtraTreesRegressor(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42)
    et.fit(X_tr_n, y_tr)
    oof_preds[val_idx, 3] = et.predict(X_va_n)
    test_preds[:, 3] += et.predict(X_test_num) / 5

# 4. YÖNETİCİ MODEL (META-LEARNER)
print("\n-> Yönetici Model (Ridge) nihai kararı veriyor...")
meta_model = RidgeCV(alphas=[0.1, 1.0, 10.0])
meta_model.fit(oof_preds, y)

# Puanların 0-10 aralığında kalmasını sağla
nihai_preds = np.clip(meta_model.predict(test_preds), 0.0, 10.0)
final_cv_rmse = root_mean_squared_error(y, meta_model.predict(oof_preds))

print("\n==================================================")
print(f"🔥 OPERASYON TAMAMLANDI - NİHAİ CV RMSE: {final_cv_rmse:.5f} 🔥")
print(f"Model Ağırlıkları -> CB: {meta_model.coef_[0]:.2f}, XGB: {meta_model.coef_[1]:.2f}, LGBM: {meta_model.coef_[2]:.2f}, ET: {meta_model.coef_[3]:.2f}")
print("==================================================")

# 5. DOSYAYI KAYDET
submission = pd.DataFrame({'id': test_id, hedef: nihai_preds})
dosya_adi = 'submission_GOD_MODE_050.csv'
submission.to_csv(dosya_adi, index=False)

print(f"\n🚀 Zirveye Giden Biletin Hazır: {dosya_adi}")
print("Bu dosyayı Liderlik Tablosuna yükle ve Takım 83'ün yükselişini izle!")