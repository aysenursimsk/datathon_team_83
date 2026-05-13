import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import RidgeCV
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("🦅 HEDEF 1.19: KESKİN NİŞANCI OPERASYONU BAŞLADI 🦅\n")

# 1. VERİLERİ OKUMA
train = pd.read_csv('train_ULTRA_temiz.csv')
test = pd.read_csv('test_ULTRA_temiz.csv')

hedef_kolon = 'bilissel_performans_skoru'

# Test setinde 'id' sütunu en sonda yer alıyor, onu ayırıyoruz
test_id = test['id'].astype(int)
test = test.drop(columns=['id'])

y = train[hedef_kolon]
X = train.drop(columns=[hedef_kolon])
X_test = test.copy()

# 2. EN BÜYÜK HATANIN ÇÖZÜMÜ: KATEGORİK ZEKANIN UYANDIRILMASI
# Bu sütunlar ondalıklı sayı değil, insan kategorileridir!
kategorikler = ['cinsiyet', 'meslek', 'ulke', 'kronotip', 'ruh_sagligi_durumu', 'mevsim', 'gun_tipi']

print("-> Kategorik veriler algoritmalara doğru şekilde tanıtılıyor...")
for col in kategorikler:
    # Önce float'tan int'e, sonra category tipine zorluyoruz
    X[col] = X[col].astype(int).astype('category')
    X_test[col] = X_test[col].astype(int).astype('category')

# 3. YAVAŞ VE DERİN EĞİTİM MİMARİSİ (10-FOLD CV)
print("-> 10-Fold Keskin Nişancı Eğitimi Başlıyor (Öğrenme hızı düşürüldü, ağaçlar artırıldı)...\n")
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')

oof_cat = np.zeros(len(X)); test_cat = np.zeros(len(X_test))
oof_lgb = np.zeros(len(X)); test_lgb = np.zeros(len(X_test))
oof_xgb = np.zeros(len(X)); test_xgb = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    # --- MODEL 1: CATBOOST (Kategorik Kralı) ---
    # Öğrenme hızı 0.015'e çekildi, L2 regülasyonu 5 yapıldı (ezberlemeyi önler)
    model_cat = CatBoostRegressor(
        iterations=2500, learning_rate=0.015, depth=6, l2_leaf_reg=5, 
        cat_features=kategorikler, random_seed=42, verbose=0
    )
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=150)
    oof_cat[val_idx] = model_cat.predict(X_va)
    test_cat += model_cat.predict(X_test) / skf.n_splits
    
    # --- MODEL 2: LIGHTGBM (Huber Kaybı ile Sapan Değer Zırhı) ---
    # Öğrenme hızı 0.01, ağaçlar 2500, num_leaves 31 (hassas ayar)
    model_lgb = lgb.LGBMRegressor(
        n_estimators=2500, learning_rate=0.01, max_depth=6, num_leaves=31,
        objective='huber', alpha=1.2, subsample=0.7, colsample_bytree=0.7,
        random_state=42, verbose=-1, n_jobs=-1
    )
    model_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_lgb[val_idx] = model_lgb.predict(X_va)
    test_lgb += model_lgb.predict(X_test) / skf.n_splits
    
    # --- MODEL 3: XGBOOST (Pseudo-Huber ile Stabilizasyon) ---
    model_xgb = xgb.XGBRegressor(
        n_estimators=2000, learning_rate=0.01, max_depth=5, 
        objective='reg:pseudohubererror', subsample=0.8, colsample_bytree=0.8,
        enable_categorical=True, tree_method='hist', random_state=42, n_jobs=-1
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_va)
    test_xgb += model_xgb.predict(X_test) / skf.n_splits
    
    print(f"Fold {fold+1}/10 Tamamlandı - Sinyaller Çekildi.")

# 4. YÖNETİCİ KARARI VE NİHAİ SKOR (RIDGE BLENDING)
print("\nMeta Yönetici Karar Veriyor...")
X_meta_train = np.column_stack((oof_cat, oof_lgb, oof_xgb))
X_meta_test = np.column_stack((test_cat, test_lgb, test_xgb))

meta_model = RidgeCV(alphas=(0.1, 1.0, 10.0))
meta_model.fit(X_meta_train, y)

# Puanlar biyolojik olarak 0-10 arasındadır, tıraşlıyoruz
nihai_train_tahminleri = np.clip(meta_model.predict(X_meta_train), 0.0, 10.0)
nihai_test_tahminleri = np.clip(meta_model.predict(X_meta_test), 0.0, 10.0)

cv_rmse = root_mean_squared_error(y, nihai_train_tahminleri)

print("\n=======================================================")
print(f"🎯 HEDEF 1.19 NİHAİ CV RMSE: {cv_rmse:.5f} 🎯")
print(f"Model Ağırlıkları -> CatBoost: {meta_model.coef_[0]:.2f} | LGBM: {meta_model.coef_[1]:.2f} | XGB: {meta_model.coef_[2]:.2f}")
print("=======================================================")

# LİDERLİK TABLOSU DOSYASI
submission = pd.DataFrame({'id': test_id, hedef_kolon: nihai_test_tahminleri})
dosya_adi = 'submission_HEDEF_1_19_FINAL.csv'
submission.to_csv(dosya_adi, index=False)
print(f"Kusursuz Liderlik Dosyanız Hazır: {dosya_adi}")