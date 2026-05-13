import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import RidgeCV
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor
import warnings
warnings.filterwarnings('ignore')

print("🚀 ULTRA-TEMİZ VERİ İLE GRANDMASTER EĞİTİMİ BAŞLIYOR 🚀\n")

# 1. Verileri Okuma
train = pd.read_csv('train_ULTRA_temiz.csv')
test = pd.read_csv('test_ULTRA_temiz.csv')

hedef_kolon = 'bilissel_performans_skoru'
test_id = test['id'].astype(int)

y = train[hedef_kolon]
X = train.drop(columns=[hedef_kolon])
X_test = test.drop(columns=['id'])

# (Tüm veriler önceki aşamada OrdinalEncoder ile sayısallaştırıldığı için 
# ekstra bir dönüşüme gerek yok. Algoritmalar doğrudan sayıları okuyacak.)

# 2. 5-Fold Stacking Mimarisi (Güvenilir, Hızlı ve Sızıntısız)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')

oof_cat = np.zeros(len(X)); test_cat = np.zeros(len(X_test))
oof_lgb = np.zeros(len(X)); test_lgb = np.zeros(len(X_test))
oof_xgb = np.zeros(len(X)); test_xgb = np.zeros(len(X_test))
oof_et = np.zeros(len(X));  test_et = np.zeros(len(X_test))

print("Modeller Eğitiliyor (CatBoost, LightGBM, XGBoost, ExtraTrees)...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    # Model 1: CatBoost (Ağaçların Lideri)
    model_cat = CatBoostRegressor(iterations=1500, learning_rate=0.03, depth=6, random_seed=42, verbose=0)
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=100)
    oof_cat[val_idx] = model_cat.predict(X_va)
    test_cat += model_cat.predict(X_test) / skf.n_splits
    
    # Model 2: LightGBM (Hızlı ve Güçlü)
    model_lgb = lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.02, max_depth=5, random_state=42, verbose=-1, n_jobs=-1)
    model_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb[val_idx] = model_lgb.predict(X_va)
    test_lgb += model_lgb.predict(X_test) / skf.n_splits
    
    # Model 3: XGBoost (Geleneksel Güç)
    model_xgb = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.02, max_depth=4, random_state=42, n_jobs=-1)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_va)
    test_xgb += model_xgb.predict(X_test) / skf.n_splits
    
    # Model 4: ExtraTrees (Kaos ve Varyans Düşürücü)
    model_et = ExtraTreesRegressor(n_estimators=300, max_depth=12, min_samples_split=5, random_state=42, n_jobs=-1)
    model_et.fit(X_tr, y_tr)
    oof_et[val_idx] = model_et.predict(X_va)
    test_et += model_et.predict(X_test) / skf.n_splits
    
    print(f"Fold {fold+1}/5 Tamamlandı.")

# 3. Seviye-1 Yönetici Model (RidgeCV)
print("\nYönetici Model Karar Veriyor...")
X_meta_train = np.column_stack((oof_cat, oof_lgb, oof_xgb, oof_et))
X_meta_test = np.column_stack((test_cat, test_lgb, test_xgb, test_et))

meta_model = RidgeCV(alphas=(0.1, 1.0, 10.0))
meta_model.fit(X_meta_train, y)

nihai_train_tahminleri = np.clip(meta_model.predict(X_meta_train), 0.0, 10.0)
nihai_test_tahminleri = np.clip(meta_model.predict(X_meta_test), 0.0, 10.0)

cv_rmse = root_mean_squared_error(y, nihai_train_tahminleri)

print("\n=======================================================")
print(f"🏆 ULTRA TEMİZ VERİ - NİHAİ CV RMSE: {cv_rmse:.5f} 🏆")
print(f"Ağırlıklar: CatBoost: {meta_model.coef_[0]:.2f} | LGBM: {meta_model.coef_[1]:.2f} | XGB: {meta_model.coef_[2]:.2f} | ET: {meta_model.coef_[3]:.2f}")
print("=======================================================")

# Dosya Kaydetme
submission = pd.DataFrame({'id': test_id, hedef_kolon: nihai_test_tahminleri})
dosya_adi = 'submission_TEAM83_ULTRA_CLEAN.csv'
submission.to_csv(dosya_adi, index=False)
print(f"Yeni Liderlik Dosyanız Hazır: {dosya_adi}")