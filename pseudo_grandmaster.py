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

print("🧪 1.19 OPERASYONU: PSEUDO-LABELING (SAHTE ETİKETLEME) BAŞLADI 🧪\n")

# 1. ORİJİNAL VE EN GÜÇLÜ VERİYE DÖNÜŞ (Aykırı değerlerin SİLİNMEDİĞİ saf veri)
train = pd.read_csv('train_temiz.csv')
test = pd.read_csv('test_temiz.csv')

hedef_kolon = 'bilissel_performans_skoru'

# Datathon platformu için id'yi garantileme
if 'id' in test.columns:
    test_id = test['id']
    test = test.drop(columns=['id'])
else:
    test_id = pd.read_csv('test_x.csv')['id']

if 'id' in train.columns: 
    train = train.drop(columns=['id'])

y = train[hedef_kolon]
X = train.drop(columns=[hedef_kolon])
X_test = test.copy()

kategorikler = ['cinsiyet', 'meslek', 'ulke', 'kronotip', 'ruh_sagligi_durumu', 'mevsim', 'gun_tipi']

# =======================================================
# AŞAMA 1: TEST SETİ İÇİN 'ÖĞRETMEN' MODELİN EĞİTİLMESİ
# =======================================================
print("Aşama 1: Öğretmen Model Eğitiliyor ve Test Seti Tahmin Ediliyor...")
ogretmen_model = CatBoostRegressor(
    iterations=2000, learning_rate=0.02, depth=6, l2_leaf_reg=4,
    cat_features=kategorikler, random_seed=42, verbose=0
)
ogretmen_model.fit(X, y)

# Test setinin "Sahte" cevaplarını oluşturuyoruz
pseudo_y = ogretmen_model.predict(X_test)
pseudo_y = np.clip(pseudo_y, 0.0, 10.0) # Sınırları koru

# =======================================================
# AŞAMA 2: VERİLERİN BİRLEŞTİRİLMESİ (BÜYÜ BURADA)
# =======================================================
print("Aşama 2: Test Seti, Eğitim Setine 'Sahte Etiketlerle' Ekleniyor...")
X_test_pseudo = X_test.copy()
X_test_pseudo[hedef_kolon] = pseudo_y

# Orijinal X'e hedefi ekleyip birleştirelim ki veri tipleri kaybolmasın
train_orijinal = X.copy()
train_orijinal[hedef_kolon] = y

# Orijinal Train ile Pseudo Test'i alt alta birleştir! (80.000 Satırlık Devasa Veri)
train_genisletilmis = pd.concat([train_orijinal, X_test_pseudo], axis=0).reset_index(drop=True)

y_genis = train_genisletilmis[hedef_kolon]
X_genis = train_genisletilmis.drop(columns=[hedef_kolon])

# İŞTE KRİTİK DÜZELTME: BİRLEŞTİRMEDEN SONRA LİGHTGBM İÇİN CATEGORY'YE ZORLAMA
for col in kategorikler:
    X_genis[col] = X_genis[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# =======================================================
# AŞAMA 3: ÖĞRENCİ MODELLERİN EĞİTİMİ (YENİ UZAYDA)
# =======================================================
print("Aşama 3: Öğrenci Modeller 80.000 Satırlık Yeni Veride Eğitiliyor (5-Fold)...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
y_binned = pd.qcut(y_genis, q=10, labels=False, duplicates='drop')

oof_cat = np.zeros(len(X_genis)); final_test_cat = np.zeros(len(X_test))
oof_lgb = np.zeros(len(X_genis)); final_test_lgb = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_genis, y_binned)):
    X_tr, y_tr = X_genis.iloc[train_idx], y_genis.iloc[train_idx]
    X_va, y_va = X_genis.iloc[val_idx], y_genis.iloc[val_idx]
    
    # Yeni Öğrenci CatBoost
    cat = CatBoostRegressor(iterations=2500, learning_rate=0.015, depth=6, cat_features=kategorikler, random_seed=777, verbose=0)
    cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=100)
    oof_cat[val_idx] = cat.predict(X_va)
    final_test_cat += cat.predict(X_test) / skf.n_splits
    
    # Yeni Öğrenci LightGBM
    lgb_m = lgb.LGBMRegressor(n_estimators=2500, learning_rate=0.01, max_depth=6, objective='huber', random_state=777, verbose=-1, n_jobs=-1)
    lgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_lgb[val_idx] = lgb_m.predict(X_va)
    final_test_lgb += lgb_m.predict(X_test) / skf.n_splits
    
    print(f"Fold {fold+1}/5 Tamamlandı.")

# =======================================================
# AŞAMA 4: NİHAİ KARAR VE DOSYA OLUŞTURMA
# =======================================================
print("\nYönetici Model Karar Veriyor...")
# Sadece orijinal eğitim setinin uzunluğu kadar olan kısmı ölçüyoruz ki gerçek hatamızı bilelim (Testi değil)
orijinal_uzunluk = len(y)
y_gercek = y_genis.iloc[:orijinal_uzunluk]

X_meta_train = np.column_stack((oof_cat[:orijinal_uzunluk], oof_lgb[:orijinal_uzunluk]))
X_meta_test = np.column_stack((final_test_cat, final_test_lgb))

meta = RidgeCV(alphas=(0.1, 1.0, 10.0))
meta.fit(X_meta_train, y_gercek)

nihai_cv_tahmin = np.clip(meta.predict(X_meta_train), 0.0, 10.0)
nihai_test_tahmin = np.clip(meta.predict(X_meta_test), 0.0, 10.0)

cv_rmse = root_mean_squared_error(y_gercek, nihai_cv_tahmin)

print("\n=======================================================")
print(f"🔥 PSEUDO-LABELING NİHAİ CV RMSE: {cv_rmse:.5f} 🔥")
print("=======================================================")

dosya_adi = 'submission_HEDEF_1_19_PSEUDO_LABEL.csv'
submission = pd.DataFrame({'id': test_id, hedef_kolon: nihai_test_tahmin})
submission.to_csv(dosya_adi, index=False)
print(f"1.19 Barajını Kıracak Altın Dosyanız Hazır: {dosya_adi}")