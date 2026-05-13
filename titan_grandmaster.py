import pandas as pd
import numpy as np
import time
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import RidgeCV, BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("🔥 TAKIM 83: TİTAN MİMARİSİ (30-FOLD DEEP STACKING) BAŞLATILIYOR 🔥")
print("UYARI: Bu işlem saatler sürebilir. Arkanıza yaslanın ve fan sesinin tadını çıkarın...\n")
baslangic_zamani = time.time()

# ---------------------------------------------------------
# 1. VERİ OKUMA VE ÖZELLİK MÜHENDİSLİĞİ (Feature Engineering)
# ---------------------------------------------------------
print("[1/5] Veriler Okunuyor ve Elit Sinyaller İşleniyor...")
train_df = pd.read_csv('train_temiz.csv')
test_df = pd.read_csv('test_temiz.csv')

hedef_kolon = 'bilissel_performans_skoru'
test_id = test_df['id'] if 'id' in test_df.columns else pd.read_csv('test_x.csv')['id']

if 'id' in train_df.columns: train_df = train_df.drop(columns=['id'])
if 'id' in test_df.columns: test_df = test_df.drop(columns=['id'])

# Sinerjik Özellikler
for df in [train_df, test_df]:
    df['sinerjik_zihinsel_yuk'] = df['stres_skoru'] * df['gunluk_calisma_saati']
    df['uyku_onarim_indeksi'] = (df['rem_yuzdesi'] + df['derin_uyku_yuzdesi']) / (df['gecelik_uyanma_sayisi'] + 1e-5)
    df['hareket_stres_orani'] = df['gunluk_adim_sayisi'] / (df['stres_skoru'] + 1)
    df['gizli_tukenmislik'] = df['stres_skoru'] ** 2 / (df['derin_uyku_yuzdesi'] + 1)

kategorik_kolonlar = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
sayisal_kolonlar = [col for col in train_df.columns if col not in kategorik_kolonlar and col != hedef_kolon]

# Hedef Kodlama (Target Encoding - Çapraz Doğrulama İçi Sızıntıyı Önlemek İçin Basit Versiyon)
print("[2/5] Hedef Kodlama (Target Encoding) Uygulanıyor...")
for col in kategorik_kolonlar:
    ortalama_map = train_df.groupby(col)[hedef_kolon].mean()
    genel_ort = train_df[hedef_kolon].mean()
    train_df[f'{col}_hedef_kod'] = train_df[col].map(ortalama_map)
    test_df[f'{col}_hedef_kod'] = test_df[col].map(ortalama_map).fillna(genel_ort)

sayisal_kolonlar_genis = [col for col in train_df.columns if col not in kategorik_kolonlar and col != hedef_kolon]

# Ağaçlar için Kategorik Format
for col in kategorik_kolonlar:
    train_df[col] = train_df[col].astype('category')
    test_df[col] = test_df[col].astype('category')

y = train_df[hedef_kolon]
X = train_df.drop(columns=[hedef_kolon])
X_test = test_df.copy()

# ---------------------------------------------------------
# 2. SİNİR AĞLARI VE KNN İÇİN MATEMATİKSEL UZAY (PCA & SCALING)
# ---------------------------------------------------------
print("[3/5] Matematiksel Uzay Hazırlığı (One-Hot, Scaler, PCA)...")
X_nn = pd.get_dummies(X, columns=kategorik_kolonlar, drop_first=True)
X_test_nn = pd.get_dummies(X_test, columns=kategorik_kolonlar, drop_first=True)
X_test_nn = X_test_nn.reindex(columns=X_nn.columns, fill_value=0)

scaler = StandardScaler()
X_nn_scaled = scaler.fit_transform(X_nn)
X_test_nn_scaled = scaler.transform(X_test_nn)

# PCA (Boyut İndirgeme) - Ana Sinyalleri Çıkarma
pca = PCA(n_components=10, random_state=42)
X_pca_train = pca.fit_transform(scaler.fit_transform(X[sayisal_kolonlar_genis]))
X_pca_test = pca.transform(scaler.transform(X_test[sayisal_kolonlar_genis]))

X_nn_final = np.hstack((X_nn_scaled, X_pca_train))
X_test_nn_final = np.hstack((X_test_nn_scaled, X_pca_test))

# ---------------------------------------------------------
# 3. SEVİYE-0: DEVASAL MODEL EĞİTİMİ (30 FOLD CROSS-VALIDATION)
# ---------------------------------------------------------
print(f"\n[4/5] 10 MODEL x 30 FOLD EĞİTİMİ BAŞLIYOR (Toplam 300 Model Eğitilecek!)")
# 10 Fold, 3 Tekrar = 30 Fold
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')

# İşçi Modellerin Tahminlerini Tutacak Matrisler
oof_train = np.zeros((len(X), 10))
oof_test = np.zeros((len(X_test), 10))

# Modellerin Listesi (Döngü içi tanımlanacak parametreler)
for fold, (train_idx, val_idx) in enumerate(rskf.split(X, y_binned)):
    fold_start = time.time()
    
    # Veri Bölme
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    X_tr_nn, X_va_nn = X_nn_final[train_idx], X_nn_final[val_idx]
    
    # 1. CatBoost (Derin)
    cb1 = CatBoostRegressor(iterations=1500, learning_rate=0.03, depth=6, cat_features=kategorik_kolonlar, random_seed=42, verbose=0)
    cb1.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=100)
    oof_train[val_idx, 0] += cb1.predict(X_va) / 3 # 3 tekrardan dolayı /3
    oof_test[:, 0] += cb1.predict(X_test) / 30     # 30 folddan dolayı /30
    
    # 2. CatBoost (Sığ - Varyans Düşürücü)
    cb2 = CatBoostRegressor(iterations=1500, learning_rate=0.04, depth=4, l2_leaf_reg=5, cat_features=kategorik_kolonlar, random_seed=777, verbose=0)
    cb2.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=100)
    oof_train[val_idx, 1] += cb2.predict(X_va) / 3
    oof_test[:, 1] += cb2.predict(X_test) / 30

    # 3. LightGBM (Standart RMSE)
    lgb1 = lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.02, max_depth=6, num_leaves=40, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=-1)
    lgb1.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_train[val_idx, 2] += lgb1.predict(X_va) / 3
    oof_test[:, 2] += lgb1.predict(X_test) / 30
    
    # 4. LightGBM (Huber Loss - Aykırı Değer Savaşçısı)
    lgb2 = lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.02, max_depth=5, objective='huber', alpha=1.2, random_state=777, verbose=-1, n_jobs=-1)
    lgb2.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_train[val_idx, 3] += lgb2.predict(X_va) / 3
    oof_test[:, 3] += lgb2.predict(X_test) / 30

    # 5. XGBoost (Standart)
    xgb1 = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.02, max_depth=5, subsample=0.8, colsample_bytree=0.8, enable_categorical=True, tree_method='hist', random_state=42, n_jobs=-1)
    xgb1.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_train[val_idx, 4] += xgb1.predict(X_va) / 3
    oof_test[:, 4] += xgb1.predict(X_test) / 30
    
    # 6. XGBoost (Pseudo-Huber)
    xgb2 = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.02, max_depth=4, objective='reg:pseudohubererror', enable_categorical=True, tree_method='hist', random_state=777, n_jobs=-1)
    xgb2.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_train[val_idx, 5] += xgb2.predict(X_va) / 3
    oof_test[:, 5] += xgb2.predict(X_test) / 30
    
    # 7. Yapay Sinir Ağı (MLP)
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', learning_rate_init=0.005, max_iter=250, early_stopping=True, random_state=42)
    mlp.fit(X_tr_nn, y_tr)
    oof_train[val_idx, 6] += mlp.predict(X_va_nn) / 3
    oof_test[:, 6] += mlp.predict(X_test_nn_final) / 30
    
    # 8. ExtraTrees (Aşırı Rastgele Ağaçlar)
    et = ExtraTreesRegressor(n_estimators=200, max_depth=12, min_samples_split=10, random_state=42, n_jobs=-1)
    et.fit(X_tr_nn, y_tr)
    oof_train[val_idx, 7] += et.predict(X_va_nn) / 3
    oof_test[:, 7] += et.predict(X_test_nn_final) / 30

    # 9. RandomForest
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_split=10, random_state=42, n_jobs=-1)
    rf.fit(X_tr_nn, y_tr)
    oof_train[val_idx, 8] += rf.predict(X_va_nn) / 3
    oof_test[:, 8] += rf.predict(X_test_nn_final) / 30
    
    # 10. K-Nearest Neighbors (Uzaysal Benzerlik)
    knn = KNeighborsRegressor(n_neighbors=30, weights='distance', n_jobs=-1)
    knn.fit(X_tr_nn, y_tr)
    oof_train[val_idx, 9] += knn.predict(X_va_nn) / 3
    oof_test[:, 9] += knn.predict(X_test_nn_final) / 30
    
    gecen_sure = (time.time() - fold_start) / 60
    print(f"Fold {fold+1}/30 Tamamlandı! ({gecen_sure:.1f} dakika)")

# ---------------------------------------------------------
# 4. SEVİYE-1: YÖNETİCİ MODELLER (META-LEARNERS)
# ---------------------------------------------------------
print("\n[5/5] Yönetici Modeller Eğitiliyor ve Nihai Karar Veriliyor...")

# Meta Model 1: Ridge Regresyon (Doğrusal Harmanlama)
meta_ridge = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0))
meta_ridge.fit(oof_train, y)
ridge_preds_train = meta_ridge.predict(oof_train)
ridge_preds_test = meta_ridge.predict(oof_test)

# Meta Model 2: Bayesian Ridge (Olasılıksal Harmanlama)
meta_bayes = BayesianRidge()
meta_bayes.fit(oof_train, y)
bayes_preds_train = meta_bayes.predict(oof_train)
bayes_preds_test = meta_bayes.predict(oof_test)

# ---------------------------------------------------------
# 5. SEVİYE-2: NİHAİ İTTİFAK (BLENDING OF META-LEARNERS)
# ---------------------------------------------------------
# Yöneticilerin kararlarını da eşit ağırlıkla birleştiriyoruz
final_train_preds = (ridge_preds_train * 0.6) + (bayes_preds_train * 0.4)
final_test_preds = (ridge_preds_test * 0.6) + (bayes_preds_test * 0.4)

# 0-10 Sınırı Tıraşlama
final_train_preds = np.clip(final_train_preds, 0.0, 10.0)
final_test_preds = np.clip(final_test_preds, 0.0, 10.0)

nihai_cv_rmse = root_mean_squared_error(y, final_train_preds)
toplam_sure = (time.time() - baslangic_zamani) / 60

print("\n=======================================================")
print(f"🏆 TAKIM 83 - TİTAN MİMARİSİ NİHAİ CV RMSE: {nihai_cv_rmse:.6f} 🏆")
print(f"⏱️ Toplam Eğitim Süresi: {toplam_sure:.1f} dakika")
print("=======================================================")

# Dosya Kaydetme
submission = pd.DataFrame({'id': test_id, hedef_kolon: final_test_preds})
dosya_adi = 'submission_TEAM83_TITAN_STACK.csv'
submission.to_csv(dosya_adi, index=False)
print(f"Efsanevi Liderlik Dosyanız Hazır: {dosya_adi}")