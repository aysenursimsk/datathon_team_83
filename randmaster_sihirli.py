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

# ---------------------------------------------------------
# 1. TAM KAPSAMLI VERİ HAZIRLIĞI (SİHİRLİ SİNYALLER EKLENDİ)
# ---------------------------------------------------------
def tam_kapsamli_veri_hazirligi():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test_x.csv')
    test_id = test_df['id'].copy() if 'id' in test_df.columns else None
    
    if 'id' in train_df.columns: train_df = train_df.drop(columns=['id'])
    if 'id' in test_df.columns: test_df = test_df.drop(columns=['id'])

    ulke_mapping = {'Spain': 'Ispanya', 'South Korea': 'Guney Kore', 'Sweden': 'Isvec'}
    train_df['ulke'] = train_df['ulke'].replace(ulke_mapping)
    test_df['ulke'] = test_df['ulke'].replace(ulke_mapping)
    train_df['meslek'] = train_df['meslek'].replace({'Lawyer': 'Avukat'})
    test_df['meslek'] = test_df['meslek'].replace({'Lawyer': 'Avukat'})

    meslek_stres_map = train_df.groupby('meslek')['stres_skoru'].mean().to_dict()
    meslek_mesai_map = train_df.groupby('meslek')['gunluk_calisma_saati'].mean().to_dict()
    
    train_df['gecici_uyku_kalitesi'] = (train_df['rem_yuzdesi'] + train_df['derin_uyku_yuzdesi']) / (train_df['gecelik_uyanma_sayisi'] + 1)
    ulke_uyku_map = train_df.groupby('ulke')['gecici_uyku_kalitesi'].mean().to_dict()
    train_df = train_df.drop(columns=['gecici_uyku_kalitesi'])

    for df in [train_df, test_df]:
        df['meslek_stres_ortalamasi'] = df['meslek'].map(meslek_stres_map)
        df['meslek_mesai_ortalamasi'] = df['meslek'].map(meslek_mesai_map)
        df['ulke_uyku_kalitesi_ortalamasi'] = df['ulke'].map(ulke_uyku_map)
        df['meslektastan_stres_sapmasi'] = df['stres_skoru'] - df['meslek_stres_ortalamasi']

    for df in [train_df, test_df]:
        kategorik_kolonlar = df.select_dtypes(include=['object', 'category']).columns
        for col in kategorik_kolonlar: df[col] = df[col].fillna('Bilinmiyor')
                
        sayisal_kolonlar = df.select_dtypes(exclude=['object', 'category']).columns
        for col in sayisal_kolonlar:
            if col in df.columns and col != 'bilissel_performans_skoru':
                df[col] = df[col].fillna(df[col].median())

        if 'uyku_oncesi_kafein_mg' in df.columns: df['uyku_oncesi_kafein_mg'] = np.log1p(df['uyku_oncesi_kafein_mg'])
        if 'uyku_oncesi_ekran_suresi_dk' in df.columns:
            df['uyku_oncesi_ekran_suresi_dk'] = df['uyku_oncesi_ekran_suresi_dk'].clip(upper=df['uyku_oncesi_ekran_suresi_dk'].quantile(0.99))

        df['sinerjik_zihinsel_yuk'] = df['stres_skoru'] * df['gunluk_calisma_saati']
        df['uyku_onarim_indeksi'] = (df['rem_yuzdesi'] + df['derin_uyku_yuzdesi']) / (df['gecelik_uyanma_sayisi'] + 1e-5)
        hafta_sonu_kolonu = 'hafta_sonu_uyku_farki' if 'hafta_sonu_uyku_farki' in df.columns else 'hafta_sonu_uyku_farki_saat'
        df['sosyal_jet_lag'] = (df[hafta_sonu_kolonu] > 2.0).astype(int)
        
        kafein_esik = df['uyku_oncesi_kafein_mg'].median()
        ekran_esik = df['uyku_oncesi_ekran_suresi_dk'].median()
        df['kotu_uyku_hijyeni'] = ((df['kronotip'] == 'Gece') & (df['uyku_oncesi_kafein_mg'] > kafein_esik) & (df['uyku_oncesi_ekran_suresi_dk'] > ekran_esik)).astype(int)
        df['hareket_stres_orani'] = df['gunluk_adim_sayisi'] / (df['stres_skoru'] + 1)

        # -----------------------------------------------------------------
        # YENİ SİHİRLİ SİNYALLER (Hata Analizi Sonucu)
        # -----------------------------------------------------------------
        # 1. Vardiya ve Sabahlama Riski
        riskli_meslekler = ['Saglik Personeli', 'Ogrenci', 'Lojistik Calisani']
        df['vardiya_sabahlama_riski'] = ((df['meslek'].isin(riskli_meslekler)) & (df['gun_tipi'] == 'Hafta ici')).astype(int)
        
        # 2. Gizli Tükenmişlik (Riskli grubun stresi normalden 2 kat yıkıcıdır)
        df['gizli_tukenmislik'] = df['vardiya_sabahlama_riski'] * df['stres_skoru']
        
        # 3. Telafi Edilemeyen Uyku (Riskli grupta derin uyku eksikse tam çöküş)
        df['telafi_edilemeyen_uyku'] = df['vardiya_sabahlama_riski'] / (df['derin_uyku_yuzdesi'] + 1)

    X = train_df.drop(columns=['bilissel_performans_skoru'])
    y = train_df['bilissel_performans_skoru']
    X_test = test_df.copy()

    kategorik_kolonlar = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in kategorik_kolonlar:
        X[col] = X[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    return X, y, X_test, test_id, kategorik_kolonlar

print("Sihirli Sinyaller Veriye İşleniyor...")
X, y, X_test, test_id, kategorik_kolonlar = tam_kapsamli_veri_hazirligi()
y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')

# ---------------------------------------------------------
# 2. GRANDMASTER MİMARİSİ (Sihirli Veri ile)
# ---------------------------------------------------------
seed_list = [42, 2026, 777, 1024, 888]
final_test_predictions = np.zeros(len(X_test))
seed_skorlari = []

print(f"\n🚀 SİHİRLİ GRANDMASTER BAŞLIYOR (75 Model Eğitiliyor)...")

for i, seed in enumerate(seed_list):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    oof_cat = np.zeros(len(X)); test_cat = np.zeros(len(X_test))
    oof_lgb = np.zeros(len(X)); test_lgb = np.zeros(len(X_test))
    oof_xgb = np.zeros(len(X)); test_xgb = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model_cat = CatBoostRegressor(iterations=2594, learning_rate=0.0195, depth=5, l2_leaf_reg=4, cat_features=kategorik_kolonlar, random_seed=seed, verbose=0)
        model_cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=150)
        oof_cat[val_idx] = model_cat.predict(X_val)
        test_cat += model_cat.predict(X_test) / skf.n_splits
        
        model_lgb = lgb.LGBMRegressor(n_estimators=1332, learning_rate=0.01078, max_depth=5, num_leaves=36, subsample=0.6176, colsample_bytree=0.7123, random_state=seed, n_jobs=-1, verbose=-1)
        model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_lgb[val_idx] = model_lgb.predict(X_val)
        test_lgb += model_lgb.predict(X_test) / skf.n_splits
        
        model_xgb = xgb.XGBRegressor(n_estimators=1156, learning_rate=0.0129, max_depth=4, subsample=0.6543, colsample_bytree=0.8145, enable_categorical=True, tree_method='hist', random_state=seed, n_jobs=-1)
        model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx] = model_xgb.predict(X_val)
        test_xgb += model_xgb.predict(X_test) / skf.n_splits

    X_meta_train = np.column_stack((oof_cat, oof_lgb, oof_xgb))
    X_meta_test = np.column_stack((test_cat, test_lgb, test_xgb))

    meta_model = RidgeCV(alphas=(0.1, 1.0, 10.0))
    meta_model.fit(X_meta_train, y)
    
    nihai_train_tahminleri = meta_model.predict(X_meta_train)
    nihai_test_tahminleri = meta_model.predict(X_meta_test)
    
    seed_rmse = root_mean_squared_error(y, nihai_train_tahminleri)
    print(f"[{i+1}/{len(seed_list)}] Tohum: {seed} | Ridge CV RMSE: {seed_rmse:.5f}")
    
    seed_skorlari.append(seed_rmse)
    final_test_predictions += nihai_test_tahminleri / len(seed_list)

print(f"\n=======================================================")
print(f"🌟 YENİ REKOR CV RMSE: {np.mean(seed_skorlari):.5f} 🌟")
print(f"=======================================================")

submission_stacking = pd.DataFrame({'id': test_id, 'bilissel_performans_skoru': final_test_predictions})
dosya_adi = 'submission_MAGIC_Grandmaster.csv'
submission_stacking.to_csv(dosya_adi, index=False)
print(f"Sihirli Liderlik Dosyası Hazır: {dosya_adi}")