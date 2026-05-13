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
# 1. ELİT VERİ HAZIRLIĞI (GÜRÜLTÜDEN ARINDIRILMIŞ)
# ---------------------------------------------------------
def elit_veri_hazirligi(df):
    df = df.copy()
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    ulke_mapping = {'Spain': 'Ispanya', 'South Korea': 'Guney Kore', 'Sweden': 'Isvec'}
    if 'ulke' in df.columns:
        df['ulke'] = df['ulke'].replace(ulke_mapping)
        
    if 'meslek' in df.columns:
        df['meslek'] = df['meslek'].replace({'Lawyer': 'Avukat'})

    kategorik_kolonlar = df.select_dtypes(include=['object', 'category']).columns
    for col in kategorik_kolonlar:
        df[col] = df[col].fillna('Bilinmiyor')
            
    sayisal_kolonlar = df.select_dtypes(exclude=['object', 'category']).columns
    for col in sayisal_kolonlar:
        if col in df.columns and col != 'bilissel_performans_skoru':
            df[col] = df[col].fillna(df[col].median())

    if 'uyku_oncesi_kafein_mg' in df.columns:
        df['uyku_oncesi_kafein_mg'] = np.log1p(df['uyku_oncesi_kafein_mg'])
        
    if 'uyku_oncesi_ekran_suresi_dk' in df.columns:
        p99 = df['uyku_oncesi_ekran_suresi_dk'].quantile(0.99)
        df['uyku_oncesi_ekran_suresi_dk'] = df['uyku_oncesi_ekran_suresi_dk'].clip(upper=p99)

    # --- KANITLANMIŞ ELİT ÖZELLİKLER ---
    if 'stres_skoru' in df.columns and 'gunluk_calisma_saati' in df.columns:
        df['sinerjik_zihinsel_yuk'] = df['stres_skoru'] * df['gunluk_calisma_saati']
        
    if 'rem_yuzdesi' in df.columns and 'derin_uyku_yuzdesi' in df.columns and 'gecelik_uyanma_sayisi' in df.columns:
        df['uyku_onarim_indeksi'] = (df['rem_yuzdesi'] + df['derin_uyku_yuzdesi']) / (df['gecelik_uyanma_sayisi'] + 1e-5)
        
    hafta_sonu_kolonu = 'hafta_sonu_uyku_farki' if 'hafta_sonu_uyku_farki' in df.columns else 'hafta_sonu_uyku_farki_saat'
    if hafta_sonu_kolonu in df.columns:
        df['sosyal_jet_lag'] = (df[hafta_sonu_kolonu] > 2.0).astype(int)

    if all(c in df.columns for c in ['kronotip', 'uyku_oncesi_kafein_mg', 'uyku_oncesi_ekran_suresi_dk']):
        kafein_esik = df['uyku_oncesi_kafein_mg'].median()
        ekran_esik = df['uyku_oncesi_ekran_suresi_dk'].median()
        df['kotu_uyku_hijyeni'] = ((df['kronotip'] == 'Gece') & 
                                   (df['uyku_oncesi_kafein_mg'] > kafein_esik) & 
                                   (df['uyku_oncesi_ekran_suresi_dk'] > ekran_esik)).astype(int)

    # Yeni Yıldızımız: Hareket ve Stres Oranı
    if 'gunluk_adim_sayisi' in df.columns and 'stres_skoru' in df.columns:
        df['hareket_stres_orani'] = df['gunluk_adim_sayisi'] / (df['stres_skoru'] + 1)

    return df

print("Veriler Okunuyor ve Elit Sinyaller Filtreleniyor...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test_x.csv')

test_id = test_df['id'].copy() if 'id' in test_df.columns else None

islenmis_train = elit_veri_hazirligi(train_df)
islenmis_test = elit_veri_hazirligi(test_df)

hedef_kolon = 'bilissel_performans_skoru'
X = islenmis_train.drop(columns=[hedef_kolon])
y = islenmis_train[hedef_kolon]
X_test = islenmis_test.copy()

# ---------------------------------------------------------
# 2. GLOBAL ENCODING (Algoritmalar arası uyum için)
# ---------------------------------------------------------
# LightGBM ve XGBoost metinleri algılayabilsin diye 'category' tipine çeviriyoruz
kategorik_kolonlar = X.select_dtypes(include=['object', 'category']).columns.tolist()
for col in kategorik_kolonlar:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# ---------------------------------------------------------
# 3. STACKING MİMARİSİ (Out-Of-Fold Eğitimi)
# ---------------------------------------------------------
print("🚀 Seviye-0 İşçi Modeller Eğitiliyor (CatBoost, LightGBM, XGBoost)...")
y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_cat = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))

test_cat = np.zeros(len(X_test))
test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
    print(f"\n--- STACKING FOLD {fold + 1} ---")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # 1. İşçi: CatBoost (Optuna Altın Parametreleri)
    model_cat = CatBoostRegressor(
        iterations=2594, learning_rate=0.0195, depth=5, l2_leaf_reg=4,
        cat_features=kategorik_kolonlar, eval_metric='RMSE', random_seed=42, verbose=0
    )
    model_cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=150)
    oof_cat[val_idx] = model_cat.predict(X_val)
    test_cat += model_cat.predict(X_test) / skf.n_splits
    
    # 2. İşçi: LightGBM
    model_lgb = lgb.LGBMRegressor(
        n_estimators=2500, learning_rate=0.015, max_depth=6, num_leaves=31, 
        random_state=42, n_jobs=-1, verbose=-1
    )
    model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(150, verbose=False)])
    oof_lgb[val_idx] = model_lgb.predict(X_val)
    test_lgb += model_lgb.predict(X_test) / skf.n_splits
    
    # 3. İşçi: XGBoost
    model_xgb = xgb.XGBRegressor(
        n_estimators=2500, learning_rate=0.015, max_depth=5, enable_categorical=True, 
        random_state=42, tree_method='hist', n_jobs=-1
    )
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_val)
    test_xgb += model_xgb.predict(X_test) / skf.n_splits
    
    print(f"CatBoost RMSE: {root_mean_squared_error(y_val, oof_cat[val_idx]):.5f}")
    print(f"LightGBM RMSE: {root_mean_squared_error(y_val, oof_lgb[val_idx]):.5f}")
    print(f"XGBoost  RMSE: {root_mean_squared_error(y_val, oof_xgb[val_idx]):.5f}")

# ---------------------------------------------------------
# 4. META-MODEL EĞİTİMİ (SEVİYE-1 YÖNETİCİ)
# ---------------------------------------------------------
print("\n🧠 Seviye-1 Meta Model (Yönetici) Karar Veriyor...")

X_meta_train = np.column_stack((oof_cat, oof_lgb, oof_xgb))
X_meta_test = np.column_stack((test_cat, test_lgb, test_xgb))

meta_model = RidgeCV(alphas=(0.1, 1.0, 10.0))
meta_model.fit(X_meta_train, y)

nihai_train_tahminleri = meta_model.predict(X_meta_train)
nihai_test_tahminleri = meta_model.predict(X_meta_test)

stacking_rmse = root_mean_squared_error(y, nihai_train_tahminleri)
print(f"\n=============================================")
print(f"🏆 NİHAİ STACKING CV RMSE: {stacking_rmse:.5f} 🏆")
print(f"Meta-Model Ağırlıkları (Cat, LGB, XGB): {meta_model.coef_}")
print(f"=============================================")

# ---------------------------------------------------------
# 5. DOSYAYI KAYDETME
# ---------------------------------------------------------
submission_stacking = pd.DataFrame({
    'id': test_id, 
    hedef_kolon: nihai_test_tahminleri
})

dosya_adi = 'submission_ULTIMATE_Stacking.csv'
submission_stacking.to_csv(dosya_adi, index=False)
print(f"\nZirveye oynayacak en gelişmiş dosya hazır: {dosya_adi}")