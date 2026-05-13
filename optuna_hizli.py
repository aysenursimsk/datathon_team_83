import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import root_mean_squared_error
from catboost import CatBoostRegressor
import optuna
import warnings
warnings.filterwarnings('ignore')

# 1. VERİ HAZIRLIĞI
def veri_hazirligi_ve_muhendislik(df):
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
    return df

print("Veriler Okunuyor...")
train_df = pd.read_csv('train.csv')
islenmis_train = veri_hazirligi_ve_muhendislik(train_df)

hedef_kolon = 'bilissel_performans_skoru'
X = islenmis_train.drop(columns=[hedef_kolon])
y = islenmis_train[hedef_kolon]
mevcut_kategorikler = X.select_dtypes(include=['object', 'category']).columns.tolist()

# 2. OPTUNA İLE HİPERPARAMETRE OPTİMİZASYONU
print("Optuna Başlıyor...")
def objective(trial):
    param = {
        'iterations': trial.suggest_int('iterations', 1000, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),
        'cat_features': mevcut_kategorikler,
        'eval_metric': 'RMSE',
        'random_seed': 42,
        'verbose': 0,
        # DİKKAT: NVIDIA EKRAN KARTIN VARSA BURAYI 'GPU' YAP
        'task_type': 'CPU' 
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_skorlari = []
    y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')

    for train_idx, val_idx in skf.split(X, y_binned):
        X_train_opt, y_train_opt = X.iloc[train_idx], y.iloc[train_idx]
        X_val_opt, y_val_opt = X.iloc[val_idx], y.iloc[val_idx]

        model = CatBoostRegressor(**param)
        model.fit(X_train_opt, y_train_opt, eval_set=(X_val_opt, y_val_opt), early_stopping_rounds=100)
        
        preds = model.predict(X_val_opt)
        rmse = root_mean_squared_error(y_val_opt, preds)
        cv_skorlari.append(rmse)

    return np.mean(cv_skorlari)

# Optuna çalışmasını başlat
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print("\n🏆 EN İYİ PARAMETRELER BULUNDU:")
print(study.best_params)
print(f"Buna Karşılık Gelen CV RMSE: {study.best_value:.5f}")