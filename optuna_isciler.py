import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import optuna
import warnings
warnings.filterwarnings('ignore')

def tam_kapsamli_veri_hazirligi():
    # 1. Okuma ve Temel Temizlik
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

    # 2. Makro-Bağlam (Grup İstatistikleri - Sızıntısız)
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

    # 3. Eksik Veri Doldurma ve Elit Özellikler
    for df in [train_df, test_df]:
        kategorik_kolonlar = df.select_dtypes(include=['object', 'category']).columns
        for col in kategorik_kolonlar: df[col] = df[col].fillna('Bilinmiyor')
                
        sayisal_kolonlar = df.select_dtypes(exclude=['object', 'category']).columns
        for col in sayisal_kolonlar:
            if col in df.columns and col != 'bilissel_performans_skoru':
                df[col] = df[col].fillna(df[col].median())

        if 'uyku_oncesi_kafein_mg' in df.columns: df['uyku_oncesi_kafein_mg'] = np.log1p(df['uyku_oncesi_kafein_mg'])
        if 'uyku_oncesi_ekran_suresi_dk' in df.columns:
            p99 = df['uyku_oncesi_ekran_suresi_dk'].quantile(0.99)
            df['uyku_oncesi_ekran_suresi_dk'] = df['uyku_oncesi_ekran_suresi_dk'].clip(upper=p99)

        df['sinerjik_zihinsel_yuk'] = df['stres_skoru'] * df['gunluk_calisma_saati']
        df['uyku_onarim_indeksi'] = (df['rem_yuzdesi'] + df['derin_uyku_yuzdesi']) / (df['gecelik_uyanma_sayisi'] + 1e-5)
        
        hafta_sonu_kolonu = 'hafta_sonu_uyku_farki' if 'hafta_sonu_uyku_farki' in df.columns else 'hafta_sonu_uyku_farki_saat'
        df['sosyal_jet_lag'] = (df[hafta_sonu_kolonu] > 2.0).astype(int)
        
        kafein_esik = df['uyku_oncesi_kafein_mg'].median()
        ekran_esik = df['uyku_oncesi_ekran_suresi_dk'].median()
        df['kotu_uyku_hijyeni'] = ((df['kronotip'] == 'Gece') & (df['uyku_oncesi_kafein_mg'] > kafein_esik) & (df['uyku_oncesi_ekran_suresi_dk'] > ekran_esik)).astype(int)
        df['hareket_stres_orani'] = df['gunluk_adim_sayisi'] / (df['stres_skoru'] + 1)

    hedef_kolon = 'bilissel_performans_skoru'
    X = train_df.drop(columns=[hedef_kolon])
    y = train_df[hedef_kolon]
    X_test = test_df.copy()

    kategorik_kolonlar = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in kategorik_kolonlar:
        X[col] = X[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    return X, y, X_test, test_id

print("Veriler Okunuyor ve Tüm Mimariler Birleştiriliyor...")
X, y, X_test, test_id = tam_kapsamli_veri_hazirligi()
y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# --- LIGHTGBM OPTUNA ---
print("\n🔍 LightGBM Hiperparametre Avı Başlıyor (20 Deneme)...")
def objective_lgb(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    cv_scores = []
    for train_idx, val_idx in skf.split(X, y_binned):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        model = lgb.LGBMRegressor(**param)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
        preds = model.predict(X_va)
        cv_scores.append(root_mean_squared_error(y_va, preds))
    return np.mean(cv_scores)

study_lgb = optuna.create_study(direction='minimize')
study_lgb.optimize(objective_lgb, n_trials=20)

# --- XGBOOST OPTUNA ---
print("\n🔍 XGBoost Hiperparametre Avı Başlıyor (20 Deneme)...")
def objective_xgb(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'enable_categorical': True,
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1
    }
    cv_scores = []
    for train_idx, val_idx in skf.split(X, y_binned):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        model = xgb.XGBRegressor(**param)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        preds = model.predict(X_va)
        cv_scores.append(root_mean_squared_error(y_va, preds))
    return np.mean(cv_scores)

study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=20)

print("\n==================================================")
print("🏆 LIGHTGBM İÇİN EN İYİ PARAMETRELER:")
print(study_lgb.best_params)
print("--------------------------------------------------")
print("🏆 XGBOOST İÇİN EN İYİ PARAMETRELER:")
print(study_xgb.best_params)
print("==================================================")