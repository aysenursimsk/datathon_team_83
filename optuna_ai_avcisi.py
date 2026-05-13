import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

print("🤖 OPTUNA YAPAY ZEKA AVCISI BAŞLATILDI (HEDEF: 1.19) 🤖\n")

# 1. VERİLERİ OKUMA
train = pd.read_csv('train_ULTRA_temiz.csv')
hedef_kolon = 'bilissel_performans_skoru'

y = train[hedef_kolon]
X = train.drop(columns=[hedef_kolon])

# Kategorik değişkenleri tanıtma (Sayısal illüzyonu önlemek için)
kategorikler = ['cinsiyet', 'meslek', 'ulke', 'kronotip', 'ruh_sagligi_durumu', 'mevsim', 'gun_tipi']
for col in kategorikler:
    X[col] = X[col].astype(int).astype('category')

# 2. OPTUNA HEDEF FONKSİYONU
def objective(trial):
    # Optuna'nın deneyeceği parametre uzayı (En geniş ve en mantıklı sınırlar)
    params = {
        'iterations': trial.suggest_int('iterations', 1000, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'random_strength': trial.suggest_float('random_strength', 0.1, 5.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'cat_features': kategorikler,
        'random_seed': 42,
        'verbose': 0
    }

    # Hızlı ama güvenilir 3-Fold CV
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_skorlari = []

    for train_idx, val_idx in kf.split(X):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=100)
        
        preds = model.predict(X_va)
        rmse = root_mean_squared_error(y_va, preds)
        cv_skorlari.append(rmse)

    return np.mean(cv_skorlari)

# 3. EVRİMSEL ARAMA SÜRECİ
print("Evrimsel Algoritma CatBoost için en kusursuz DNA'yı arıyor...")
print("Bu işlem 30 farklı genetik mutasyon deneyecek. Lütfen bekleyin.\n")

# Yönü "minimize" (RMSE'yi en aza indirmek) olarak ayarlıyoruz
study = optuna.create_study(direction='minimize', study_name="CatBoost_Optimization")
study.optimize(objective, n_trials=30)  # Süre sıkıntın yoksa burayı 50 yapabilirsin

print("\n=======================================================")
print("🏆 OPTUNA ARAMASI TAMAMLANDI 🏆")
print(f"EN İYİ CV RMSE SKORU: {study.best_value:.5f}")
print("EN KUSURSUZ PARAMETRELER (DNA):")
for key, value in study.best_params.items():
    print(f"    '{key}': {value},")
print("=======================================================")