import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("🛡️ TAKIM 83: REHABİLİTASYON VE GÜVENLİ SÜRÜŞ BAŞLADI 🛡️\n")

# 1. TEMİZ VERİYE DÖNÜŞ
train = pd.read_csv('train.csv')
test = pd.read_csv('test_x.csv')
hedef = 'bilissel_performans_skoru'

test_id = test['id'].copy()
mapping = {'Spain': 'Ispanya', 'South Korea': 'Guney Kore', 'Sweden': 'Isvec', 'Lawyer': 'Avukat'}

for df in [train, test]:
    df.replace(mapping, inplace=True)
    # Sayısal ve Kategorik temizliği 'Safe' moda alıyoruz
    for col in df.columns:
        if col not in ['id', hedef]:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('Bilinmiyor').astype(str)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())

# 2. SADECE EN GÜÇLÜ VE GENEL FORMÜLLER (Overfit yapmayanlar)
for df in [train, test]:
    # Senin bulduğun o %67 korelasyonlu altın oran
    df['stres_rem_index'] = df['stres_skoru'] / (df['rem_yuzdesi'] + 1e-6)
    # Fiziksel ve zihinsel yorgunluk (Daha geniş bir bakış)
    df['yorgunluk_skoru'] = (df['stres_skoru'] * df['gunluk_calisma_saati']) / 100

# 3. ROBUST (DAYANIKLI) CATBOOST
cat_features = ['cinsiyet', 'meslek', 'ulke', 'kronotip', 'ruh_sagligi_durumu', 'mevsim', 'gun_tipi']
X = train.drop(columns=['id', hedef])
y = train[hedef]
X_test = test.drop(columns=['id'])

print("-> Model 'Ezber' modundan 'Öğrenme' moduna alınıyor (Depth: 7)...")
# Derinliği 12'den 7'ye düşürerek modelin 'genelleme' yapmasını sağlıyoruz
model = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.02,
    depth=7,                # GÜVENLİ DERİNLİK
    l2_leaf_reg=10,         # EZBERLEMEYİ ÖNLEYEN SERT CEZA
    cat_features=cat_features,
    random_seed=42,
    verbose=500
)

# 10-Fold CV ile skoru mühürleyelim (Sürpriz istemiyoruz)
kf = KFold(n_splits=10, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=100, verbose=0)
    
    oof_preds[val_idx] = model.predict(X_va)
    test_preds += model.predict(X_test) / 10
    print(f"Fold {fold+1} tamamlandı.")

cv_rmse = root_mean_squared_error(y, oof_preds)

print("\n==================================================")
print(f"✅ GERÇEKÇİ CV RMSE: {cv_rmse:.5f}")
print("==================================================")

submission = pd.DataFrame({'id': test_id, hedef: np.clip(test_preds, 0.0, 10.0)})
submission.to_csv('submission_SAFE_RECOVERY.csv', index=False)
print("Güvenli Dosyan Hazır: submission_SAFE_RECOVERY.csv")