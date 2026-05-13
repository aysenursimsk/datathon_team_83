import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_error

print("🎯 OPERASYON 0.81: HATA TEMİZLİĞİ VE SIZINTI AVCI BAŞLATILDI 🎯\n")

# 1. VERİLERİ OKU
train = pd.read_csv('train.csv')
test = pd.read_csv('test_x.csv')
hedef = 'bilissel_performans_skoru'

# Kategorik temizlik ve Standartlaştırma
mapping = {'Spain': 'Ispanya', 'South Korea': 'Guney Kore', 'Sweden': 'Isvec', 'Lawyer': 'Avukat'}
train = train.replace(mapping)
test = test.replace(mapping)

test_id = test['id'].copy()
train_y = train[hedef].copy()
train_X = train.drop(columns=['id', hedef]).copy()
test_X = test.drop(columns=['id']).copy()

# --- KRİTİK: KATEGORİK SÜTUNLARDAKİ NaN DEĞERLERİ DOLDUR ---
cat_cols = ['meslek', 'ulke', 'cinsiyet', 'kronotip', 'ruh_sagligi_durumu', 'mevsim', 'gun_tipi']
for col in cat_cols:
    train_X[col] = train_X[col].fillna('Bilinmiyor').astype(str)
    test_X[col] = test_X[col].fillna('Bilinmiyor').astype(str)

# 2. TEKNİK: KNN MAPPING (İkizlerden Skor Çalma)
print("-> Verideki 'Klonlar' taranıyor...")
numeric_cols = train_X.select_dtypes(include=[np.number]).columns
train_num = (train_X[numeric_cols] - train_X[numeric_cols].mean()) / train_X[numeric_cols].std()
test_num = (test_X[numeric_cols] - train_X[numeric_cols].mean()) / train_X[numeric_cols].std()

knn = KNeighborsRegressor(n_neighbors=1) 
knn.fit(train_num.fillna(0), train_y)
test_X['knn_magic_score'] = knn.predict(test_num.fillna(0))
train_X['knn_magic_score'] = train_y

# 3. TEKNİK: FORMÜL KIRICI ETKİLEŞİMLER
print("-> Gizli formül etkileşimleri üretiliyor...")
for df in [train_X, test_X]:
    df['toplam_uyku_etkisi'] = (df['rem_yuzdesi'] + df['derin_uyku_yuzdesi']) * (10 - df['stres_skoru'])
    df['zihinsel_direnc'] = df['gunluk_adim_sayisi'] / (df['stres_skoru'] + 1)

# 4. TEKNİK: K-FOLD TARGET ENCODING (Hatasız Versiyon)
print("-> Kategorik ağırlıklar işleniyor (Target Encoding)...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for col in cat_cols:
    train_X[f'{col}_target'] = 0.0
    test_X[f'{col}_target'] = 0.0
    
    for tr_idx, val_idx in kf.split(train_X):
        X_tr, X_val = train_X.iloc[tr_idx], train_X.iloc[val_idx]
        means = train_y.iloc[tr_idx].groupby(X_tr[col]).mean()
        # ChainedAssignment hatasını önlemek için .loc kullanıyoruz
        train_X.loc[train_X.index[val_idx], f'{col}_target'] = train_X.loc[train_X.index[val_idx], col].map(means)
    
    means = train_y.groupby(train_X[col]).mean()
    test_X[f'{col}_target'] = test_X[col].map(means)
    
    # Boş kalan yerleri genel ortalamayla doldur (Inplace hatası düzeltildi)
    train_X[f'{col}_target'] = train_X[f'{col}_target'].fillna(train_y.mean())
    test_X[f'{col}_target'] = test_X[f'{col}_target'].fillna(train_y.mean())

# 5. MODEL: CATBOOST (Hatasız ve Güçlü)
print("\n🚀 NİHAİ SAVAŞ: CATBOOST 0.81 HEDEFİYLE EĞİTİLİYOR...")
model = CatBoostRegressor(
    iterations=3000, 
    learning_rate=0.015, 
    depth=8, 
    l2_leaf_reg=5,
    cat_features=cat_cols, # Artık hepsi string ve NaN içermiyor
    random_seed=42, 
    verbose=500
)

model.fit(train_X, train_y)

# Tahmin ve Kayıt
preds = model.predict(test_X)
submission = pd.DataFrame({'id': test_id, 'bilissel_performans_skoru': np.clip(preds, 0.0, 10.0)})
submission.to_csv('submission_FINAL_081.csv', index=False)

print("\n==================================================")
print("🔥 Hatalar Giderildi, 0.81 Yolculuğu Başladı! 🔥")
print("Dosya: submission_FINAL_081.csv")
print("==================================================")