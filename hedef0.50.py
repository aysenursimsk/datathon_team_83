import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("👑 TAKIM 83: 0.22 HEDEFİ - FORMÜL VE KATEGORİK ENJEKSİYON OPERASYONU 👑\n")

# 1. VERİLERİ OKU VE TEMİZLE
train = pd.read_csv('train.csv')
test = pd.read_csv('test_x.csv')
hedef = 'bilissel_performans_skoru'

test_id = test['id'].copy()
mapping = {'Spain': 'Ispanya', 'South Korea': 'Guney Kore', 'Sweden': 'Isvec', 'Lawyer': 'Avukat'}

for df in [train, test]:
    df.replace(mapping, inplace=True)
    # Sayısal sütunları temizle
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if col != 'id' and col != hedef:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())

# 2. SİHİRLİ ÖZELLİK MÜHENDİSLİĞİ (KODU KIRAN ORANLAR)
def add_magic_features(df):
    # En yüksek korelasyonu veren %67'lik anahtar
    df['magic_ratio_1'] = df['stres_skoru'] / (df['rem_yuzdesi'] + 1e-6)
    # İkinci seviye etkileşimler (%65 korelasyon)
    df['magic_ratio_2'] = (df['stres_skoru'] * df['dinlenik_nabiz_bpm']) / (df['rem_yuzdesi'] + 1e-6)
    df['magic_ratio_3'] = (df['stres_skoru'] * df['oda_sicakligi_celsius']) / (df['rem_yuzdesi'] + 1e-6)
    # Fizyolojik yük
    df['magic_load'] = (df['stres_skoru'] * df['gunluk_calisma_saati']) / (df['derin_uyku_yuzdesi'] + 1)
    return df

train = add_magic_features(train)
test = add_magic_features(test)

# 3. KATEGORİK OFFSET (ZİRVEYE GİDEN YOL)
# Bazı mesleklerin/ülkelerin skora doğrudan + veya - etkisi var.
cat_cols = ['meslek', 'ulke', 'cinsiyet', 'kronotip', 'ruh_sagligi_durumu']

for col in cat_cols:
    # Target Encoding: Kategorilerin ortalama puanlarını modele enjekte et
    means = train.groupby(col)[hedef].mean()
    train[f'{col}_impact'] = train[col].map(means)
    test[f'{col}_impact'] = test[col].map(means)
    # Boşları doldur
    train[f'{col}_impact'] = train[f'{col}_impact'].fillna(train[hedef].mean())
    test[f'{col}_impact'] = test[f'{col}_impact'].fillna(train[hedef].mean())

# Model için veriyi hazırla
y = train[hedef]
X = train.drop(columns=['id', hedef])
X_test = test.drop(columns=['id'])

# CatBoost'un kategorik olduğunu bilmesi gerekenler
cat_features = [col for col in cat_cols if col in X.columns]
for col in cat_features:
    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

# 4. DERİN ÖĞRENME MODELİ (DEPTH 12)
print("\n🚀 0.22 Barajı İçin Derin Analiz Başlıyor...")
model = CatBoostRegressor(
    iterations=5000,
    learning_rate=0.01,
    depth=12,           # Ağaç derinliğini sonuna kadar zorluyoruz
    l2_leaf_reg=5,
    random_strength=1,
    cat_features=cat_features,
    random_seed=42,
    verbose=500
)

model.fit(X, y)

# 5. TAHMİN VE KAYIT
preds = model.predict(X_test)
# Bilişsel skor 0 ile 10 arasındadır, dışına taşanları kırp
submission = pd.DataFrame({'id': test_id, hedef: np.clip(preds, 0.0, 10.0)})
submission.to_csv('submission_FINAL_TARGET_022.csv', index=False)

print("\n==================================================")
print(f"🔥 İşlem Tamam! Eğitim RMSE: {model.get_best_score()['learning']['RMSE']:.5f}")
print("==================================================")