import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("🚀 TAKIM 83: 0.50 HEDEFİ - NİHAİ VE HATASIZ MODEL BAŞLATILDI 🚀\n")

# 1. HAM VERİLERİ OKU
# (En saf halleriyle train.csv ve test_x.csv kullanıyoruz ki veri kaybı olmasın)
train = pd.read_csv('train.csv')
test = pd.read_csv('test_x.csv')
hedef = 'bilissel_performans_skoru'

# ID'leri güvenliğe al
test_id = test['id'].copy()

# 2. CERRAHİ TEMİZLİK VE VERİ TİPİ ZORLAMA
mapping = {'Spain': 'Ispanya', 'South Korea': 'Guney Kore', 'Sweden': 'Isvec', 'Lawyer': 'Avukat'}

cat_features = ['cinsiyet', 'meslek', 'ulke', 'kronotip', 'ruh_sagligi_durumu', 'mevsim', 'gun_tipi']
num_features = [col for col in train.columns if col not in cat_features + ['id', hedef]]

print("-> Kategorik ve sayısal sütunlar 'nan' hatasına karşı mühürleniyor...")
for df in [train, test]:
    df.replace(mapping, inplace=True)
    # Sayısal sütunları temizle
    for col in num_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
    # KATEGORİK HATA ÇÖZÜMÜ: Önce doldur, sonra string yap (CatBoost kuralı)
    for col in cat_features:
        df[col] = df[col].fillna('Bilinmiyor').astype(str)

# 3. 0.22 ALANLARIN GİZLİ SİLAHI: MATEMATİKSEL ORANLAR
print("-> %67 korelasyonlu sihirli oranlar enjekte ediliyor...")
for df in [train, test]:
    # Stres ve REM arasındaki o devasa sızıntıyı kullanıyoruz
    df['magic_ratio_1'] = df['stres_skoru'] / (df['rem_yuzdesi'] + 1e-6)
    df['magic_ratio_2'] = df['stres_skoru'] / (df['derin_uyku_yuzdesi'] + 1e-6)
    # Fizyolojik yük denklemi
    df['fizyolojik_yuk'] = (df['stres_skoru'] * df['dinlenik_nabiz_bpm']) / (df['rem_yuzdesi'] + 1)
    # Zihinsel yorgunluk
    df['zihinsel_yorgunluk'] = (df['stres_skoru'] * df['gunluk_calisma_saati']) / (df['derin_uyku_yuzdesi'] + 1)

# 4. HEDEF KODLAMA (TARGET ENCODING) - Sızıntısız
# (Bazı mesleklerin ve ülkelerin skora olan sabit etkisini modele öğretiyoruz)
print("-> Kategorik ağırlıklar (Smoothing ile) hesaplanıyor...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for col in ['meslek', 'ulke', 'kronotip']:
    train[f'{col}_impact'] = 0.0
    for tr_idx, val_idx in kf.split(train):
        X_tr, X_val = train.iloc[tr_idx], train.iloc[val_idx]
        means = X_tr.groupby(col)[hedef].mean()
        train.loc[train.index[val_idx], f'{col}_impact'] = X_val[col].map(means)
    
    # Test setine genel ortalamayı bas
    test[f'{col}_impact'] = test[col].map(train.groupby(col)[hedef].mean())
    # Eksikleri genel ortalamayla doldur
    global_mean = train[hedef].mean()
    train[f'{col}_impact'] = train[f'{col}_impact'].fillna(global_mean)
    test[f'{col}_impact'] = test[f'{col}_impact'].fillna(global_mean)

# 5. MODEL EĞİTİMİ (CATBOOST DEPTH 12)
X = train.drop(columns=['id', hedef])
y = train[hedef]
X_test = test.drop(columns=['id'])

print("\n🚀 0.50 Altı Hedefi İçin Derin Analiz Başlıyor (Bu işlem 10-15 dk sürebilir)...")
# Organizatörlerin gizli denklemini çözmek için derinliği 12 yaptık
model = CatBoostRegressor(
    iterations=5000,
    learning_rate=0.015,
    depth=12,               # Mikro kuralları yakalamak için en kritik ayar
    l2_leaf_reg=5,
    random_strength=1,
    cat_features=cat_features,
    random_seed=42,
    verbose=500
)

model.fit(X, y)

# 6. TAHMİN VE KAYIT
preds = model.predict(X_test)
# Bilişsel skoru 0-10 arasına sabitle
submission = pd.DataFrame({'id': test_id, 'bilissel_performans_skoru': np.clip(preds, 0.0, 10.0)})
dosya_adi = 'submission_ULTIMATE_ZIRVE_050.csv'
submission.to_csv(dosya_adi, index=False)

print("\n==================================================")
print(f"✅ İŞLEM TAMAM! Eğitim RMSE: {model.get_best_score()['learning']['RMSE']:.5f}")
print(f"Liderlik Dosyan Hazır: {dosya_adi}")
print("==================================================")