import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

print("🦅 TAKIM 83 - 0.22 HEDEFİ: GİZLİ DENKLEM SÖKÜCÜ BAŞLATILDI 🦅\n")

# 1. HAM VERİLERİ OKU
train = pd.read_csv('train.csv')
test = pd.read_csv('test_x.csv')
hedef = 'bilissel_performans_skoru'

# Kategorik düzeltmeler
mapping = {'Spain': 'Ispanya', 'South Korea': 'Guney Kore', 'Sweden': 'Isvec', 'Lawyer': 'Avukat'}
train = train.replace(mapping).fillna('Bilinmiyor')
test = test.replace(mapping).fillna('Bilinmiyor')

test_id = test['id']
y = train[hedef]
X = train.drop(columns=['id', hedef])
X_test = test.drop(columns=['id'])

# 2. SİHİRLİ FORMÜL ENJEKSİYONU (Korelasyon Analizinden Gelenler)
print("-> Organizatörlerin gizli çarpanları veriye enjekte ediliyor...")
for df in [X, X_test]:
    # Senin bulduğun o 0.66'lık devasa sinyaller:
    df['magic_ratio_1'] = df['stres_skoru'] / (df['rem_yuzdesi'] + 1e-6)
    df['magic_ratio_2'] = df['stres_skoru'] / (df['derin_uyku_yuzdesi'] + 1e-6)
    df['magic_mul_1'] = df['stres_skoru'] * df['dinlenik_nabiz_bpm']
    df['magic_mul_2'] = df['stres_skoru'] * df['gunluk_calisma_saati']
    
    # Ekstra Formül Tahminleri (Sentetik Veri Klasiği)
    df['total_sleep'] = df['rem_yuzdesi'] + df['derin_uyku_yuzdesi']
    df['sleep_stress_interaction'] = df['total_sleep'] * (10 - df['stres_skoru'])

# 3. KATEGORİK TANIMLAMA
cat_features = ['cinsiyet', 'meslek', 'ulke', 'kronotip', 'ruh_sagligi_durumu', 'mevsim', 'gun_tipi']
for col in cat_features:
    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

# 4. DERİN MANTIK EĞİTİMİ (Depth 10 ve Çok Düşük Learning Rate)
print("\n🚀 Model, Gizli Denklemi Çözmek İçin Derin Analize Başlıyor...")
# 0.22'ye inmek için modelin "aşırı detaycı" olması lazım.
model = CatBoostRegressor(
    iterations=5000,           # Çok daha uzun eğitim
    learning_rate=0.01,        # Çok daha yavaş ve hassas öğrenme
    depth=10,                  # ÇOK ÖNEMLİ: Karmaşık mantık bloklarını yakalamak için
    l2_leaf_reg=3,
    cat_features=cat_features,
    random_seed=42,
    verbose=500
)

model.fit(X, y)

# 5. DOSYA KAYDETME
preds = model.predict(X_test)
submission = pd.DataFrame({'id': test_id, hedef: np.clip(preds, 0.0, 10.0)})
dosya_adi = 'submission_ULTIMATE_022_BREAK.csv'
submission.to_csv(dosya_adi, index=False)

print("\n==================================================")
print(f"🔥 FORMÜL KIRILDI! Eğitim RMSE: {model.get_best_score()['learning']['RMSE']:.5f} 🔥")
print(f"Liderlik Dosyan Hazır: {dosya_adi}")
print("==================================================")