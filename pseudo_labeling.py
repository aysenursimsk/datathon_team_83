import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import root_mean_squared_error
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# 1. VERİLERİ VE EN İYİ TAHMİNLERİMİZİ OKUMA
print("Veriler ve Pseudo-Label (Sözde-Etiket) Dosyası Okunuyor...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test_x.csv')

# En iyi dosyamızı okuyoruz (1.205'lik skor getiren dosya)
try:
    pseudo_labels = pd.read_csv('submission_GRANDMASTER_Deep_Stacking.csv')
except FileNotFoundError:
    print("HATA: 'submission_GRANDMASTER_Deep_Stacking.csv' dosyası bulunamadı!")
    exit()

test_id = test_df['id'].copy() if 'id' in test_df.columns else None

if 'id' in train_df.columns: train_df = train_df.drop(columns=['id'])
if 'id' in test_df.columns: test_df = test_df.drop(columns=['id'])

# 2. PSEUDO-LABELING (TEST SETİNİ TRAIN SETİNE KATMA)
print("Sözde-Etiketler Test Setine Entegre Ediliyor ve Train ile Birleştiriliyor...")
pseudo_test = test_df.copy()
pseudo_test['bilissel_performans_skoru'] = pseudo_labels['bilissel_performans_skoru']

# Train ve Pseudo-Test'i alt alta birleştir (Devasa Veri Seti)
dev_train_df = pd.concat([train_df, pseudo_test], axis=0).reset_index(drop=True)

# 3. STANDART VERİ HAZIRLIĞI (Bu devasa veri üzerinde)
ulke_mapping = {'Spain': 'Ispanya', 'South Korea': 'Guney Kore', 'Sweden': 'Isvec'}
dev_train_df['ulke'] = dev_train_df['ulke'].replace(ulke_mapping)
test_df['ulke'] = test_df['ulke'].replace(ulke_mapping)

dev_train_df['meslek'] = dev_train_df['meslek'].replace({'Lawyer': 'Avukat'})
test_df['meslek'] = test_df['meslek'].replace({'Lawyer': 'Avukat'})

kategorik_kolonlar = dev_train_df.select_dtypes(include=['object', 'category']).columns
for col in kategorik_kolonlar:
    dev_train_df[col] = dev_train_df[col].fillna('Bilinmiyor')
    test_df[col] = test_df[col].fillna('Bilinmiyor')
        
sayisal_kolonlar = dev_train_df.select_dtypes(exclude=['object', 'category']).columns
for col in sayisal_kolonlar:
    if col != 'bilissel_performans_skoru':
        dev_train_df[col] = dev_train_df[col].fillna(dev_train_df[col].median())
        test_df[col] = test_df[col].fillna(test_df[col].median())

# Elit Özellikler
for df in [dev_train_df, test_df]:
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

X_dev = dev_train_df.drop(columns=['bilissel_performans_skoru'])
y_dev = dev_train_df['bilissel_performans_skoru']
X_test_final = test_df.copy()

kategorik_kolonlar = X_dev.select_dtypes(include=['object', 'category']).columns.tolist()

# 4. YENİ EĞİTİM (Devasa Veri Üzerinde Sadece CatBoost)
print(f"🚀 Toplam Eğitim Verisi Boyutu: {len(X_dev)} satır (Orijinal + Test)")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_binned = pd.qcut(y_dev, q=10, labels=False, duplicates='drop')

test_tahminler = np.zeros(len(X_test_final))
oof_tahminler = np.zeros(len(X_dev))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_dev, y_binned)):
    X_train, y_train = X_dev.iloc[train_idx], y_dev.iloc[train_idx]
    X_val, y_val = X_dev.iloc[val_idx], y_dev.iloc[val_idx]
    
    model = CatBoostRegressor(
        iterations=2594, learning_rate=0.0195, depth=5, l2_leaf_reg=4,
        cat_features=kategorik_kolonlar, eval_metric='RMSE', random_seed=42, verbose=0
    )
    
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=150)
    
    val_preds = model.predict(X_val)
    oof_tahminler[val_idx] = val_preds
    fold_rmse = root_mean_squared_error(y_val, val_preds)
    print(f"Fold {fold + 1} RMSE Skoru: {fold_rmse:.5f}")
    
    test_tahminler += model.predict(X_test_final) / skf.n_splits

genel_rmse = root_mean_squared_error(y_dev, oof_tahminler)
print(f"\n==================================================")
print(f"🔥 PSEUDO-LABEL CV RMSE: {genel_rmse:.5f} 🔥")
print(f"==================================================")

# 0-10 Arasına Tıraşlama
test_tahminler = np.clip(test_tahminler, 0.0, 10.0)

submission_final = pd.DataFrame({'id': test_id, 'bilissel_performans_skoru': test_tahminler})
dosya_adi = 'submission_PSEUDO_LABEL.csv'
submission_final.to_csv(dosya_adi, index=False)
print(f"Hileli Dosyanız Hazır: {dosya_adi}")