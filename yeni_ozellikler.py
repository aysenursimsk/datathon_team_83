import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import root_mean_squared_error
from sklearn.cluster import KMeans
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. TEMEL HAZIRLIK VE MÜHENDİSLİK FONKSİYONU
# ---------------------------------------------------------
def veri_hazirligi_ve_ileri_muhendislik(df):
    df = df.copy()
    
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    # Standartlaştırma
    ulke_mapping = {'Spain': 'Ispanya', 'South Korea': 'Guney Kore', 'Sweden': 'Isvec'}
    if 'ulke' in df.columns:
        df['ulke'] = df['ulke'].replace(ulke_mapping)
        
    if 'meslek' in df.columns:
        df['meslek'] = df['meslek'].replace({'Lawyer': 'Avukat'})

    # Eksik Veri Doldurma
    kategorik_kolonlar = df.select_dtypes(include=['object', 'category']).columns
    for col in kategorik_kolonlar:
        df[col] = df[col].fillna('Bilinmiyor')
            
    sayisal_kolonlar = df.select_dtypes(exclude=['object', 'category']).columns
    for col in sayisal_kolonlar:
        if col in df.columns and col != 'bilissel_performans_skoru':
            df[col] = df[col].fillna(df[col].median())

    # Dönüşümler
    if 'uyku_oncesi_kafein_mg' in df.columns:
        df['uyku_oncesi_kafein_mg'] = np.log1p(df['uyku_oncesi_kafein_mg'])
        
    if 'uyku_oncesi_ekran_suresi_dk' in df.columns:
        p99 = df['uyku_oncesi_ekran_suresi_dk'].quantile(0.99)
        df['uyku_oncesi_ekran_suresi_dk'] = df['uyku_oncesi_ekran_suresi_dk'].clip(upper=p99)

    # FAZ 1: Önceki Başarılı Mühendislikler
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

    # FAZ 2: SIFIRDAN YENİ ÖZELLİK MÜHENDİSLİĞİ (1.200 ALTI İÇİN)
    
    # Yeni 1: Biyolojik Uyku Açığı (Yaşlanan bedenin restorasyon kapasitesi)
    if 'yas' in df.columns and 'derin_uyku_yuzdesi' in df.columns:
        df['biyolojik_uyku_acigi'] = df['yas'] / (df['derin_uyku_yuzdesi'] + 1)
        
    # Yeni 2: VKI Optimizasyonu (İdeal kilodan olan mutlak sapma)
    if 'vucut_kitle_indeksi' in df.columns:
        df['vki_sapmasi'] = np.abs(df['vucut_kitle_indeksi'] - 22.5)
        
    # Yeni 3: Hareket ve Stres Paradoksu (Fiziksel yorgunluk mu, spor mu?)
    if 'gunluk_adim_sayisi' in df.columns and 'stres_skoru' in df.columns:
        df['hareket_stres_orani'] = df['gunluk_adim_sayisi'] / (df['stres_skoru'] + 1)

    return df

# ---------------------------------------------------------
# 2. VERİLERİ İŞLEME VE K-MEANS KÜMELEME
# ---------------------------------------------------------
print("Veriler Okunuyor ve Gelişmiş Mühendislik Uygulanıyor...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test_x.csv')

test_id = test_df['id'].copy() if 'id' in test_df.columns else None

islenmis_train = veri_hazirligi_ve_ileri_muhendislik(train_df)
islenmis_test = veri_hazirligi_ve_ileri_muhendislik(test_df)

# Yeni 4: K-Means Yaşam Tarzı Kümelemesi (Sızıntıyı önlemek için sadece Train'de fit ediyoruz)
print("K-Means ile Yaşam Tarzı Kümeleri Oluşturuluyor...")
kmeans_kolonlari = ['stres_skoru', 'gecelik_uyanma_sayisi', 'gunluk_calisma_saati']
if all(col in islenmis_train.columns for col in kmeans_kolonlari):
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    islenmis_train['yasam_tarzi_kumesi'] = kmeans.fit_predict(islenmis_train[kmeans_kolonlari])
    islenmis_test['yasam_tarzi_kumesi'] = kmeans.predict(islenmis_test[kmeans_kolonlari])
    
    # Modelin bunları sayısal bir büyüklük değil, kategori olarak algılaması için string yapıyoruz
    islenmis_train['yasam_tarzi_kumesi'] = islenmis_train['yasam_tarzi_kumesi'].astype(str)
    islenmis_test['yasam_tarzi_kumesi'] = islenmis_test['yasam_tarzi_kumesi'].astype(str)

hedef_kolon = 'bilissel_performans_skoru'
X = islenmis_train.drop(columns=[hedef_kolon])
y = islenmis_train[hedef_kolon]
X_test = islenmis_test.copy()

mevcut_kategorikler = X.select_dtypes(include=['object', 'category']).columns.tolist()

# ---------------------------------------------------------
# 3. NİHAİ MODEL EĞİTİMİ (ALTIN PARAMETRELERLE)
# ---------------------------------------------------------
print("Optuna'nın Bulduğu Altın Parametrelerle 5-Fold Eğitimi Başlıyor...")
y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_tahminler = np.zeros(len(X))
test_tahminler = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model = CatBoostRegressor(
        iterations=2594, 
        learning_rate=0.01950096908168973, 
        depth=5, 
        l2_leaf_reg=4,
        cat_features=mevcut_kategorikler,
        eval_metric='RMSE',
        random_seed=42,
        verbose=0
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=150,
        use_best_model=True
    )
    
    val_preds = model.predict(X_val)
    oof_tahminler[val_idx] = val_preds
    
    fold_rmse = root_mean_squared_error(y_val, val_preds)
    print(f"Fold {fold + 1} RMSE Skoru: {fold_rmse:.5f} | Durulan İterasyon: {model.get_best_iteration()}")
    
    test_tahminler += model.predict(X_test) / skf.n_splits

genel_rmse = root_mean_squared_error(y, oof_tahminler)
print(f"\n--- YENİ NESİL YEREL CV RMSE: {genel_rmse:.5f} ---")

submission_final = pd.DataFrame({
    'id': test_id, 
    hedef_kolon: test_tahminler
})

dosya_adi = 'submission_1_200_alti_denemesi.csv'
submission_final.to_csv(dosya_adi, index=False)
print(f"\nDosya başarıyla kaydedildi: {dosya_adi}")

# Modelin en çok değer verdiği ilk 20 özelliği listele
onem_dereceleri = model.get_feature_importance(prettified=True)
print("\n--- 🔍 EN ÖNEMLİ 20 ÖZELLİK ---")
print(onem_dereceleri.head(20))