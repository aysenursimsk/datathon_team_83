import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("🌟 SEVİYE-2 GRANDMASTER MİMARİSİ BAŞLATILIYOR 🌟\n")

# ---------------------------------------------------------
# 1. TEMİZ VERİLERİ OKUMA VE ÖZELLİK MÜHENDİSLİĞİ
# ---------------------------------------------------------
print("1. Temiz Veriler Okunuyor ve Elit Sinyaller İşleniyor...")
train_df = pd.read_csv('train_temiz.csv')
test_df = pd.read_csv('test_temiz.csv')

# Datathon platformları için orijinal ID'leri garantiye alalım
try:
    test_id = pd.read_csv('test_x.csv')['id']
except FileNotFoundError:
    test_id = test_df['id'] if 'id' in test_df.columns else np.arange(len(test_df))

if 'id' in train_df.columns: train_df = train_df.drop(columns=['id'])
if 'id' in test_df.columns: test_df = test_df.drop(columns=['id'])

# Makro-Bağlam (Veri Sızıntısız)
meslek_stres_map = train_df.groupby('meslek')['stres_skoru'].mean().to_dict()
meslek_mesai_map = train_df.groupby('meslek')['gunluk_calisma_saati'].mean().to_dict()

for df in [train_df, test_df]:
    df['meslek_stres_ortalamasi'] = df['meslek'].map(meslek_stres_map)
    df['meslektastan_stres_sapmasi'] = df['stres_skoru'] - df['meslek_stres_ortalamasi']
    df['sinerjik_zihinsel_yuk'] = df['stres_skoru'] * df['gunluk_calisma_saati']
    df['uyku_onarim_indeksi'] = (df['rem_yuzdesi'] + df['derin_uyku_yuzdesi']) / (df['gecelik_uyanma_sayisi'] + 1e-5)
    df['hareket_stres_orani'] = df['gunluk_adim_sayisi'] / (df['stres_skoru'] + 1)

hedef_kolon = 'bilissel_performans_skoru'
X = train_df.drop(columns=[hedef_kolon])
y = train_df[hedef_kolon]
X_test = test_df.copy()

kategorik_kolonlar = X.select_dtypes(include=['object', 'category']).columns.tolist()

# ---------------------------------------------------------
# 2. SİNİR AĞLARI (MLP) İÇİN ÖZEL VERİ HAZIRLIĞI
# ---------------------------------------------------------
print("2. Derin Öğrenme İçin Uzamsal Veri Dönüşümü (One-Hot & Scaler) Yapılıyor...")
X_nn = pd.get_dummies(X, columns=kategorik_kolonlar, drop_first=True)
X_test_nn = pd.get_dummies(X_test, columns=kategorik_kolonlar, drop_first=True)

# Train ve Test'teki One-Hot sütunlarını eşitleme
X_test_nn = X_test_nn.reindex(columns=X_nn.columns, fill_value=0)

scaler = StandardScaler()
X_nn_scaled = scaler.fit_transform(X_nn)
X_test_nn_scaled = scaler.transform(X_test_nn)

# Ağaçlar için Kategorik Format
for col in kategorik_kolonlar:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# ---------------------------------------------------------
# 3. DEEP SEED BLENDING (4 FARKLI ALGORİTMA HARMANI)
# ---------------------------------------------------------
seed_list = [42, 2026, 777, 1024, 888]
final_test_predictions = np.zeros(len(X_test))
seed_skorlari = []

print(f"\n🚀 {len(seed_list)} Tohumlu Ağır Siklet Eğitim Başlıyor (Ağaçlar + Sinir Ağları)...")
y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')

for i, seed in enumerate(seed_list):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    oof_cat = np.zeros(len(X)); test_cat = np.zeros(len(X_test))
    oof_lgb = np.zeros(len(X)); test_lgb = np.zeros(len(X_test))
    oof_xgb = np.zeros(len(X)); test_xgb = np.zeros(len(X_test))
    oof_mlp = np.zeros(len(X)); test_mlp = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
        # Ağaç Verileri
        X_tr_tree, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va_tree, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        # Sinir Ağı Verileri (Ölçeklenmiş)
        X_tr_nn, X_va_nn = X_nn_scaled[train_idx], X_nn_scaled[val_idx]
        
        # --- 1. İŞÇİ: CATBOOST (Güvenilir Liman) ---
        model_cat = CatBoostRegressor(
            iterations=2000, learning_rate=0.02, depth=5, l2_leaf_reg=4,
            cat_features=kategorik_kolonlar, random_seed=seed, verbose=0
        )
        model_cat.fit(X_tr_tree, y_tr, eval_set=(X_va_tree, y_va), early_stopping_rounds=100)
        oof_cat[val_idx] = model_cat.predict(X_va_tree)
        test_cat += model_cat.predict(X_test) / skf.n_splits
        
        # --- 2. İŞÇİ: LIGHTGBM (Huber Loss ile Aykırı Değer Koruması) ---
        model_lgb = lgb.LGBMRegressor(
            n_estimators=1500, learning_rate=0.015, max_depth=5, num_leaves=31, 
            objective='huber', alpha=1.5, # Huber parametresi
            subsample=0.7, colsample_bytree=0.7, random_state=seed, n_jobs=-1, verbose=-1
        )
        model_lgb.fit(X_tr_tree, y_tr, eval_set=[(X_va_tree, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_lgb[val_idx] = model_lgb.predict(X_va_tree)
        test_lgb += model_lgb.predict(X_test) / skf.n_splits
        
        # --- 3. İŞÇİ: XGBOOST (Pseudo-Huber Loss) ---
        model_xgb = xgb.XGBRegressor(
            n_estimators=1500, learning_rate=0.015, max_depth=4, 
            objective='reg:pseudohubererror', # Pseudo Huber
            subsample=0.7, colsample_bytree=0.8, enable_categorical=True, tree_method='hist', random_state=seed, n_jobs=-1
        )
        model_xgb.fit(X_tr_tree, y_tr, eval_set=[(X_va_tree, y_va)], verbose=False)
        oof_xgb[val_idx] = model_xgb.predict(X_va_tree)
        test_xgb += model_xgb.predict(X_test) / skf.n_splits
        
        # --- 4. İŞÇİ: YAPAY SİNİR AĞI (MLP - Doğrusal Olmayan Algı) ---
        model_mlp = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
            learning_rate_init=0.005, max_iter=200, early_stopping=True, validation_fraction=0.1,
            random_state=seed
        )
        model_mlp.fit(X_tr_nn, y_tr)
        oof_mlp[val_idx] = model_mlp.predict(X_va_nn)
        test_mlp += model_mlp.predict(X_test_nn_scaled) / skf.n_splits

    # --- SEVİYE-1 YÖNETİCİ: RIDGE REGRESYON ---
    X_meta_train = np.column_stack((oof_cat, oof_lgb, oof_xgb, oof_mlp))
    X_meta_test = np.column_stack((test_cat, test_lgb, test_xgb, test_mlp))

    meta_model = RidgeCV(alphas=(0.1, 1.0, 10.0))
    meta_model.fit(X_meta_train, y)
    
    nihai_train_tahminleri = np.clip(meta_model.predict(X_meta_train), 0.0, 10.0)
    nihai_test_tahminleri = np.clip(meta_model.predict(X_meta_test), 0.0, 10.0)
    
    seed_rmse = root_mean_squared_error(y, nihai_train_tahminleri)
    print(f"[{i+1}/{len(seed_list)}] Tohum: {seed} | Ridge CV RMSE: {seed_rmse:.5f} | Ağırlıklar: {np.round(meta_model.coef_, 2)}")
    
    seed_skorlari.append(seed_rmse)
    final_test_predictions += nihai_test_tahminleri / len(seed_list)

print(f"\n=======================================================")
print(f"🌟 ULTIMATE GRANDMASTER NİHAİ CV RMSE: {np.mean(seed_skorlari):.5f} 🌟")
print(f"=======================================================")

# Dosyayı Kaydetme
submission_stacking = pd.DataFrame({'id': test_id, hedef_kolon: final_test_predictions})
dosya_adi = 'submission_ULTIMATE_GRANDMASTER_MLP.csv'
submission_stacking.to_csv(dosya_adi, index=False)
print(f"En Gelişmiş Liderlik Dosyanız Hazır: {dosya_adi}")