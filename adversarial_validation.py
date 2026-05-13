import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("🕵️‍♂️ ÇEKİŞMELİ DOĞRULAMA (ADVERSARIAL VALIDATION) BAŞLADI 🕵️‍♂️\n")

# 1. Verileri Oku
train_df = pd.read_csv('train_temiz.csv')
test_df = pd.read_csv('test_temiz.csv')

# Hedef değişkeni ve ID'yi Train'den ayır
if 'bilissel_performans_skoru' in train_df.columns:
    train_df = train_df.drop(columns=['bilissel_performans_skoru'])
if 'id' in train_df.columns:
    train_df = train_df.drop(columns=['id'])
if 'id' in test_df.columns:
    test_df = test_df.drop(columns=['id'])

# 2. Sahte Hedef Değişken Yarat (Train = 0, Test = 1)
train_df['is_test'] = 0
test_df['is_test'] = 1

# Verileri Alt Alta Birleştir
dev_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

y_adv = dev_df['is_test']
X_adv = dev_df.drop(columns=['is_test'])

# Kategorik değişkenleri ayarla
kategorik_kolonlar = X_adv.select_dtypes(include=['object', 'category']).columns.tolist()
for col in kategorik_kolonlar:
    X_adv[col] = X_adv[col].astype('category')

# 3. Dedektif Modeli Eğit (LightGBM Classifier)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_skorlari = []
feature_importances = np.zeros(X_adv.shape[1])

print("Test Setini Eğitim Setinden Ayırt Etmeye Çalışıyoruz...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_adv, y_adv)):
    X_tr, y_tr = X_adv.iloc[train_idx], y_adv.iloc[train_idx]
    X_va, y_va = X_adv.iloc[val_idx], y_adv.iloc[val_idx]
    
    model = lgb.LGBMClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=5, 
        random_state=42, n_jobs=-1, verbose=-1
    )
    
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(20, verbose=False)])
    
    preds = model.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, preds)
    auc_skorlari.append(auc)
    
    feature_importances += model.feature_importances_ / skf.n_splits

ortalama_auc = np.mean(auc_skorlari)

print("\n==================================================")
print(f"🚨 ROC AUC SKORU: {ortalama_auc:.4f} 🚨")
print("==================================================")

if ortalama_auc > 0.60:
    print("\n❌ TEHLİKE! Model Train ve Test setlerini kolayca ayırt edebiliyor.")
    print("Bu durum, Liderlik Tablosunda çuvallamanızın asıl sebebidir (Dağılım Kayması / Data Drift).")
    print("\nModeli 'Kopya Çekmeye' İten O Zehirli Sütunlar Şunlar:")
    
    # En çok ele veren sütunları yazdır
    importance_df = pd.DataFrame({'Sutun': X_adv.columns, 'Onem': feature_importances})
    importance_df = importance_df.sort_values(by='Onem', ascending=False).head(5)
    print(importance_df)
    print("\nÇÖZÜM: Bu sütunları eğitimden çıkarmalı veya yapılarını değiştirmeliyiz!")
    
elif ortalama_auc <= 0.60 and ortalama_auc > 0.52:
    print("\n⚠️ UYARI! Hafif bir dağılım kayması var ama ölümcül değil.")
else:
    print("\n✅ KUSURSUZ! Train ve Test setlerinizin yapısı tamamen aynı.")
    print("Liderlik Tablosunda aldığınız skor, yerel CV'nize çok yakın olacaktır.")