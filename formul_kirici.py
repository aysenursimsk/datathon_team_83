import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

print("🧮 SİNTETİK VERİ FORMÜL KIRICI (REVERSE ENGINEERING) BAŞLATILDI 🧮\n")

# 1. Temiz Verileri Oku
train = pd.read_csv('train_temiz.csv')
test = pd.read_csv('test_temiz.csv')

y = train['bilissel_performans_skoru']
X = train.drop(columns=['bilissel_performans_skoru'])
X_test = test.copy()

# ID'leri ayıkla
if 'id' in X.columns: X = X.drop(columns=['id'])
if 'id' in X_test.columns: X_test = X_test.drop(columns=['id'])

# 2. Sadece Sayısal Verileri Al (Formüller rakamlarla yazılır)
sayisal_kolonlar = X.select_dtypes(exclude=['object', 'category']).columns
X_num = X[sayisal_kolonlar]
X_test_num = X_test[sayisal_kolonlar]

# 3. Organizatörler Çarpım Formülü Kullandı mı? (Polinomsal Genişletme)
print("1. İhtimaller Hesaplanıyor (Tüm Sütunların Birbiriyle Çarpımı)...")
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_num)
sutun_isimleri = poly.get_feature_names_out(sayisal_kolonlar)

# Standartlaştırma (Katsayıları adil görmek için)
scaler = StandardScaler()
X_poly_scaled = scaler.fit_transform(X_poly)

# 4. LASSO İLE GİZLİ DENKLEMİ ÇÖZME
print("2. Lasso Modeli ile Gizli Denklem Çözülüyor (Gereksizler Sıfırlanıyor)...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# LassoCV otomatik olarak en iyi alpha'yı (ceza katsayısını) bulur
lasso = LassoCV(cv=kf, random_state=42, n_jobs=-1, max_iter=2000)
lasso.fit(X_poly_scaled, y)

# RMSE Skorunu Hesapla
lasso_rmse = np.sqrt(np.mean((lasso.predict(X_poly_scaled) - y) ** 2))

print(f"\n==================================================")
print(f"🎯 FORMÜL KIRICI (LASSO) EĞİTİM RMSE: {lasso_rmse:.5f}")
print(f"==================================================")

if lasso_rmse < 1.15:
    print("\n🚨 BİNGÖ! ORGANİZATÖRLERİN GİZLİ DENKLEMİNİ KIRDIK! 🚨")
    print("Team 136'nın 1.08 skorunu nasıl aldığını şimdi anlıyoruz. Olay ağaç algoritmaları değilmiş!")
else:
    print("\n-> Hata 1.15'in altına inmedi. Demek ki formül sadece basit çarpımlardan oluşmuyor, işin içinde ciddi gürültü var.")

# 5. GERÇEK FORMÜLÜN İSKELETİ
print("\n--- ORGANİZATÖRLERİN YAZDIĞI MUHTEMEL DENKLEM (EN BÜYÜK KATSAYILAR) ---")
katsayilar = pd.DataFrame({'Degisken': sutun_isimleri, 'Katsayi': lasso.coef_})
katsayilar['Mutlak_Katsayi'] = katsayilar['Katsayi'].abs()
# Etkisi 0 olmayan en güçlü katsayıları getir
gercek_formul = katsayilar[katsayilar['Mutlak_Katsayi'] > 0.05].sort_values(by='Mutlak_Katsayi', ascending=False)

for index, row in gercek_formul.head(10).iterrows():
    isaret = "+" if row['Katsayi'] > 0 else ""
    print(f"{isaret}{row['Katsayi']:.4f} * ({row['Degisken']})")