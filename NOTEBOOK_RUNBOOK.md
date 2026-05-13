# Notebook Runbook

Bu notlar Kaggle/Colab/VSCode notebook'a hizli gecis icin hazirlandi.

## Hedef
- guvenli CV hattinda `1.20` bandina yaklasmak
- mevcut en iyi lokal sonuc:
  `1.20133` 5-fold CV

## Kullanilacak dosyalar
- `train_temiz.csv`
- `test_temiz.csv`
- `sleep_health_dataset.csv`
- `hedef_1_1_pro.py`
- opsiyonel referans:
  `magic_moe_stack.py`

## Notebook hucre plani

### 1. Ortam
```python
import os
import pandas as pd
import numpy as np
```

### 2. Ana fonksiyonlari import et
```python
from hedef_1_1_pro import (
    build_feature_space,
    train_external_proxy,
    find_best_blend,
    TARGET,
)
```

### 3. Veriyi oku
```python
train_df = pd.read_csv("train_temiz.csv")
test_df = pd.read_csv("test_temiz.csv")
```

### 4. Feature space olustur
```python
X_train, X_test, y = build_feature_space(train_df, test_df)
```

Beklenen:
- gelismis feature uzayi
- manuel ordinal/rank feature'lar
- toplam feature sayisi yaklasik `272`

### 5. External proxy ekle
```python
train_proxy, test_proxy = train_external_proxy(X_train.copy(), X_test.copy())
X_train["ext_proxy_cog"] = train_proxy
X_test["ext_proxy_cog"] = test_proxy
```

### 6. CV egitimi
Notebook'ta iki CatBoost rejimini ayri OOF olarak tut:
- smooth:
  `depth=5, l2_leaf_reg=8, random_strength=1.1, bagging_temperature=0.15`
- deep:
  `depth=6, l2_leaf_reg=6, random_strength=0.8, bagging_temperature=0.35`

Blend:
```python
best_weight_deep, best_score = find_best_blend(y.values, oof_smooth, oof_deep)
```

### 7. Submission
```python
submission = pd.DataFrame({
    "id": test_id,
    TARGET: final_test_preds
})
submission.to_csv("submission_EXTERNAL_PROXY_CATBOOST_BLEND.csv", index=False)
```

## Deney gunlugu

### Basarili
- `ext_proxy_cog` eklendiginde buyuk kazanc var
- manuel ordinal/rank feature'lar kucuk ama gercek bir iyilestirme sagliyor
- 2'li CatBoost blend tek modelden daha iyi

### Basarisiz
- `sleep_quality_score`, `sleep_duration_hrs`, `felt_rested`, `sleep_disorder_risk` proxy'lerini ayni anda eklemek
- `oda_sicakligi_celsius`, `hafta_sonu_uyku_farki_saat`, `uyku_oncesi_kafein_mg` kolonlarini topluca drop etmek
- `gun_tipi`, `meslek`, `ruh_sagligi_durumu` lokal uzman modelleri

## Notebook'ta denenebilecek son 3 fikir

### 1. OOF target encoding katmani
- `meslek`
- `gun_tipi`
- `ruh_sagligi_durumu`
- `meslek__ruh_sagligi_durumu`

Bu TE kolonlarini sadece meta feature olarak ekle, ham kategorileri silme.

### 2. Adversarial weight
- external proxy tahminini train/test benzerlik skoruyla carp
- benzer satirlarda proxy agirligini arttir

### 3. Submission blend
- `submission_EXTERNAL_PROXY_CATBOOST_BLEND.csv`
- `submission_SCIPY_OPTIMIZED_1_1X.csv`
- `submission_MAGIC_MOE_STACK.csv`

LB odakli line search ile dosya seviyesinde blend dene.

## Mevcut en iyi komut
```powershell
$env:N_SPLITS='5'
& .\.venv\Scripts\python.exe -u hedef_1_1_pro.py
```
