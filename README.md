# YZTA 2026 Datathon | Team 83

Bu repo, YZTA 2026 Datathon boyunca Team 83 tarafından geliştirilen tüm ana fikirleri, deney hatlarını, yarışma boyunca evrilen model mimarilerini ve özellikle yarışmanın kırılma noktasını oluşturan dış veri tabanlı retrieval yaklaşımını belgelemek için hazırlandı.

Bu çalışma klasik bir tabular regression yarışması olarak başlamadı. Çok erken aşamada verinin sentetik olduğunu, dolayısıyla “gerçek dünya mantığına uygun feature engineering” kadar, hatta ondan daha fazla, veriyi üreten gizli mekanizmanın tersine mühendisliğine ihtiyaç duyduğunu gördük. Bu repo o yolculuğun tamamını anlatır:

- ilk güvenli CatBoost bazları
- overfit eden ama leaderboard’da patlayan sentetik feature saldırıları
- stacking, blending ve pseudo-labeling denemeleri
- external proxy yaklaşımı
- en sonunda yarışmayı gerçek anlamda kıran external nearest-neighbor retrieval hattı

## Hızlı Bakış

- Problem tipi: sentetik tabular regresyon
- Hedef değişken: `bilissel_performans_skoru`
- Ana metrik: `RMSE`
- Güvenli ağaç tabanlı baz: yaklaşık `1.22`
- Güçlü supervised / proxy hattı: yaklaşık `1.20`
- Yarışmayı kıran final yaklaşım: `external 1-NN retrieval + hafif Ridge kalibrasyonu`
- En dikkat çekici public leaderboard seviyesi: yaklaşık `0.15592`

Bu repo yalnızca final kodu değil, karar verme sürecini de saklar. Yani burada hem başarılı yöntemler hem de bilinçli olarak elenen yollar korunmuştur.

## Skor Özeti

| Aşama | Ana yaklaşım | Yaklaşık skor | Yorum |
| --- | --- | ---: | --- |
| Güvenli baz | CatBoost + temel feature engineering | `1.22` bandı | CV ve LB en tutarlı başlangıç çizgisi |
| Gelişmiş supervised hat | Cluster feature'lar + CatBoost ağırlıklı blend | `1.20` bandı | Ek modeller sınırlı yeni sinyal üretti |
| External proxy dönemi | `sleep_health_dataset.csv` üzerinden proxy feature üretimi | `1.20` altına yaklaşan güvenli iyileşme | Dış verinin gerçekten değerli olduğu burada netleşti |
| Final kırılma noktası | One-hot + Manhattan `1-NN` external retrieval | train tarafında çok düşük hata | Generator mantığını yakalama ihtimali çok yükseldi |
| Final submission | `1-NN + Ridge` kalibrasyonu | public LB yaklaşık `0.15592` | Yarışmayı asıl sıçratan çözüm |

## English Abstract

This repository documents Team 83's full modeling journey for the YZTA 2026 Datathon. The competition started as a synthetic tabular regression problem, but our final breakthrough did not come from standard tree ensembles alone. After strong CatBoost baselines, multiple stacking attempts, and external proxy features, the decisive gain came from aligning the provided external sleep dataset with the competition feature space and transferring scores through nearest-neighbor retrieval. In short, the project evolved from conventional supervised learning into a retrieval-driven reverse-engineering pipeline for the hidden synthetic data generator.

## 1. Yarışma Özeti

### Problem tanımı

Amaç, sentetik biyometrik ve yaşam tarzı değişkenlerinden `bilissel_performans_skoru` değerini tahmin etmekti.

Hedef değişken:

- `bilissel_performans_skoru`

Görev tipi:

- Regresyon

Skor metriği:

- RMSE

Veri büyüklüğü:

- Eğitim verisi: yaklaşık `56.000` satır
- Test verisi: yaklaşık `24.000` satır

### Kritik yarışma gerçeği

Bu veri seti gerçek ölçüm verisi değil, sentetik olarak üretilmiş bir veri setiydi. Bu yüzden:

- “stres artarsa uyku bozulur” gibi doğal dünya ilişkileri yardımcı oldu ama tek başına yeterli olmadı
- asıl başarı, üretici formüle yaklaşan temsiller kurmakla geldi
- leaderboard başarısı ile lokal CV başarısı arasında büyük fark oluşabildi

## 2. En Önemli Sonuçlar

Yarışma boyunca birkaç ayrı kırılma noktası yaşadık.

### Erken dönem güvenli baz

- CatBoost + makul feature engineering ile yaklaşık `1.22` bandı
- bu skor güvenilir ve leaderboard ile tutarlıydı

### Orta dönem gelişmiş ama sınırlı hat

- cluster feature’lar
- CatBoost / LightGBM / XGBoost / ExtraTrees harmanları
- ağırlık optimizasyonu
- yaklaşık `1.20` bandı

Bu aşamada şunu fark ettik:

- optimizer sürekli CatBoost’a çok yüksek ağırlık veriyordu
- diğer modeller yeterince yeni sinyal üretmiyordu

### Kırılma noktası

Harici `sleep_health_dataset.csv` dosyasını yarışma kolonlarına hizaladık ve önce proxy learning, daha sonra da doğrudan external retrieval tabanlı tahmin mekanizmasına geçtik.

Bu, yarışmanın karakterini tamamen değiştirdi.

### En güçlü bulgu

External veri ile:

- yarışma satırlarının feature uzayında en yakın harici örneğini bulmak
- bu örneğin bilişsel skorunu taşımak
- gerektiğinde çok hafif kalibrasyon yapmak

yaklaşımı klasik boosting mimarilerinin çok ötesine geçti.

Bu repo içindeki en kritik final artefact’lar:

- `submission_BEST_LOCAL_CV_120058.csv`
- `submission_BEST_LB_SAFE_BLEND.csv`
- `submission_FINAL_BEST_CHANCE_1NN_RIDGE.csv`
- `submission_BACKUP_1NN_PURE.csv`

Yarışma sırasında alınan en dikkat çekici public leaderboard sonucu:

- yaklaşık `0.15592`

Bu skor, artık sadece “iyi genelleyen bir tree ensemble” değil, büyük olasılıkla veri üretim kaynağını gerçekten yakalayan bir çözüm hattı elde ettiğimizi gösterdi.

## 3. Stratejik Yol Haritası

Yarışmadaki tüm çalışma kabaca beş faza ayrıldı.

### Faz 1 | Güvenli taban kurma

Bu fazda hedef:

- sağlam CV
- leaderboard ile uyumlu skor
- kategori + sayısal etkileşimlerin güvenli biçimde kullanılması

Ana script aileleri:

- `guvenli.py`
- `ultra_model_egitimi.py`
- `datathon_rmse_ultimate_pipeline.py`
- `datathon_rmse_final_attack.py`

Öğrenim:

- düşük variance’lı CatBoost bazları değerliydi
- bazı “çok iyi görünen” lokal feature’lar aslında leaderboard’da çöküyordu

### Faz 2 | Sentetik formül arayışı

Bu fazda hedef:

- veriyi üreten gizli denklemi açığa çıkarabilecek oranlar ve etkileşimler kurmak

Bu dönemde çok sayıda script yazıldı:

- `formul_kirici.py`
- `hedef0.50.py`
- `final_050_operasyonu.py`
- `operasyon_022_*`
- `randmaster_sihirli.py`

Öğrenim:

- aşırı keskin oranlar lokal CV’de çok iyi görünse bile gerçek testte kırılabiliyor
- tek başına “magic ratio” yaklaşımı sürdürülebilir olmadı

### Faz 3 | Stacking ve blend dönemi

Bu fazda hedef:

- CatBoost’un yakaladığı sinyale ek olarak başka modellerden tamamlayıcı sinyal almak

Denenen modeller:

- CatBoost
- LightGBM
- XGBoost
- ExtraTrees
- MLP
- Ridge meta model

İlgili dosyalar:

- `grandmaster_stacking.py`
- `titan_grandmaster.py`
- `stacking_nihai.py`
- `ultimate_grandmaster.py`
- `submission_blend_lab.py`

Öğrenim:

- farklı model isimleri kullanmak çeşitlilik üretmek anlamına gelmiyor
- aynı feature uzayında benzer inductive bias’lar çoğu zaman yine CatBoost’un gölgesinde kalıyor

### Faz 4 | External proxy dönemi

Bu fazda yarışma dışı kaynak olan `sleep_health_dataset.csv` kullanılmaya başlandı.

Yaklaşım:

1. External veri yarışma kolonlarına map edildi
2. External veri üzerinde bilişsel skor tahmini öğrenildi
3. Bu bilgi yarışma veri setine proxy feature olarak taşındı

Ana dosyalar:

- `hedef_1_1_pro.py`
- `domain_proxy_experiments.py`
- `catboost_variant_search_v2.py`
- `catboost_param_search.py`
- `magic_moe_stack.py`

Öğrenim:

- `ext_proxy_cog` tek başına ciddi fayda sağladı
- external veriyi yarışma dağılımına göre ağırlıklandırmak ek kazanç verdi
- manual ordinal/rank feature’lar küçük ama gerçek katkı sundu

### Faz 5 | External retrieval dönemi

Bu repo içindeki en önemli metodolojik sıçrama budur.

Temel fikir:

- external veri yarışmanın gerçek kaynağına çok yakın olabilir
- bu durumda boosting ile global fonksiyon öğrenmek yerine
- feature space içinde “en yakın external örneği” bularak doğrudan skor transfer etmek

Denenen retrieval türevleri:

- label encoded + scaled KNN
- one-hot + Manhattan distance KNN
- `1-NN` saf retrieval
- `1-NN + Ridge` hafif kalibrasyon

Ana script:

- `external_1nn_retrieval.py`

En çarpıcı sonuç:

- train üzerinde one-hot + Manhattan `1-NN` yaklaşımı son derece düşük hata verdi
- train/test nearest-distance dağılımlarının neredeyse aynı çıkması, bu hattın testte de çökme riskini ciddi biçimde düşürdü

Buradan üretilen nihai dosyalar:

- `submission_EXTERNAL_1NN_PURE.csv`
- `submission_EXTERNAL_1NN_RIDGE.csv`
- `submission_FINAL_BEST_CHANCE_1NN_RIDGE.csv`
- `submission_BACKUP_1NN_PURE.csv`

## 4. Repo Yapısı

Bu repo bilerek “çok temizlenmiş tek final kod” mantığıyla değil, yarışma yolculuğunu taşıyacak şekilde düzenlendi.

### Hızlı repo haritası

```text
.
├── README.md
├── requirements.txt
├── sleep_health_dataset.csv
├── hedef_1_1_pro.py
├── external_1nn_retrieval.py
├── submission_blend_lab.py
├── aggressive_submission_lab.py
├── submission_*.csv
├── submission_blends/
├── submission_aggressive_lab/
├── data/
└── deney / analiz scriptleri
```

Bu yapı özellikle iki şeyi görünür kılmak için korundu:

- nihai olarak işe yarayan ana hatlar
- o sonuca giderken denenmiş alternatif araştırma patikaları

### Ana çalışma dosyaları

- `hedef_1_1_pro.py`
  Yarışma sonuna doğru oluşan en güçlü supervised pipeline. External proxy, weighted proxy, manual rank features ve CatBoost blend içerir.

- `external_1nn_retrieval.py`
  Yarışmanın son aşamasında oyunu değiştiren external nearest-neighbor retrieval hattı. One-hot + Manhattan `1-NN` retrieval ve Ridge kalibrasyonunu içerir.

- `magic_moe_stack.py`
  Geniş feature engineering, target encoding ve MoE benzeri fikirlerin toplandığı güçlü deney hattı.

- `domain_proxy_experiments.py`
  External proxy ve domain similarity yaklaşımını ölçmek için kullanıldı.

- `catboost_variant_search_v2.py`
  Güçlü feature/proxy uzayı sabitken CatBoost varyant taraması yapmak için yazıldı.

- `aggressive_submission_lab.py`
  Daha agresif pseudo-label ve stage-2 blend submission’ları üretir.

- `submission_blend_lab.py`
  Dosya seviyesinde submission blend laboratuvarı.

### Analiz ve deney dosyaları

- `adversarial_validation.py`
- `formul_kirici.py`
- `pseudo_labeling.py`
- `pseudo_grandmaster.py`
- `grandmaster_stacking.py`
- `titan_grandmaster.py`
- `ultimate_grandmaster.py`
- ve diğer deney scriptleri

Bu dosyaların hepsi üretim kalitesinde nihai kod olmak zorunda değildir. Bir kısmı yarışma içinde fikir test etmek için hızlı şekilde yazılmıştır. Bu repo açısından değerleri şuradadır:

- hangi fikirlerin denendiğini görünür kılmak
- başarısız patikaları da kayıt altına almak
- model geliştirme sürecini tam hikâyesiyle belgelemek

### Submission klasörleri

- `submission_blends/`
  daha güvenli blend denemeleri

- `submission_aggressive_lab/`
  daha agresif pseudo-label ve stage-2 submission’lar

### Dokümanlar

- `NOTEBOOK_FEATURE_GAP_ANALYSIS.md`
- `NOTEBOOK_RUNBOOK.md`

Bu iki dosya notebook tarafında hızlı deney yürütmek için oluşturuldu.

## 5. Veri Politikası

Bu repo yarışma kodunu ve deney çıktılarının önemli bölümünü içerir. Ancak yarışmaya ait özel veri dosyaları varsayılan olarak repoya dahil edilmemelidir.

Özellikle aşağıdaki dosyalar yarışma verisi olduğu için bilinçli olarak dışarıda bırakılmalıdır:

- `train.csv`
- `train_temiz.csv`
- `train_ULTRA_temiz.csv`
- `test_x.csv`
- `test_temiz.csv`
- `test_ULTRA_temiz.csv`

Buna karşılık aşağıdaki external kaynak bu çalışmanın merkezindedir ve public/paylaşılabilir nitelikte kullanılmıştır:

- `sleep_health_dataset.csv`

Repo clone edildikten sonra yarışma kodlarını yeniden çalıştırmak isteyen bir kullanıcı, yukarıdaki competition dosyalarını kendi ortamına yerleştirmelidir.

## 6. Kurulum

### Python sürümü

Önerilen:

- Python `3.11+`

### Temel bağımlılıklar

Bu repo için ana bağımlılıklar:

- `pandas`
- `numpy`
- `scikit-learn`
- `catboost`
- `lightgbm`
- `xgboost`
- `scipy`

Kurulum:

```bash
pip install -r requirements.txt
```

## 7. Kullanım

### Ana supervised pipeline

```bash
python hedef_1_1_pro.py
```

Bu script:

- yarışma feature space’ini kurar
- external proxy feature’larını üretir
- CatBoost varyantlarını eğitir
- blend eder
- submission üretir

### Final retrieval pipeline

```bash
python external_1nn_retrieval.py
```

Bu script:

- yarışma feature space’ini external veri ile hizalar
- one-hot + Manhattan nearest-neighbor retrieval kurar
- `1-NN` saf transfer tahmini üretir
- Ridge ile hafif kalibrasyon yapar
- final submission dosyalarını yazar

### Blend laboratuvarı

```bash
python submission_blend_lab.py
```

### Agresif submission denemeleri

```bash
python aggressive_submission_lab.py
```

### Domain / proxy araştırmaları

```bash
python domain_proxy_experiments.py
```

## 8. En Önemli Teknik Öğrenimler

### 1. Sentetik yarışmalarda güvenli CV şart, ama yeterli değil

Başta `1.22` bandında güvenli bir temel kurmak çok değerliydi. Çünkü leaderboard ile lokal skor arasındaki farkı bu sayede sürekli takip edebildik.

### 2. Çok iyi görünen lokal feature’lar kolayca overfit olabilir

Özellikle agresif ratio feature’lar ve bazı sentetik etkileşimler lokal CV’de aşırı iyi görünüp gerçek leaderboard’da başarısız oldu.

### 3. Model çeşitliliği isimle değil sinyalle ölçülür

CatBoost + LightGBM + XGBoost yazmak otomatik olarak güçlü ensemble yaratmıyor. Eğer aynı feature space içinde benzer ilişkileri öğreniyorlarsa optimizer yine tek modele kaçıyor.

### 4. Dış veri sadece yardımcı feature değil, bazen gerçek anahtar olabilir

Bu yarışmada external data başlangıçta bir yardımcı bilgi kaynağı gibi görünse de, finalde asıl çözüm external retrieval yaklaşımı oldu.

### 5. Retrieval bazı sentetik yarışmalarda fonksiyon öğrenmekten daha güçlü olabilir

Eğer test verisi generator mantığı bakımından external source ile aynı aileden geliyorsa, nearest-neighbor retrieval çok yüksek değer üretir.

## 9. Hangi Denemeler İşe Yaramadı

Repo’nun önemli değerlerinden biri yalnızca işe yarayanları değil, işe yaramayanları da kaydetmesidir.

Başarısız ya da sınırlı kalan yönler:

- aşırı keskin magic ratio feature’lar
- local cluster expert modelleri
- pseudo-label’in agresif sürümleri
- bazı stage-2 blend’lerin public LB’de geriye düşmesi
- meta stack’in son aşamada weighted blend’i geçememesi

Bu denemelerin başarısız olması boşa gitmedi. Her biri final yöntemi daraltmamıza yardımcı oldu.

## 10. Nihai Değer

Bu repo yalnızca “yarışma sonunda çalışan script” deposu değildir.

Bu repo:

- sentetik tabular yarışmalarda nasıl düşünülmesi gerektiğini
- güvenli model geliştirme ile agresif leaderboard arayışı arasındaki dengeyi
- external data leverage etmenin farklı biçimlerini
- retrieval tabanlı çözümün nasıl keşfedildiğini
- çok sayıda denemenin nasıl sistematik biçimde daraltıldığını

gösteren bir mühendislik günlüğüdür.

Bir başka deyişle, bu çalışma yalnızca skor üretmedi; aynı zamanda problem çözme sürecini görünür, öğretilebilir ve tekrar incelenebilir hale getirdi.

## 11. Son Not

Yarışma boyunca ortaya çıkan en değerli şeylerden biri şuydu:

başarı tek bir “harika model” kurmakla değil, çok sayıda hipotezi disiplinli biçimde test edip doğru anda paradigma değiştirebilmekle geldi.

Team 83 için bu repo tam olarak bunu temsil ediyor.
