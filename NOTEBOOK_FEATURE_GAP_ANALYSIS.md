# Notebook Feature Gap Analysis

Amaç: `1.20` altına inmek için kullanici tarafindan onerilen feature bloklarinin mevcut pipeline ile farkini netlestirmek.

## Veri kaynaklari
- `train_temiz.csv`
- `test_temiz.csv`
- `sleep_health_dataset.csv`

## Mevcut pipeline'da zaten olanlar
- `kafein_log` benzeri:
  `uyku_oncesi_kafein_mg_log1p`
- kare/polinom benzeri:
  `stres_skoru_sq`, `gunluk_calisma_saati_sq`
- uyku toplam/oran:
  `uyku_kalite_toplam`, `uyku_kalite_oran`
- stres-calisma etkilesimi:
  `sinerjik_zihinsel_yuk`
- stres-uyanma etkilesimi:
  `stres_skoru__gecelik_uyanma_sayisi_mul`
- kategori kombinasyonlari:
  `meslek__ruh_sagligi_durumu`, `meslek__kronotip`, `mevsim__gun_tipi`
- grup baglami:
  `meslek_*_mean`, `ruh_sagligi_durumu_*_mean`, `bio_k*_*`
- unsupervised profil:
  `bio_k4`, `bio_k6`, uzaklik kolonlari

## Kullanici blogundan eksik olup eklenenler
- ordinal/rank:
  `ruh_sagligi_enc`, `kronotip_enc`, `cinsiyet_enc`, `hafta_sonu_enc`, `mevsim_enc`, `meslek_rank`, `ulke_enc`
- davranissal lineer etkilesimler:
  `stres_x_kafein`, `stres_x_ekran`, `aktif_yasam`, `adim_stres_ort`
- manuel baglamsal feature'lar:
  `uyku_verimliligi`, `gece_uyaricilari`, `bmi_sapma_kare`, `yas_grubu`
- gun tipi etkilesimleri:
  `gun_x_ruh`, `gun_x_stres`, `gun_x_rem`, `gun_x_calisma`
- meslek etkilesimleri:
  `meslek_x_ruh`, `meslek_x_gun`, `meslek_x_stres`
- ruh sagligi etkilesimleri:
  `ruh_x_rem`, `ruh_x_stres`, `kronotip_x_rem`

## Test sonucu
- `base_proxy`:
  gelismis feature uzayi + `ext_proxy_cog`
  3-fold tek model RMSE: `1.20401`
- `base_proxy_plus_manual`:
  yukaridaki yapi + manuel rank/ordinal feature'lar
  3-fold tek model RMSE: `1.20365`
- `plus_manual_drop_noise`:
  manuel feature'lar + `oda_sicakligi_celsius`, `hafta_sonu_uyku_farki_saat`, `uyku_oncesi_kafein_mg` drop
  3-fold tek model RMSE: `1.20408`

Sonuc:
- manuel rank/ordinal katmani faydali
- kullanicinin onerdiği "gurultu kolonlarini drop et" adimi bizim pipeline'da faydali cikmadi
- lokal uzman modeller (`gun_tipi`, `meslek`, `ruh_sagligi_durumu`) global CatBoost'tan daha kotu cikti

## Dis veri transferi sonucu
- `sleep_health_dataset.csv` ile egitilen `ext_proxy_cog` acik ara en guclu yeni sinyal
- ek proxy'ler denendi:
  `sleep_quality_score`, `sleep_duration_hrs`, `felt_rested`, `sleep_disorder_risk`
- bu ek proxy'ler birlikte faydadan cok gurultu ekledi
- bu yuzden sadece `ext_proxy_cog` tutuldu

## En iyi mevcut hat
- dosya:
  `hedef_1_1_pro.py`
- yapi:
  gelismis feature uzayi + manuel rank/ordinal feature'lar + `ext_proxy_cog` + 2'li CatBoost blend
- 5-fold CV RMSE:
  `1.20133`

## Sonraki mantikli denemeler
- OOF target encoding ile `meslek_rank`, `gun_tipi`, `ruh_sagligi_enc` uzerinden yeni metafeature katmani
- `sleep_health_dataset.csv` icin domain adaptation:
  external proxy'yi train/test benzerlik skoruyla agirliklandirmak
- submission-level blend:
  `submission_EXTERNAL_PROXY_CATBOOST_BLEND.csv` ile eski guclu submission'larin LB bazli blend testi
