import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def veri_okuma_ve_grup_istatistikleri():
    print("1. Veriler Okunuyor...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test_x.csv')
    
    # Standartlaştırma Adımları
    ulke_mapping = {'Spain': 'Ispanya', 'South Korea': 'Guney Kore', 'Sweden': 'Isvec'}
    train_df['ulke'] = train_df['ulke'].replace(ulke_mapping)
    test_df['ulke'] = test_df['ulke'].replace(ulke_mapping)
    
    train_df['meslek'] = train_df['meslek'].replace({'Lawyer': 'Avukat'})
    test_df['meslek'] = test_df['meslek'].replace({'Lawyer': 'Avukat'})
    
    print("2. Makro-Bağlam (Grup İstatistikleri) Hesaplanıyor...")
    # Veri sızıntısını önlemek için ortalamalar SADECE Train'den hesaplanır
    
    # Mesleğe göre ortalama stres ve çalışma saati
    meslek_stres_map = train_df.groupby('meslek')['stres_skoru'].mean().to_dict()
    meslek_mesai_map = train_df.groupby('meslek')['gunluk_calisma_saati'].mean().to_dict()
    
    # Ülkeye göre ortalama uyku onarımı (uyku evreleri üzerinden)
    train_df['gecici_uyku_kalitesi'] = (train_df['rem_yuzdesi'] + train_df['derin_uyku_yuzdesi']) / (train_df['gecelik_uyanma_sayisi'] + 1)
    ulke_uyku_map = train_df.groupby('ulke')['gecici_uyku_kalitesi'].mean().to_dict()
    train_df = train_df.drop(columns=['gecici_uyku_kalitesi'])

    # Haritalamaları hem Train hem Test setine uygulama
    for df in [train_df, test_df]:
        df['meslek_stres_ortalamasi'] = df['meslek'].map(meslek_stres_map)
        df['meslek_mesai_ortalamasi'] = df['meslek'].map(meslek_mesai_map)
        df['ulke_uyku_kalitesi_ortalamasi'] = df['ulke'].map(ulke_uyku_map)
        
        # Bireyin kendi meslektaşlarından stres sapması (Fark ne kadar büyükse, risk o kadar yüksek)
        df['meslektastan_stres_sapmasi'] = df['stres_skoru'] - df['meslek_stres_ortalamasi']

    print("İşlem Tamam! Yeni özellikler veriye başarıyla entegre edildi.")
    return train_df, test_df

# Sadece test etmek için çalıştırıyoruz
train, test = veri_okuma_ve_grup_istatistikleri()
print("Eğitim seti boyutu:", train.shape)
print("Yeni eklenen sütunlar:", [col for col in train.columns if 'ortalamasi' in col or 'sapmasi' in col])