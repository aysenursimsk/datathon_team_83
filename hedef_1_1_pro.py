import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from catboost import CatBoostRegressor
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

try:
    from magic_moe_stack import build_features as advanced_build_features
except Exception:
    advanced_build_features = None


TRAIN_PATH = "train_temiz.csv"
TEST_PATH = "test_temiz.csv"
EXTERNAL_PATH = "sleep_health_dataset.csv"
TARGET = "bilissel_performans_skoru"
ID_COL = "id"
SEED = 42
N_SPLITS = int(os.getenv("N_SPLITS", "5"))
FAST_MODE = os.getenv("FAST_MODE", "0") == "1"

MESLEK_RANK = {
    "Avukat": 0,
    "Saglik Personeli": 1,
    "Lojistik Calisani": 2,
    "Ogrenci": 3,
    "Yonetici": 4,
    "Satis ve Pazarlama Calisani": 5,
    "Muhendis": 6,
    "Egitimci": 7,
    "Serbest Calisan": 8,
    "Ev Hanimi": 9,
    "Emekli": 10,
    "Bilinmiyor": 5,
}


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def normalize_competition_labels(df):
    df = df.copy()
    mapping = {
        "Spain": "Ispanya",
        "South Korea": "Guney Kore",
        "Sweden": "Isvec",
        "Lawyer": "Avukat",
    }
    for col in ["ulke", "meslek"]:
        if col in df.columns:
            df[col] = df[col].replace(mapping)
    return df


def get_test_id(test_df):
    if ID_COL in test_df.columns:
        return test_df[ID_COL].copy()
    for path in ["test_x.csv", "sample_submission.csv"]:
        if os.path.exists(path):
            temp = pd.read_csv(path)
            if ID_COL in temp.columns and len(temp) == len(test_df):
                return temp[ID_COL].copy()
    return pd.Series(np.arange(len(test_df)), name=ID_COL)


def add_competition_features(train_df, test_df):
    train_df = normalize_competition_labels(train_df.copy())
    test_df = normalize_competition_labels(test_df.copy())

    y = train_df[TARGET].copy().reset_index(drop=True)
    train_x = train_df.drop(columns=[TARGET]).drop(columns=[ID_COL], errors="ignore").reset_index(drop=True)
    test_x = test_df.drop(columns=[ID_COL], errors="ignore").reset_index(drop=True)

    full = pd.concat([train_x, test_x], axis=0, ignore_index=True)

    cat_cols = full.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for col in cat_cols:
        full[col] = full[col].astype(str).fillna("Bilinmiyor")

    num_cols = [c for c in full.columns if c not in cat_cols]
    for col in num_cols:
        full[col] = pd.to_numeric(full[col], errors="coerce")
        full[col] = full[col].fillna(full[col].median())

    eps = 1e-6

    if {"rem_yuzdesi", "derin_uyku_yuzdesi"}.issubset(full.columns):
        full["uyku_kalite_toplam"] = full["rem_yuzdesi"] + full["derin_uyku_yuzdesi"]
        full["uyku_kalite_fark"] = full["rem_yuzdesi"] - full["derin_uyku_yuzdesi"]
        full["uyku_kalite_oran"] = full["rem_yuzdesi"] / (full["derin_uyku_yuzdesi"] + eps)
        full["hafif_uyku_tahmini"] = 100.0 - full["uyku_kalite_toplam"]

    if {"gecelik_uyanma_sayisi", "uykuya_dalma_suresi_dk"}.issubset(full.columns):
        full["uyku_bolunme_yuku"] = full["gecelik_uyanma_sayisi"] * np.log1p(full["uykuya_dalma_suresi_dk"])
        full["uyku_gecikme_cezasi"] = full["uykuya_dalma_suresi_dk"] / (full["gecelik_uyanma_sayisi"] + 1.0)

    if {"stres_skoru", "gunluk_calisma_saati"}.issubset(full.columns):
        full["sinerjik_zihinsel_yuk"] = full["stres_skoru"] * full["gunluk_calisma_saati"]
        full["stres_calisma_oran"] = full["stres_skoru"] / (full["gunluk_calisma_saati"] + 1.0)

    if {"gunluk_adim_sayisi", "stres_skoru"}.issubset(full.columns):
        full["hareket_stres_orani"] = np.log1p(full["gunluk_adim_sayisi"]) / (full["stres_skoru"] + 1.0)

    if {"dinlenik_nabiz_bpm", "stres_skoru"}.issubset(full.columns):
        full["nabiz_stres"] = full["dinlenik_nabiz_bpm"] * full["stres_skoru"]

    if {"vucut_kitle_indeksi", "stres_skoru"}.issubset(full.columns):
        full["bmi_stres"] = full["vucut_kitle_indeksi"] * full["stres_skoru"]
        full["bmi_sapma_22"] = np.abs(full["vucut_kitle_indeksi"] - 22.0)

    if "oda_sicakligi_celsius" in full.columns:
        full["sicaklik_sapma_20"] = np.abs(full["oda_sicakligi_celsius"] - 20.0)

    if "hafta_sonu_uyku_farki_saat" in full.columns:
        full["sosyal_jetlag_abs"] = np.abs(full["hafta_sonu_uyku_farki_saat"])
        full["sosyal_jetlag_sq"] = full["hafta_sonu_uyku_farki_saat"] ** 2

    pair_cols = [
        ("stres_skoru", "rem_yuzdesi"),
        ("stres_skoru", "derin_uyku_yuzdesi"),
        ("stres_skoru", "gecelik_uyanma_sayisi"),
        ("gunluk_calisma_saati", "rem_yuzdesi"),
        ("gunluk_adim_sayisi", "derin_uyku_yuzdesi"),
        ("uykuya_dalma_suresi_dk", "rem_yuzdesi"),
    ]
    for a, b in pair_cols:
        if {a, b}.issubset(full.columns):
            full[f"{a}__{b}_mul"] = full[a] * full[b]
            full[f"{a}__{b}_div"] = full[a] / (np.abs(full[b]) + 1.0)
            full[f"{a}__{b}_sum"] = full[a] + full[b]
            full[f"{a}__{b}_diff"] = full[a] - full[b]

    combo_pairs = [
        ("meslek", "ruh_sagligi_durumu"),
        ("meslek", "kronotip"),
        ("ulke", "meslek"),
        ("cinsiyet", "kronotip"),
        ("mevsim", "gun_tipi"),
    ]
    for a, b in combo_pairs:
        if a in full.columns and b in full.columns:
            combo_name = f"{a}__{b}"
            full[combo_name] = full[a].astype(str) + "__" + full[b].astype(str)

    for col in ["stres_skoru", "rem_yuzdesi", "derin_uyku_yuzdesi", "gunluk_calisma_saati", "gecelik_uyanma_sayisi"]:
        if col in full.columns:
            full[f"{col}_bin8"] = pd.qcut(full[col], q=8, duplicates="drop").astype(str)

    cat_cols = full.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for col in cat_cols:
        freq = full[col].value_counts(normalize=True)
        full[f"{col}_freq"] = full[col].map(freq).astype(float)

    bio_cols = [
        c for c in [
            "stres_skoru", "rem_yuzdesi", "derin_uyku_yuzdesi",
            "gunluk_adim_sayisi", "dinlenik_nabiz_bpm",
            "gunluk_calisma_saati", "uykuya_dalma_suresi_dk",
            "gecelik_uyanma_sayisi"
        ] if c in full.columns
    ]
    bio_scaled = StandardScaler().fit_transform(full[bio_cols])
    for k in [4, 6]:
        kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=20)
        full[f"bio_k{k}"] = kmeans.fit_predict(bio_scaled).astype(str)
        dists = kmeans.transform(bio_scaled)
        for idx in range(k):
            full[f"bio_k{k}_dist_{idx}"] = dists[:, idx]

    group_cols = [c for c in ["meslek", "ruh_sagligi_durumu", "ulke", "bio_k4", "bio_k6"] if c in full.columns]
    stat_cols = [c for c in ["stres_skoru", "rem_yuzdesi", "derin_uyku_yuzdesi", "gunluk_calisma_saati", "gunluk_adim_sayisi", "uykuya_dalma_suresi_dk", "dinlenik_nabiz_bpm"] if c in full.columns]
    for gcol in group_cols:
        grouped = full.groupby(gcol)
        for scol in stat_cols:
            mean_map = grouped[scol].transform("mean")
            full[f"{gcol}_{scol}_mean"] = mean_map
            full[f"{gcol}_{scol}_diff"] = full[scol] - mean_map

    full = full.replace([np.inf, -np.inf], np.nan)
    cat_cols = full.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_cols = [c for c in full.columns if c not in cat_cols]
    for col in num_cols:
        full[col] = pd.to_numeric(full[col], errors="coerce")
        full[col] = full[col].fillna(full[col].median())
    for col in cat_cols:
        full[col] = full[col].astype(str).fillna("Bilinmiyor")

    train_feat = full.iloc[:len(train_x)].reset_index(drop=True)
    test_feat = full.iloc[len(train_x):].reset_index(drop=True)
    return train_feat, test_feat, y


def build_manual_rank_features(raw_df, train_reference_df):
    raw_df = normalize_competition_labels(raw_df.copy())
    train_reference_df = normalize_competition_labels(train_reference_df.copy())

    if TARGET in raw_df.columns:
        raw_df = raw_df.drop(columns=[TARGET])
    if TARGET in train_reference_df.columns:
        train_reference_df = train_reference_df.drop(columns=[TARGET])

    raw_df = raw_df.drop(columns=[ID_COL], errors="ignore")
    train_reference_df = train_reference_df.drop(columns=[ID_COL], errors="ignore")

    ruh_map = {
        "Saglikli": 4,
        "Anksiyete": 3,
        "Depresyon": 2,
        "Anksiyete ve depresyon": 1,
        "Bilinmiyor": 2,
    }
    kronotip_map = {
        "Sabah insani": 2,
        "Notr": 1,
        "Gece insani": 0,
        "Bilinmiyor": 1,
    }

    feat = pd.DataFrame(index=raw_df.index)
    feat["ruh_sagligi_enc"] = raw_df["ruh_sagligi_durumu"].map(ruh_map).fillna(2).astype(float)
    feat["kronotip_enc"] = raw_df["kronotip"].map(kronotip_map).fillna(1).astype(float)
    feat["cinsiyet_enc"] = (raw_df["cinsiyet"] == "Erkek").astype(int)
    feat["hafta_sonu_enc"] = (raw_df["gun_tipi"] == "Hafta sonu").astype(int)
    feat["mevsim_enc"] = (raw_df["mevsim"] == "Ilkbahar-Yaz").astype(int)
    feat["meslek_rank"] = raw_df["meslek"].map(MESLEK_RANK).fillna(5).astype(float)

    le = LabelEncoder()
    train_ulke = train_reference_df["ulke"].astype(str)
    le.fit(train_ulke)
    safe_ulke = raw_df["ulke"].astype(str).where(raw_df["ulke"].astype(str).isin(le.classes_), le.classes_[0])
    feat["ulke_enc"] = le.transform(safe_ulke)

    feat["kafein_log_manual"] = np.log1p(np.clip(raw_df["uyku_oncesi_kafein_mg"], a_min=0, a_max=None))
    feat["uyku_bozulma_indeksi_manual"] = raw_df["gecelik_uyanma_sayisi"] * raw_df["uykuya_dalma_suresi_dk"]
    feat["uyku_verimliligi"] = (
        raw_df["rem_yuzdesi"] + raw_df["derin_uyku_yuzdesi"]
    ) / (feat["uyku_bozulma_indeksi_manual"] + 1.0)

    feat["stres_x_calisma_manual"] = raw_df["stres_skoru"] * raw_df["gunluk_calisma_saati"]
    feat["stres_x_uyanma_manual"] = raw_df["stres_skoru"] * raw_df["gecelik_uyanma_sayisi"]
    feat["stres_x_kafein"] = raw_df["stres_skoru"] * feat["kafein_log_manual"]
    feat["stres_x_ekran"] = raw_df["stres_skoru"] * raw_df["uyku_oncesi_ekran_suresi_dk"]
    feat["gece_uyaricilari"] = feat["kafein_log_manual"] + np.log1p(
        np.clip(raw_df["uyku_oncesi_ekran_suresi_dk"], a_min=0, a_max=None)
    )

    feat["aktif_yasam"] = raw_df["gunluk_adim_sayisi"] / (raw_df["gunluk_calisma_saati"] + 1.0)
    feat["adim_stres_ort"] = raw_df["gunluk_adim_sayisi"] / (raw_df["stres_skoru"] + 0.1)

    feat["bmi_sapma_manual"] = np.abs(raw_df["vucut_kitle_indeksi"] - 22.0)
    feat["bmi_sapma_kare"] = feat["bmi_sapma_manual"] ** 2
    feat["stres_kare_manual"] = raw_df["stres_skoru"] ** 2
    feat["rem_kare_manual"] = raw_df["rem_yuzdesi"] ** 2
    feat["calisma_kare_manual"] = raw_df["gunluk_calisma_saati"] ** 2

    feat["ruh_x_rem"] = feat["ruh_sagligi_enc"] * raw_df["rem_yuzdesi"]
    feat["ruh_x_stres"] = feat["ruh_sagligi_enc"] * raw_df["stres_skoru"]
    feat["kronotip_x_rem"] = feat["kronotip_enc"] * raw_df["rem_yuzdesi"]

    feat["yas_grubu"] = pd.cut(
        raw_df["yas"], bins=[17, 25, 35, 45, 70], labels=[0, 1, 2, 3]
    ).astype(float)

    feat["gun_x_ruh"] = feat["hafta_sonu_enc"] * feat["ruh_sagligi_enc"]
    feat["gun_x_stres"] = feat["hafta_sonu_enc"] * raw_df["stres_skoru"]
    feat["gun_x_rem"] = feat["hafta_sonu_enc"] * raw_df["rem_yuzdesi"]
    feat["gun_x_calisma"] = feat["hafta_sonu_enc"] * raw_df["gunluk_calisma_saati"]

    feat["meslek_x_ruh"] = feat["meslek_rank"] * feat["ruh_sagligi_enc"]
    feat["meslek_x_gun"] = feat["meslek_rank"] * feat["hafta_sonu_enc"]
    feat["meslek_x_stres"] = feat["meslek_rank"] * raw_df["stres_skoru"]

    feat = feat.replace([np.inf, -np.inf], np.nan)
    for col in feat.columns:
        feat[col] = pd.to_numeric(feat[col], errors="coerce")
        feat[col] = feat[col].fillna(feat[col].median())
    return feat


def build_feature_space(train_df, test_df):
    if advanced_build_features is not None:
        try:
            train_feat, test_feat, y, _ = advanced_build_features(train_df.copy(), test_df.copy())
            print("   Feature space: magic_moe_stack.py icindeki gelismis uzay kullanildi.", flush=True)
            manual_train = build_manual_rank_features(train_df.copy(), train_df.copy())
            manual_test = build_manual_rank_features(test_df.copy(), train_df.copy())
            for col in manual_train.columns:
                if col not in train_feat.columns:
                    train_feat[col] = manual_train[col].values
                    test_feat[col] = manual_test[col].values
            return train_feat, test_feat, y
        except Exception as exc:
            print(f"   Uyari: gelismis feature uzayi acilamadi ({exc}). Yerel feature uzayina donuluyor.", flush=True)

    print("   Feature space: hedef_1_1_pro.py icindeki yerel uzay kullanildi.", flush=True)
    train_feat, test_feat, y = add_competition_features(train_df, test_df)
    manual_train = build_manual_rank_features(train_df.copy(), train_df.copy())
    manual_test = build_manual_rank_features(test_df.copy(), train_df.copy())
    for col in manual_train.columns:
        if col not in train_feat.columns:
            train_feat[col] = manual_train[col].values
            test_feat[col] = manual_test[col].values
    return train_feat, test_feat, y


def build_external_aligned_frame():
    ext = pd.read_csv(EXTERNAL_PATH)

    occupation_map = {
        "Driver": "Lojistik Calisani",
        "Software Engineer": "Muhendis",
        "Nurse": "Saglik Personeli",
        "Doctor": "Saglik Personeli",
        "Student": "Ogrenci",
        "Lawyer": "Avukat",
        "Freelancer": "Serbest Calisan",
        "Manager": "Yonetici",
        "Homemaker": "Ev Hanimi",
        "Teacher": "Egitimci",
        "Retired": "Emekli",
        "Sales": "Satis ve Pazarlama Calisani",
    }
    country_map = {
        "USA": "Amerika",
        "Spain": "Ispanya",
        "South Korea": "Guney Kore",
        "France": "Fransa",
        "UK": "Ingiltere",
        "Netherlands": "Hollanda",
        "Argentina": "Arjantin",
        "Sweden": "Isvec",
        "Mexico": "Meksika",
        "China": "Cin",
        "Portugal": "Portekiz",
        # Yarismada olmayan ulkeler en yakin mevcut etikete baglaniyor.
        "Japan": "Yeni Zelanda",
        "India": "Cin",
        "Brazil": "Arjantin",
        "Australia": "Yeni Zelanda",
        "Canada": "Amerika",
        "Germany": "Hollanda",
        "Italy": "Ispanya",
    }
    chronotype_map = {
        "Morning": "Sabah insani",
        "Evening": "Gece insani",
        "Neutral": "Notr",
    }
    mental_map = {
        "Healthy": "Saglikli",
        "Depression": "Depresyon",
        "Anxiety": "Anksiyete",
        "Both": "Anksiyete ve depresyon",
    }
    season_map = {
        "Autumn": "Sonbahar-Kis",
        "Winter": "Sonbahar-Kis",
        "Spring": "Ilkbahar-Yaz",
        "Summer": "Ilkbahar-Yaz",
    }
    day_type_map = {
        "Weekday": "Hafta ici",
        "Weekend": "Hafta sonu",
    }
    gender_map = {
        "Female": "Kadin",
        "Male": "Erkek",
    }

    aligned = pd.DataFrame({
        "yas": ext["age"],
        "cinsiyet": ext["gender"].map(gender_map).fillna("Bilinmiyor"),
        "meslek": ext["occupation"].map(occupation_map).fillna("Bilinmiyor"),
        "vucut_kitle_indeksi": ext["bmi"],
        "ulke": ext["country"].map(country_map).fillna("Bilinmiyor"),
        "rem_yuzdesi": ext["rem_percentage"],
        "derin_uyku_yuzdesi": ext["deep_sleep_percentage"],
        "uykuya_dalma_suresi_dk": ext["sleep_latency_mins"],
        "gecelik_uyanma_sayisi": ext["wake_episodes_per_night"],
        "uyku_oncesi_kafein_mg": ext["caffeine_mg_before_bed"],
        "uyku_oncesi_ekran_suresi_dk": ext["screen_time_before_bed_mins"],
        "gunluk_adim_sayisi": ext["steps_that_day"],
        "sekerleme_suresi_dk": ext["nap_duration_mins"],
        "stres_skoru": ext["stress_score"],
        "gunluk_calisma_saati": ext["work_hours_that_day"],
        "kronotip": ext["chronotype"].map(chronotype_map).fillna("Bilinmiyor"),
        "ruh_sagligi_durumu": ext["mental_health_condition"].map(mental_map).fillna("Bilinmiyor"),
        "dinlenik_nabiz_bpm": ext["heart_rate_resting_bpm"],
        "oda_sicakligi_celsius": ext["room_temperature_celsius"],
        "hafta_sonu_uyku_farki_saat": ext["weekend_sleep_diff_hrs"],
        "mevsim": ext["season"].map(season_map).fillna("Sonbahar-Kis"),
        "gun_tipi": ext["day_type"].map(day_type_map).fillna("Hafta ici"),
        TARGET: ext["cognitive_performance_score"] / 10.0,
    })
    return aligned


def build_domain_similarity(train_features, external_features):
    common_cols = [c for c in train_features.columns if c in external_features.columns]
    train_dom = train_features[common_cols].copy()
    ext_dom = external_features[common_cols].copy()

    domain_x = pd.concat([train_dom, ext_dom], axis=0, ignore_index=True)
    domain_y = np.r_[np.ones(len(train_dom), dtype=int), np.zeros(len(ext_dom), dtype=int)]
    cat_cols = domain_x.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    oof = np.zeros(len(domain_x), dtype=np.float32)
    pred_full = np.zeros(len(domain_x), dtype=np.float32)

    for tr_idx, va_idx in skf.split(domain_x, domain_y):
        model = CatBoostClassifier(
            iterations=600 if FAST_MODE else 900,
            learning_rate=0.03,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            domain_x.iloc[tr_idx],
            domain_y[tr_idx],
            cat_features=cat_cols,
            eval_set=(domain_x.iloc[va_idx], domain_y[va_idx]),
            early_stopping_rounds=80,
        )
        oof[va_idx] = model.predict_proba(domain_x.iloc[va_idx])[:, 1]
        pred_full += model.predict_proba(domain_x)[:, 1] / skf.n_splits

    auc = roc_auc_score(domain_y, oof)
    contest_prob_train = pred_full[:len(train_dom)]
    contest_prob_ext = pred_full[len(train_dom):]
    return common_cols, contest_prob_train, contest_prob_ext, auc


def train_external_proxy_features(train_features, test_features):
    print("2. Dis veri yukleniyor ve proxy feature'lar egitiliyor...", flush=True)
    external_aligned = build_external_aligned_frame()
    external_x, _, external_y = build_feature_space(
        external_aligned,
        external_aligned.drop(columns=[TARGET]).copy(),
    )
    common_cols, contest_prob_train, contest_prob_ext, auc = build_domain_similarity(train_features, external_x)
    print(
        f"   Domain AUC={auc:.6f} | train_prob_mean={contest_prob_train.mean():.4f} | "
        f"external_prob_mean={contest_prob_ext.mean():.4f}",
        flush=True,
    )

    external_x_common = external_x[common_cols].copy()
    train_common = train_features[common_cols].copy()
    test_common = test_features[common_cols].copy()
    proxy_cat_cols = external_x_common.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    proxy_iterations = 900 if FAST_MODE else 1800

    base_proxy_model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=proxy_iterations,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5.0,
        random_strength=0.8,
        bagging_temperature=0.2,
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
    )
    base_proxy_model.fit(external_x_common, external_y, cat_features=proxy_cat_cols)
    train_proxy = base_proxy_model.predict(train_common)
    test_proxy = base_proxy_model.predict(test_common)

    weight_scale = 0.2 + 2.5 * np.clip(contest_prob_ext, 1e-3, 1.0)
    weighted_proxy_model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=proxy_iterations,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5.0,
        random_strength=0.8,
        bagging_temperature=0.2,
        random_seed=SEED + 13,
        verbose=False,
        allow_writing_files=False,
    )
    weighted_proxy_model.fit(
        external_x_common,
        external_y,
        cat_features=proxy_cat_cols,
        sample_weight=weight_scale,
    )
    train_proxy_weighted = weighted_proxy_model.predict(train_common)
    test_proxy_weighted = weighted_proxy_model.predict(test_common)

    print(
        f"   Base proxy std={train_proxy.std():.4f} | Weighted proxy std={train_proxy_weighted.std():.4f}",
        flush=True,
    )

    return {
        "ext_proxy_cog": train_proxy,
        "ext_proxy_cog_weighted": train_proxy_weighted,
        "ext_similarity": contest_prob_train,
    }, {
        "ext_proxy_cog": test_proxy,
        "ext_proxy_cog_weighted": test_proxy_weighted,
        "ext_similarity": np.full(len(test_features), np.nan, dtype=np.float32),
    }, external_x, contest_prob_ext, common_cols


def find_best_blend(y_true, pred_a, pred_b):
    best_weight = 0.5
    best_score = rmse(y_true, 0.5 * pred_a + 0.5 * pred_b)
    for weight_b in np.linspace(0.0, 1.0, 41):
        blended = pred_a * (1.0 - weight_b) + pred_b * weight_b
        score = rmse(y_true, blended)
        if score < best_score:
            best_score = score
            best_weight = weight_b
    return best_weight, best_score


def main():
    print(f"HEDEF 1.1 PRO basliyor | folds={N_SPLITS} | fast={FAST_MODE}", flush=True)
    print("1. Yarisma verileri okunuyor ve zengin feature engineering uygulaniyor...", flush=True)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    test_id = get_test_id(test_df)

    train_features, test_features, y = build_feature_space(train_df, test_df)
    print(f"   Train shape={train_features.shape} | Test shape={test_features.shape}", flush=True)

    train_proxy_dict, test_proxy_dict, external_x, contest_prob_ext, common_cols = train_external_proxy_features(
        train_features,
        test_features,
    )
    for col, values in train_proxy_dict.items():
        train_features[col] = values
    for col, values in test_proxy_dict.items():
        test_features[col] = values

    # Test icin similarity dogrudan hesaplanir ki fold-bagimsiz sabit bir proxy olmasin.
    if np.isnan(test_features["ext_similarity"]).any():
        domain_train = train_features[common_cols].copy()
        domain_test = test_features[common_cols].copy()
        domain_ext = external_x[common_cols].copy()
        domain_all = pd.concat([domain_train, domain_ext], axis=0, ignore_index=True)
        domain_y = np.r_[np.ones(len(domain_train), dtype=int), np.zeros(len(domain_ext), dtype=int)]
        cat_cols_dom = domain_all.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        domain_model = CatBoostClassifier(
            iterations=600 if FAST_MODE else 900,
            learning_rate=0.03,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False,
        )
        domain_model.fit(domain_all, domain_y, cat_features=cat_cols_dom)
        test_features["ext_similarity"] = domain_model.predict_proba(domain_test)[:, 1]

    cat_cols = train_features.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    y_bins = pd.qcut(y, q=12, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof_model_a = np.zeros(len(train_features), dtype=np.float32)
    oof_model_b = np.zeros(len(train_features), dtype=np.float32)
    pred_model_a = np.zeros(len(test_features), dtype=np.float32)
    pred_model_b = np.zeros(len(test_features), dtype=np.float32)

    model_a_iterations = 1400 if FAST_MODE else 3000
    model_b_iterations = 1100 if FAST_MODE else 2200

    print("3. OOF CatBoost blend egitimi basliyor...", flush=True)
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_features, y_bins), start=1):
        X_tr = train_features.iloc[train_idx]
        X_va = train_features.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        y_va = y.iloc[val_idx]

        model_a = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=model_a_iterations,
            learning_rate=0.018,
            depth=6,
            l2_leaf_reg=6.0,
            random_strength=0.8,
            bagging_temperature=0.35,
            random_seed=2026,
            verbose=False,
            allow_writing_files=False,
        )
        model_a.fit(
            X_tr,
            y_tr,
            cat_features=cat_cols,
            eval_set=(X_va, y_va),
            early_stopping_rounds=150,
        )
        val_pred_a = model_a.predict(X_va)
        test_pred_a = model_a.predict(test_features)
        oof_model_a[val_idx] = val_pred_a
        pred_model_a += test_pred_a / N_SPLITS

        model_b = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=model_b_iterations,
            learning_rate=0.022,
            depth=6,
            l2_leaf_reg=6.0,
            random_strength=0.8,
            bagging_temperature=0.35,
            random_seed=SEED + 17,
            verbose=False,
            allow_writing_files=False,
        )
        model_b.fit(
            X_tr,
            y_tr,
            cat_features=cat_cols,
            eval_set=(X_va, y_va),
            early_stopping_rounds=150,
        )
        val_pred_b = model_b.predict(X_va)
        test_pred_b = model_b.predict(test_features)
        oof_model_b[val_idx] = val_pred_b
        pred_model_b += test_pred_b / N_SPLITS

        fold_blend_weight, fold_blend_score = find_best_blend(y_va.values, val_pred_a, val_pred_b)
        print(
            f"   Fold {fold}/{N_SPLITS} | model_a={rmse(y_va, val_pred_a):.5f} | "
            f"model_b={rmse(y_va, val_pred_b):.5f} | best_w_model_b={fold_blend_weight:.2f} | "
            f"blend={fold_blend_score:.5f}",
            flush=True,
        )

    best_weight_deep, best_oof_score = find_best_blend(y.values, oof_model_a, oof_model_b)
    blend_oof = np.clip(oof_model_a * (1.0 - best_weight_deep) + oof_model_b * best_weight_deep, 0.0, 10.0)
    blend_test = np.clip(pred_model_a * (1.0 - best_weight_deep) + pred_model_b * best_weight_deep, 0.0, 10.0)

    smooth_score = rmse(y, oof_model_a)
    deep_score = rmse(y, oof_model_b)
    blend_score = rmse(y, blend_oof)

    meta_train = pd.DataFrame({
        "pred_a": oof_model_a,
        "pred_b": oof_model_b,
        "pred_gap": oof_model_a - oof_model_b,
        "ext_proxy_cog": train_features["ext_proxy_cog"].values,
        "ext_proxy_cog_weighted": train_features["ext_proxy_cog_weighted"].values,
        "ext_similarity": train_features["ext_similarity"].values,
        "meslek_rank": train_features["meslek_rank"].values,
        "ruh_sagligi_enc": train_features["ruh_sagligi_enc"].values,
        "hafta_sonu_enc": train_features["hafta_sonu_enc"].values,
        "mevsim_enc": train_features["mevsim_enc"].values,
        "gun_x_stres": train_features["gun_x_stres"].values,
        "meslek_x_stres": train_features["meslek_x_stres"].values,
        "uyku_verimliligi": train_features["uyku_verimliligi"].values,
    })
    meta_test = pd.DataFrame({
        "pred_a": pred_model_a,
        "pred_b": pred_model_b,
        "pred_gap": pred_model_a - pred_model_b,
        "ext_proxy_cog": test_features["ext_proxy_cog"].values,
        "ext_proxy_cog_weighted": test_features["ext_proxy_cog_weighted"].values,
        "ext_similarity": test_features["ext_similarity"].values,
        "meslek_rank": test_features["meslek_rank"].values,
        "ruh_sagligi_enc": test_features["ruh_sagligi_enc"].values,
        "hafta_sonu_enc": test_features["hafta_sonu_enc"].values,
        "mevsim_enc": test_features["mevsim_enc"].values,
        "gun_x_stres": test_features["gun_x_stres"].values,
        "meslek_x_stres": test_features["meslek_x_stres"].values,
        "uyku_verimliligi": test_features["uyku_verimliligi"].values,
    })

    meta_oof = np.zeros(len(meta_train), dtype=np.float32)
    for train_idx, val_idx in skf.split(meta_train, y_bins):
        meta_model = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-4, 2, 15)),
        )
        meta_model.fit(meta_train.iloc[train_idx], y.iloc[train_idx])
        meta_oof[val_idx] = meta_model.predict(meta_train.iloc[val_idx])

    meta_model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-4, 2, 15)),
    )
    meta_model.fit(meta_train, y)
    meta_test_pred = np.clip(meta_model.predict(meta_test), 0.0, 10.0)
    meta_score = rmse(y, meta_oof)

    if meta_score < blend_score:
        final_oof = np.clip(meta_oof, 0.0, 10.0)
        final_test = meta_test_pred
        final_score = meta_score
        final_mode = "meta_stack"
    else:
        final_oof = blend_oof
        final_test = blend_test
        final_score = blend_score
        final_mode = "weighted_blend"

    print("\n4. Nihai skorlar", flush=True)
    print(f"   Model A RMSE      : {smooth_score:.5f}", flush=True)
    print(f"   Model B RMSE      : {deep_score:.5f}", flush=True)
    print(f"   Blend model_b w   : {best_weight_deep:.2f}", flush=True)
    print(f"   Meta stack RMSE   : {meta_score:.5f}", flush=True)
    print(f"   Secilen mod       : {final_mode}", flush=True)
    print(f"   Final CV RMSE     : {final_score:.5f}", flush=True)

    output_path = "submission_EXTERNAL_PROXY_CATBOOST_BLEND.csv"
    pd.DataFrame({
        ID_COL: test_id,
        TARGET: final_test,
    }).to_csv(output_path, index=False)

    print("\n5. Dosya kaydedildi", flush=True)
    print(f"   {output_path}", flush=True)


if __name__ == "__main__":
    main()
