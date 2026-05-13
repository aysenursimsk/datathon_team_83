import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


TRAIN_PATH = "train_temiz.csv"
TEST_PATH = "test_temiz.csv"
TARGET = "bilissel_performans_skoru"
ID_COL = "id"
N_SPLITS = int(os.getenv("N_SPLITS", "5"))
SEED = 42
CLIP_MIN, CLIP_MAX = 0.0, 10.0
FAST_MODE = os.getenv("FAST_MODE", "0") == "1"


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def forward_target(y):
    y = np.asarray(y, dtype=float)
    p = np.clip(y / 10.0, 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def inverse_target(z):
    z = np.asarray(z, dtype=float)
    p = 1.0 / (1.0 + np.exp(-z))
    return np.clip(p * 10.0, CLIP_MIN, CLIP_MAX)


def get_test_id(test_df):
    if ID_COL in test_df.columns:
        return test_df[ID_COL].copy()
    for candidate in ["test_x.csv", "sample_submission.csv"]:
        path = Path(candidate)
        if path.exists():
            tmp = pd.read_csv(path)
            if ID_COL in tmp.columns and len(tmp) == len(test_df):
                return tmp[ID_COL].copy()
    return pd.Series(np.arange(len(test_df)), name=ID_COL)


def normalize_labels(df):
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


def build_features(train_df, test_df):
    train_df = normalize_labels(train_df.copy())
    test_df = normalize_labels(test_df.copy())

    y = train_df[TARGET].copy().reset_index(drop=True)
    train_x = train_df.drop(columns=[TARGET]).copy()
    test_id = get_test_id(test_df)
    train_x = train_x.drop(columns=[ID_COL], errors="ignore")
    test_x = test_df.drop(columns=[ID_COL], errors="ignore")

    full = pd.concat([train_x, test_x], axis=0, ignore_index=True)

    cat_cols = full.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for col in cat_cols:
        full[col] = full[col].astype(str).fillna("Eksik")

    num_cols = [c for c in full.columns if c not in cat_cols]
    for col in num_cols:
        full[col] = pd.to_numeric(full[col], errors="coerce")
        full[col] = full[col].fillna(full[col].median())

    eps = 1e-6

    # Smooth numeric transformations
    for col in ["stres_skoru", "gunluk_calisma_saati", "gunluk_adim_sayisi", "uyku_oncesi_kafein_mg",
                "uyku_oncesi_ekran_suresi_dk", "sekerleme_suresi_dk", "yas", "dinlenik_nabiz_bpm"]:
        if col in full.columns:
            full[f"{col}_log1p"] = np.log1p(np.clip(full[col], a_min=0, a_max=None))
            full[f"{col}_sq"] = full[col] ** 2

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

    # Symbolic pair features for smoother models
    pair_cols = [
        ("stres_skoru", "rem_yuzdesi"),
        ("stres_skoru", "derin_uyku_yuzdesi"),
        ("stres_skoru", "gunluk_adim_sayisi"),
        ("stres_skoru", "gecelik_uyanma_sayisi"),
        ("gunluk_calisma_saati", "rem_yuzdesi"),
        ("gunluk_calisma_saati", "derin_uyku_yuzdesi"),
        ("uykuya_dalma_suresi_dk", "rem_yuzdesi"),
        ("gunluk_adim_sayisi", "derin_uyku_yuzdesi"),
    ]
    for a, b in pair_cols:
        if {a, b}.issubset(full.columns):
            full[f"{a}__{b}_mul"] = full[a] * full[b]
            full[f"{a}__{b}_div"] = full[a] / (np.abs(full[b]) + 1.0)
            full[f"{a}__{b}_sum"] = full[a] + full[b]
            full[f"{a}__{b}_diff"] = full[a] - full[b]

    # Quantile bins as categories
    for col in ["stres_skoru", "rem_yuzdesi", "derin_uyku_yuzdesi", "gunluk_calisma_saati",
                "gecelik_uyanma_sayisi", "uykuya_dalma_suresi_dk", "gunluk_adim_sayisi", "yas"]:
        if col in full.columns:
            try:
                full[f"{col}_bin8"] = pd.qcut(full[col], q=8, duplicates="drop").astype(str)
            except ValueError:
                full[f"{col}_bin8"] = pd.cut(full[col], bins=8, duplicates="drop").astype(str)

    # Frequency encodings for raw categories
    cat_cols = full.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for col in cat_cols:
        freq = full[col].value_counts(normalize=True)
        full[f"{col}_freq"] = full[col].map(freq).astype(float)

    # Interaction categories
    combo_pairs = [
        ("meslek", "ruh_sagligi_durumu"),
        ("meslek", "kronotip"),
        ("ulke", "meslek"),
        ("cinsiyet", "kronotip"),
        ("mevsim", "gun_tipi"),
    ]
    for a, b in combo_pairs:
        if a in full.columns and b in full.columns:
            name = f"{a}__{b}"
            full[name] = full[a].astype(str) + "__" + full[b].astype(str)
            freq = full[name].value_counts(normalize=True)
            full[f"{name}_freq"] = full[name].map(freq).astype(float)

    # Unsupervised profile features
    bio_cols = [
        c for c in [
            "stres_skoru", "rem_yuzdesi", "derin_uyku_yuzdesi",
            "gunluk_adim_sayisi", "dinlenik_nabiz_bpm", "gunluk_calisma_saati",
            "uykuya_dalma_suresi_dk", "gecelik_uyanma_sayisi"
        ] if c in full.columns
    ]
    bio_scaled = StandardScaler().fit_transform(full[bio_cols])
    for k in (4, 6):
        km = KMeans(n_clusters=k, random_state=SEED, n_init=20)
        full[f"bio_k{k}"] = km.fit_predict(bio_scaled).astype(str)
        dists = km.transform(bio_scaled)
        for idx in range(k):
            full[f"bio_k{k}_dist_{idx}"] = dists[:, idx]

    # Group-relative features
    group_cols = [c for c in ["meslek", "ulke", "ruh_sagligi_durumu", "kronotip", "bio_k6"] if c in full.columns]
    stat_cols = [c for c in ["stres_skoru", "rem_yuzdesi", "derin_uyku_yuzdesi", "gunluk_calisma_saati",
                             "gunluk_adim_sayisi", "uykuya_dalma_suresi_dk", "dinlenik_nabiz_bpm"] if c in full.columns]
    for gcol in group_cols:
        grouped = full.groupby(gcol)
        for scol in stat_cols:
            mean_map = grouped[scol].transform("mean")
            std_map = grouped[scol].transform("std").fillna(0.0)
            full[f"{gcol}_{scol}_mean"] = mean_map
            full[f"{gcol}_{scol}_diff"] = full[scol] - mean_map
            full[f"{gcol}_{scol}_z"] = (full[scol] - mean_map) / (std_map + 1.0)

    # Row-wise standardized summary stats
    row_stat_cols = [c for c in ["stres_skoru", "rem_yuzdesi", "derin_uyku_yuzdesi", "gunluk_calisma_saati",
                                 "gunluk_adim_sayisi", "uykuya_dalma_suresi_dk", "gecelik_uyanma_sayisi",
                                 "dinlenik_nabiz_bpm", "vucut_kitle_indeksi", "oda_sicakligi_celsius"] if c in full.columns]
    zmat = full[row_stat_cols].copy()
    zmat = (zmat - zmat.mean()) / (zmat.std() + 1e-6)
    full["row_z_mean"] = zmat.mean(axis=1)
    full["row_z_std"] = zmat.std(axis=1)
    full["row_z_max"] = zmat.max(axis=1)
    full["row_z_min"] = zmat.min(axis=1)

    full = full.replace([np.inf, -np.inf], np.nan)
    cat_cols = full.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_cols = [c for c in full.columns if c not in cat_cols]
    for col in num_cols:
        full[col] = pd.to_numeric(full[col], errors="coerce")
        full[col] = full[col].fillna(full[col].median())
    for col in cat_cols:
        full[col] = full[col].astype(str).fillna("Eksik")

    train_x = full.iloc[:len(train_x)].reset_index(drop=True)
    test_x = full.iloc[len(train_x):].reset_index(drop=True)

    return train_x, test_x, y, test_id


def build_target_encoding(train_df, test_df, y, folds, te_cols, smoothing=25.0):
    global_mean = float(y.mean())
    global_std = float(y.std())

    train_te = pd.DataFrame(index=train_df.index)
    test_te = pd.DataFrame(index=test_df.index)

    for col in te_cols:
        oof_mean = np.zeros(len(train_df), dtype=np.float32)
        oof_std = np.zeros(len(train_df), dtype=np.float32)
        oof_count = np.zeros(len(train_df), dtype=np.float32)

        test_mean_acc = np.zeros(len(test_df), dtype=np.float32)
        test_std_acc = np.zeros(len(test_df), dtype=np.float32)
        test_count_acc = np.zeros(len(test_df), dtype=np.float32)

        for tr_idx, va_idx in folds:
            tr_x = train_df.iloc[tr_idx][col].astype(str)
            va_x = train_df.iloc[va_idx][col].astype(str)
            stats = (
                pd.DataFrame({"key": tr_x.values, "target": y.iloc[tr_idx].values})
                .groupby("key")["target"]
                .agg(["mean", "std", "count"])
            )

            smooth_mean = (
                stats["mean"] * stats["count"] + global_mean * smoothing
            ) / (stats["count"] + smoothing)
            smooth_std = stats["std"].fillna(global_std)
            smooth_count = np.log1p(stats["count"])

            oof_mean[va_idx] = va_x.map(smooth_mean).fillna(global_mean).values
            oof_std[va_idx] = va_x.map(smooth_std).fillna(global_std).values
            oof_count[va_idx] = va_x.map(smooth_count).fillna(0.0).values

            test_key = test_df[col].astype(str)
            test_mean_acc += test_key.map(smooth_mean).fillna(global_mean).values.astype(np.float32) / len(folds)
            test_std_acc += test_key.map(smooth_std).fillna(global_std).values.astype(np.float32) / len(folds)
            test_count_acc += test_key.map(smooth_count).fillna(0.0).values.astype(np.float32) / len(folds)

        train_te[f"{col}_te_mean"] = oof_mean
        train_te[f"{col}_te_std"] = oof_std
        train_te[f"{col}_te_count"] = oof_count

        test_te[f"{col}_te_mean"] = test_mean_acc
        test_te[f"{col}_te_std"] = test_std_acc
        test_te[f"{col}_te_count"] = test_count_acc

    return train_te, test_te


def main():
    print(f"Magic MoE Stack basliyor... | folds={N_SPLITS} | fast={FAST_MODE}", flush=True)
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_x, test_x, y, test_id = build_features(train_df, test_df)

    cat_cols = train_x.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_cols = [c for c in train_x.columns if c not in cat_cols]

    y_bins = pd.qcut(y, q=12, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(skf.split(train_x, y_bins))

    te_cols = [c for c in cat_cols if train_x[c].nunique() <= 32]
    train_te, test_te = build_target_encoding(train_x, test_x, y, folds, te_cols)

    train_num = pd.concat([train_x[num_cols].reset_index(drop=True), train_te.reset_index(drop=True)], axis=1)
    test_num = pd.concat([test_x[num_cols].reset_index(drop=True), test_te.reset_index(drop=True)], axis=1)

    train_num = train_num.replace([np.inf, -np.inf], np.nan).fillna(train_num.median())
    test_num = test_num.replace([np.inf, -np.inf], np.nan).fillna(train_num.median())

    cat_iterations = 2200 if FAST_MODE else 4000
    cat_logit_iterations = 2600 if FAST_MODE else 4500
    local_iterations = 900 if FAST_MODE else 2500
    rbf_components = 160 if FAST_MODE else 384

    cat_models = {
        "cat_raw": CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=cat_iterations,
            learning_rate=0.02,
            depth=6,
            l2_leaf_reg=6.0,
            random_strength=0.8,
            bagging_temperature=0.35,
            border_count=254,
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False,
        ),
        "cat_logit": CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=cat_logit_iterations,
            learning_rate=0.018,
            depth=5,
            l2_leaf_reg=8.0,
            random_strength=1.1,
            bagging_temperature=0.15,
            border_count=254,
            random_seed=SEED + 13,
            verbose=False,
            allow_writing_files=False,
        ),
    }

    local_expert_template = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=local_iterations,
        learning_rate=0.025,
        depth=5,
        l2_leaf_reg=5.0,
        random_strength=0.6,
        bagging_temperature=0.2,
        border_count=254,
        random_seed=SEED + 29,
        verbose=False,
        allow_writing_files=False,
    )

    elastic_model = make_pipeline(
        RobustScaler(),
        RidgeCV(alphas=np.logspace(-4, 2, 15)),
    )

    rbf_model = make_pipeline(
        StandardScaler(),
        RBFSampler(gamma=0.03, n_components=rbf_components, random_state=SEED),
        RidgeCV(alphas=np.logspace(-2, 2, 9)),
    )

    model_names = ["cat_raw", "cat_logit", "cat_local", "elastic", "rbf"]
    oof = {name: np.zeros(len(train_x), dtype=np.float32) for name in model_names}
    pred = {name: np.zeros(len(test_x), dtype=np.float32) for name in model_names}

    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        print(f"\nFold {fold}/{N_SPLITS}", flush=True)

        X_tr_cat = train_x.iloc[tr_idx].copy()
        X_va_cat = train_x.iloc[va_idx].copy()
        X_test_cat = test_x.copy()

        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        X_tr_num = train_num.iloc[tr_idx].copy()
        X_va_num = train_num.iloc[va_idx].copy()
        X_test_num = test_num.copy()

        cat_raw_model = clone(cat_models["cat_raw"])
        cat_raw_model.fit(X_tr_cat, y_tr, cat_features=cat_cols, eval_set=(X_va_cat, y_va), early_stopping_rounds=200)
        va_cat_raw = cat_raw_model.predict(X_va_cat)
        te_cat_raw = cat_raw_model.predict(X_test_cat)
        oof["cat_raw"][va_idx] = va_cat_raw
        pred["cat_raw"] += te_cat_raw / N_SPLITS
        print(f"  cat_raw   : {rmse(y_va, va_cat_raw):.5f}", flush=True)

        cat_logit_model = clone(cat_models["cat_logit"])
        y_tr_logit = forward_target(y_tr.values)
        y_va_logit = forward_target(y_va.values)
        cat_logit_model.fit(X_tr_cat, y_tr_logit, cat_features=cat_cols, eval_set=(X_va_cat, y_va_logit), early_stopping_rounds=200)
        va_cat_logit = inverse_target(cat_logit_model.predict(X_va_cat))
        te_cat_logit = inverse_target(cat_logit_model.predict(X_test_cat))
        oof["cat_logit"][va_idx] = va_cat_logit
        pred["cat_logit"] += te_cat_logit / N_SPLITS
        print(f"  cat_logit : {rmse(y_va, va_cat_logit):.5f}", flush=True)

        # Cluster-specific experts with global fallback
        va_local = va_cat_raw.copy()
        te_local = te_cat_raw.copy()
        cluster_col = "bio_k6"
        for cluster_value in X_tr_cat[cluster_col].astype(str).unique():
            tr_mask = X_tr_cat[cluster_col].astype(str) == cluster_value
            va_mask = X_va_cat[cluster_col].astype(str) == cluster_value
            te_mask = X_test_cat[cluster_col].astype(str) == cluster_value

            if tr_mask.sum() < 500 or va_mask.sum() == 0:
                continue

            local_model = clone(local_expert_template)
            local_model.fit(
                X_tr_cat.loc[tr_mask],
                y_tr.loc[tr_mask],
                cat_features=cat_cols,
                eval_set=(X_va_cat.loc[va_mask], y_va.loc[va_mask]),
                early_stopping_rounds=120,
            )
            va_local[va_mask.values] = local_model.predict(X_va_cat.loc[va_mask])
            if te_mask.sum() > 0:
                te_local[te_mask.values] = local_model.predict(X_test_cat.loc[te_mask])

        oof["cat_local"][va_idx] = va_local
        pred["cat_local"] += te_local / N_SPLITS
        print(f"  cat_local : {rmse(y_va, va_local):.5f}", flush=True)

        elastic = clone(elastic_model)
        elastic.fit(X_tr_num, y_tr)
        va_elastic = np.clip(elastic.predict(X_va_num), CLIP_MIN, CLIP_MAX)
        te_elastic = np.clip(elastic.predict(X_test_num), CLIP_MIN, CLIP_MAX)
        oof["elastic"][va_idx] = va_elastic
        pred["elastic"] += te_elastic / N_SPLITS
        print(f"  elastic   : {rmse(y_va, va_elastic):.5f}", flush=True)

        rbf = clone(rbf_model)
        rbf.fit(X_tr_num, y_tr)
        va_rbf = np.clip(rbf.predict(X_va_num), CLIP_MIN, CLIP_MAX)
        te_rbf = np.clip(rbf.predict(X_test_num), CLIP_MIN, CLIP_MAX)
        oof["rbf"][va_idx] = va_rbf
        pred["rbf"] += te_rbf / N_SPLITS
        print(f"  rbf       : {rmse(y_va, va_rbf):.5f}", flush=True)

    print("\nTekil model OOF skorlar", flush=True)
    for name in model_names:
        print(f"{name:10s}: {rmse(y, oof[name]):.5f}", flush=True)

    meta_train = np.column_stack([oof[name] for name in model_names])
    meta_test = np.column_stack([pred[name] for name in model_names])

    meta = RidgeCV(alphas=np.logspace(-4, 2, 15))
    meta.fit(meta_train, y)
    final_oof = np.clip(meta.predict(meta_train), CLIP_MIN, CLIP_MAX)
    final_test = np.clip(meta.predict(meta_test), CLIP_MIN, CLIP_MAX)
    final_score = rmse(y, final_oof)

    print("\nMeta agirliklari", flush=True)
    for name, weight in zip(model_names, meta.coef_):
        print(f"{name:10s}: {weight:.6f}", flush=True)
    print(f"\nFinal CV RMSE: {final_score:.5f}", flush=True)

    submission = pd.DataFrame({ID_COL: test_id, TARGET: final_test})
    out_path = Path("submission_MAGIC_MOE_STACK.csv")
    submission.to_csv(out_path, index=False)
    print(f"Kaydedildi: {out_path}", flush=True)


if __name__ == "__main__":
    main()
