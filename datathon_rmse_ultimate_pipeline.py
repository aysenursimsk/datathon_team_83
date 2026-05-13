# -*- coding: utf-8 -*-
"""
DATATHON RMSE ULTIMATE PIPELINE
Amaç: En düşük RMSE için güçlü tabular regression pipeline.

Gerekli dosyalar:
- train_temiz.csv
- test_temiz.csv
Opsiyonel:
- test_x.csv veya sample_submission.csv varsa id buradan alınır.

Çıktılar:
- submission_ultimate_superblend.csv
- submission_catboost_only.csv
- submission_lgbm_only.csv
- oof_report.csv

Not:
Leaderboard garantisi yoktur. En iyi sonucu bulmak için üretilen farklı submission'ları
tek tek Kaggle'a yükleyip public score'a göre final blend seçmek en güvenli yoldur.
"""

import os
import gc
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV, ElasticNetCV, BayesianRidge, HuberRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

try:
    from catboost import CatBoostRegressor, Pool
except Exception as e:
    raise ImportError("CatBoost kurulu değil. Kaggle notebook'ta genelde kurulu gelir. Hata: " + str(e))

try:
    import lightgbm as lgb
except Exception as e:
    raise ImportError("LightGBM kurulu değil. Hata: " + str(e))

try:
    import xgboost as xgb
except Exception as e:
    raise ImportError("XGBoost kurulu değil. Hata: " + str(e))

RANDOM_SEEDS = [42, 2026, 777, 1024, 888]
N_SPLITS = 7
TARGET = "bilissel_performans_skoru"
CLIP_MIN, CLIP_MAX = 0.0, 10.0
TRAIN_PATH = "train_temiz.csv"
TEST_PATH = "test_temiz.csv"


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def get_test_id(test_df):
    for p in ["sample_submission.csv", "test_x.csv", "test.csv"]:
        if os.path.exists(p):
            tmp = pd.read_csv(p)
            if "id" in tmp.columns and len(tmp) == len(test_df):
                return tmp["id"].copy()
    if "id" in test_df.columns:
        return test_df["id"].copy()
    return pd.Series(np.arange(len(test_df)), name="id")


def memory_reduce(df):
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def add_features(train_df, test_df):
    """Aynı feature işlemlerini train+test üzerinde uygular. Target kullanılmaz."""
    train_df = train_df.copy()
    test_df = test_df.copy()

    full = pd.concat(
        [train_df.drop(columns=[TARGET], errors="ignore"), test_df],
        axis=0,
        ignore_index=True,
    )

    cat_cols = full.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in full.columns if c not in cat_cols and c != "id"]

    # Eksikleri güvenli doldur
    for c in cat_cols:
        full[c] = full[c].astype(str).fillna("Eksik")
    for c in num_cols:
        full[c] = pd.to_numeric(full[c], errors="coerce")
        full[c] = full[c].fillna(full[c].median())

    eps = 1e-6

    # Temel uyku / yaşam sinyalleri
    full["toplam_kaliteli_uyku_yuzdesi"] = full["rem_yuzdesi"] + full["derin_uyku_yuzdesi"]
    full["hafif_uyku_tahmini"] = 100 - full["rem_yuzdesi"] - full["derin_uyku_yuzdesi"]
    full["rem_derin_orani"] = full["rem_yuzdesi"] / (full["derin_uyku_yuzdesi"] + eps)
    full["derin_rem_orani"] = full["derin_uyku_yuzdesi"] / (full["rem_yuzdesi"] + eps)
    full["uyku_bolunme_yuku"] = full["gecelik_uyanma_sayisi"] * full["uykuya_dalma_suresi_dk"]
    full["uyku_onarim_indeksi"] = full["toplam_kaliteli_uyku_yuzdesi"] / (full["gecelik_uyanma_sayisi"] + 1)
    full["uyku_gecikme_cezasi"] = np.log1p(full["uykuya_dalma_suresi_dk"]) * (full["gecelik_uyanma_sayisi"] + 1)
    full["haftasonu_abs_fark"] = np.abs(full["hafta_sonu_uyku_farki_saat"])

    # Stres / çalışma / ekran / kafein etkileşimleri
    full["sinerjik_zihinsel_yuk"] = full["stres_skoru"] * full["gunluk_calisma_saati"]
    full["stres_ekran"] = full["stres_skoru"] * full["uyku_oncesi_ekran_suresi_dk"]
    full["stres_kafein"] = full["stres_skoru"] * full["uyku_oncesi_kafein_mg"]
    full["kafein_ekran"] = full["uyku_oncesi_kafein_mg"] * full["uyku_oncesi_ekran_suresi_dk"]
    full["calisma_ekran"] = full["gunluk_calisma_saati"] * full["uyku_oncesi_ekran_suresi_dk"]
    full["kafein_log"] = np.log1p(full["uyku_oncesi_kafein_mg"])
    full["ekran_log"] = np.log1p(full["uyku_oncesi_ekran_suresi_dk"])
    full["adim_log"] = np.log1p(full["gunluk_adim_sayisi"])
    full["sekerleme_log"] = np.log1p(full["sekerleme_suresi_dk"])

    # Aktivite / sağlık sinyalleri
    full["hareket_stres_orani"] = full["gunluk_adim_sayisi"] / (full["stres_skoru"] + 1)
    full["adim_calisma_orani"] = full["gunluk_adim_sayisi"] / (full["gunluk_calisma_saati"] + 1)
    full["nabiz_stres"] = full["dinlenik_nabiz_bpm"] * full["stres_skoru"]
    full["bmi_stres"] = full["vucut_kitle_indeksi"] * full["stres_skoru"]
    full["sicaklik_optimum_sapma"] = np.abs(full["oda_sicakligi_celsius"] - 20.5)
    full["yas_stres"] = full["yas"] * full["stres_skoru"]
    full["yas_calisma"] = full["yas"] * full["gunluk_calisma_saati"]

    # Kategorik birleşimler
    if len(cat_cols) > 0:
        combos = [
            ("meslek", "ruh_sagligi_durumu"),
            ("meslek", "kronotip"),
            ("meslek", "gun_tipi"),
            ("ulke", "mevsim"),
            ("cinsiyet", "kronotip"),
            ("ruh_sagligi_durumu", "kronotip"),
            ("mevsim", "gun_tipi"),
        ]
        for a, b in combos:
            if a in full.columns and b in full.columns:
                full[f"{a}__{b}"] = full[a].astype(str) + "__" + full[b].astype(str)

    # Grup istatistikleri: target yok, leakage yok
    base_num = [
        "stres_skoru", "gunluk_calisma_saati", "rem_yuzdesi", "derin_uyku_yuzdesi",
        "uyku_oncesi_ekran_suresi_dk", "gunluk_adim_sayisi", "dinlenik_nabiz_bpm",
        "vucut_kitle_indeksi", "oda_sicakligi_celsius"
    ]
    group_cols = [c for c in ["meslek", "ulke", "ruh_sagligi_durumu", "kronotip", "gun_tipi", "mevsim"] if c in full.columns]
    for g in group_cols:
        for n in base_num:
            if n in full.columns:
                mean_map = full.groupby(g)[n].transform("mean")
                full[f"{g}_{n}_mean"] = mean_map
                full[f"{g}_{n}_diff"] = full[n] - mean_map

    # Basit segment/bin feature'ları
    full["yas_bin"] = pd.cut(full["yas"], bins=[0, 24, 34, 44, 54, 120], labels=False, include_lowest=True).astype(str)
    full["stres_bin"] = pd.cut(full["stres_skoru"], bins=[-1, 3, 5, 7, 8.5, 20], labels=False, include_lowest=True).astype(str)
    full["bmi_bin"] = pd.cut(full["vucut_kitle_indeksi"], bins=[0, 18.5, 25, 30, 35, 80], labels=False, include_lowest=True).astype(str)
    full["ekran_bin"] = pd.cut(full["uyku_oncesi_ekran_suresi_dk"], bins=[-1, 15, 45, 90, 150, 10000], labels=False, include_lowest=True).astype(str)

    # Son split
    train_new = full.iloc[:len(train_df)].copy()
    test_new = full.iloc[len(train_df):].copy()
    train_new[TARGET] = train_df[TARGET].values

    train_new = memory_reduce(train_new)
    test_new = memory_reduce(test_new)
    return train_new, test_new


def make_target_encoding_oof(X, y, X_test, cat_cols, folds, smoothing=20):
    """Leakage-safe target encoding. Sadece train fold ortalaması val'e yazılır."""
    X_te = pd.DataFrame(index=X.index)
    T_te = pd.DataFrame(index=X_test.index)
    global_mean = y.mean()

    use_cols = cat_cols[:]
    # Fazla kolon varsa en etkili birleşimleri de dahil et
    use_cols = [c for c in use_cols if X[c].nunique() <= 200]

    for col in use_cols:
        oof = np.zeros(len(X), dtype=np.float32)
        test_acc = np.zeros(len(X_test), dtype=np.float32)
        for tr_idx, va_idx in folds:
            tr_col = X.iloc[tr_idx][col].astype(str)
            va_col = X.iloc[va_idx][col].astype(str)
            stats = pd.DataFrame({"key": tr_col.values, "target": y.iloc[tr_idx].values}).groupby("key")["target"].agg(["mean", "count"])
            smooth = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
            oof[va_idx] = va_col.map(smooth).fillna(global_mean).values
            test_acc += X_test[col].astype(str).map(smooth).fillna(global_mean).values.astype(np.float32) / len(folds)
        X_te[f"TE_{col}"] = oof
        T_te[f"TE_{col}"] = test_acc
    return X_te, T_te


def prepare_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    test_id = get_test_id(test)
    train = train.drop(columns=["id"], errors="ignore")
    test = test.drop(columns=["id"], errors="ignore")
    train, test = add_features(train, test)
    return train, test, test_id


def run_seed(seed, train, test):
    X = train.drop(columns=[TARGET]).copy()
    y = train[TARGET].copy()
    X_test = test.copy()

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("Eksik")
        X_test[c] = X_test[c].astype(str).fillna("Eksik")

    # Stratified regression fold
    y_bins = pd.qcut(y, q=12, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    folds = list(skf.split(X, y_bins))

    X_te, X_test_te = make_target_encoding_oof(X, y, X_test, cat_cols, folds, smoothing=25)

    X_num = pd.concat([X[num_cols].reset_index(drop=True), X_te.reset_index(drop=True)], axis=1)
    X_test_num = pd.concat([X_test[num_cols].reset_index(drop=True), X_test_te.reset_index(drop=True)], axis=1)

    # NN / sklearn için one-hot + TE + scaler
    X_ohe = pd.get_dummies(X, columns=cat_cols, drop_first=False)
    X_test_ohe = pd.get_dummies(X_test, columns=cat_cols, drop_first=False)
    X_test_ohe = X_test_ohe.reindex(columns=X_ohe.columns, fill_value=0)
    X_nn = pd.concat([X_ohe.reset_index(drop=True), X_te.reset_index(drop=True)], axis=1)
    X_test_nn = pd.concat([X_test_ohe.reset_index(drop=True), X_test_te.reset_index(drop=True)], axis=1)

    model_names = ["cat1", "cat2", "lgb1", "lgb2", "xgb1", "xgb2", "hgb", "extra", "mlp1", "mlp2"]
    oof = {m: np.zeros(len(X), dtype=np.float32) for m in model_names}
    pred = {m: np.zeros(len(X_test), dtype=np.float32) for m in model_names}

    cat_features = cat_cols

    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        print(f"Seed {seed} | Fold {fold}/{N_SPLITS}")
        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        Xn_tr, Xn_va = X_num.iloc[tr_idx], X_num.iloc[va_idx]
        Xnn_tr, Xnn_va = X_nn.iloc[tr_idx], X_nn.iloc[va_idx]

        # CATBOOST 1: RMSE stabil
        cat1 = CatBoostRegressor(
            loss_function="RMSE", eval_metric="RMSE",
            iterations=8000, learning_rate=0.018, depth=6,
            l2_leaf_reg=5.5, random_strength=0.6, bagging_temperature=0.5,
            border_count=254, random_seed=seed + fold,
            allow_writing_files=False, verbose=False
        )
        cat1.fit(X_tr, y_tr, cat_features=cat_features, eval_set=(X_va, y_va), early_stopping_rounds=500, use_best_model=True)
        oof["cat1"][va_idx] = np.clip(cat1.predict(X_va), CLIP_MIN, CLIP_MAX)
        pred["cat1"] += np.clip(cat1.predict(X_test), CLIP_MIN, CLIP_MAX) / N_SPLITS

        # CATBOOST 2: daha derin / farklı regularization
        cat2 = CatBoostRegressor(
            loss_function="RMSE", eval_metric="RMSE",
            iterations=7000, learning_rate=0.015, depth=8,
            l2_leaf_reg=9.0, random_strength=1.2, bagging_temperature=0.8,
            grow_policy="SymmetricTree", border_count=128,
            random_seed=seed * 3 + fold,
            allow_writing_files=False, verbose=False
        )
        cat2.fit(X_tr, y_tr, cat_features=cat_features, eval_set=(X_va, y_va), early_stopping_rounds=500, use_best_model=True)
        oof["cat2"][va_idx] = np.clip(cat2.predict(X_va), CLIP_MIN, CLIP_MAX)
        pred["cat2"] += np.clip(cat2.predict(X_test), CLIP_MIN, CLIP_MAX) / N_SPLITS

        # LIGHTGBM için kategorikleri category yap
        Xl_tr, Xl_va, Xl_test = X_tr.copy(), X_va.copy(), X_test.copy()
        for c in cat_cols:
            Xl_tr[c] = Xl_tr[c].astype("category")
            Xl_va[c] = Xl_va[c].astype("category")
            Xl_test[c] = Xl_test[c].astype("category")

        lgb1 = lgb.LGBMRegressor(
            objective="regression", metric="rmse", n_estimators=10000,
            learning_rate=0.012, num_leaves=31, max_depth=-1,
            min_child_samples=45, subsample=0.80, subsample_freq=1,
            colsample_bytree=0.80, reg_alpha=0.08, reg_lambda=1.7,
            random_state=seed + fold, n_jobs=-1, verbose=-1
        )
        lgb1.fit(Xl_tr, y_tr, eval_set=[(Xl_va, y_va)], categorical_feature=cat_cols,
                 callbacks=[lgb.early_stopping(500, verbose=False)])
        oof["lgb1"][va_idx] = np.clip(lgb1.predict(Xl_va), CLIP_MIN, CLIP_MAX)
        pred["lgb1"] += np.clip(lgb1.predict(Xl_test), CLIP_MIN, CLIP_MAX) / N_SPLITS

        lgb2 = lgb.LGBMRegressor(
            objective="huber", alpha=0.85, metric="rmse", n_estimators=10000,
            learning_rate=0.014, num_leaves=63, max_depth=7,
            min_child_samples=60, subsample=0.75, subsample_freq=1,
            colsample_bytree=0.70, reg_alpha=0.15, reg_lambda=2.5,
            random_state=seed * 5 + fold, n_jobs=-1, verbose=-1
        )
        lgb2.fit(Xl_tr, y_tr, eval_set=[(Xl_va, y_va)], categorical_feature=cat_cols,
                 callbacks=[lgb.early_stopping(500, verbose=False)])
        oof["lgb2"][va_idx] = np.clip(lgb2.predict(Xl_va), CLIP_MIN, CLIP_MAX)
        pred["lgb2"] += np.clip(lgb2.predict(Xl_test), CLIP_MIN, CLIP_MAX) / N_SPLITS

        # XGBoost: kategorikleri kodla, TE numericleri de ekle
        Xx_tr = pd.concat([Xn_tr.reset_index(drop=True), X_tr[cat_cols].reset_index(drop=True).apply(lambda s: pd.factorize(s)[0])], axis=1)
        Xx_va = pd.concat([Xn_va.reset_index(drop=True), X_va[cat_cols].reset_index(drop=True).apply(lambda s: pd.factorize(pd.concat([X_tr[s.name], s], ignore_index=True))[0][-len(s):])], axis=1)
        Xx_test = pd.concat([X_test_num.reset_index(drop=True), X_test[cat_cols].reset_index(drop=True).apply(lambda s: pd.factorize(pd.concat([X_tr[s.name], s], ignore_index=True))[0][-len(s):])], axis=1)
        Xx_tr.columns = Xx_tr.columns.astype(str)
        Xx_va.columns = Xx_va.columns.astype(str)
        Xx_test.columns = Xx_test.columns.astype(str)

        xgb1 = xgb.XGBRegressor(
            objective="reg:squarederror", eval_metric="rmse", n_estimators=7000,
            learning_rate=0.012, max_depth=4, min_child_weight=4,
            subsample=0.82, colsample_bytree=0.82, reg_alpha=0.05, reg_lambda=2.0,
            tree_method="hist", random_state=seed + fold, n_jobs=-1
        )
        xgb1.fit(Xx_tr, y_tr, eval_set=[(Xx_va, y_va)], verbose=False)
        oof["xgb1"][va_idx] = np.clip(xgb1.predict(Xx_va), CLIP_MIN, CLIP_MAX)
        pred["xgb1"] += np.clip(xgb1.predict(Xx_test), CLIP_MIN, CLIP_MAX) / N_SPLITS

        xgb2 = xgb.XGBRegressor(
            objective="reg:pseudohubererror", eval_metric="rmse", n_estimators=5000,
            learning_rate=0.016, max_depth=5, min_child_weight=6,
            subsample=0.75, colsample_bytree=0.75, reg_alpha=0.10, reg_lambda=3.0,
            tree_method="hist", random_state=seed * 7 + fold, n_jobs=-1
        )
        xgb2.fit(Xx_tr, y_tr, eval_set=[(Xx_va, y_va)], verbose=False)
        oof["xgb2"][va_idx] = np.clip(xgb2.predict(Xx_va), CLIP_MIN, CLIP_MAX)
        pred["xgb2"] += np.clip(xgb2.predict(Xx_test), CLIP_MIN, CLIP_MAX) / N_SPLITS

        # Sklearn modelleri numeric/onehot alır
        hgb = HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.035, max_iter=900,
            max_leaf_nodes=31, l2_regularization=0.08,
            random_state=seed + fold, early_stopping=True
        )
        hgb.fit(Xn_tr, y_tr)
        oof["hgb"][va_idx] = np.clip(hgb.predict(Xn_va), CLIP_MIN, CLIP_MAX)
        pred["hgb"] += np.clip(hgb.predict(X_test_num), CLIP_MIN, CLIP_MAX) / N_SPLITS

        extra = ExtraTreesRegressor(
            n_estimators=450, max_depth=None, min_samples_leaf=6,
            max_features=0.75, random_state=seed + fold, n_jobs=-1
        )
        extra.fit(Xn_tr, y_tr)
        oof["extra"][va_idx] = np.clip(extra.predict(Xn_va), CLIP_MIN, CLIP_MAX)
        pred["extra"] += np.clip(extra.predict(X_test_num), CLIP_MIN, CLIP_MAX) / N_SPLITS

        mlp1 = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(256, 128, 64), activation="relu", solver="adam",
                alpha=0.002, learning_rate_init=0.0012, batch_size=512,
                max_iter=450, early_stopping=True, validation_fraction=0.12,
                n_iter_no_change=25, random_state=seed + fold
            )
        )
        mlp1.fit(Xnn_tr, y_tr)
        oof["mlp1"][va_idx] = np.clip(mlp1.predict(Xnn_va), CLIP_MIN, CLIP_MAX)
        pred["mlp1"] += np.clip(mlp1.predict(X_test_nn), CLIP_MIN, CLIP_MAX) / N_SPLITS

        mlp2 = make_pipeline(
            SimpleImputer(strategy="median"),
            RobustScaler(),
            MLPRegressor(
                hidden_layer_sizes=(128, 128, 64, 32), activation="relu", solver="adam",
                alpha=0.006, learning_rate_init=0.0009, batch_size=512,
                max_iter=500, early_stopping=True, validation_fraction=0.12,
                n_iter_no_change=25, random_state=seed * 11 + fold
            )
        )
        mlp2.fit(Xnn_tr, y_tr)
        oof["mlp2"][va_idx] = np.clip(mlp2.predict(Xnn_va), CLIP_MIN, CLIP_MAX)
        pred["mlp2"] += np.clip(mlp2.predict(X_test_nn), CLIP_MIN, CLIP_MAX) / N_SPLITS

        fold_scores = {m: rmse(y_va, oof[m][va_idx]) for m in model_names}
        print("  fold rmse:", {k: round(v, 5) for k, v in fold_scores.items()})

        del cat1, cat2, lgb1, lgb2, xgb1, xgb2, hgb, extra, mlp1, mlp2
        gc.collect()

    # Meta model: OOF üstünden gerçek stacking
    meta_train = np.column_stack([oof[m] for m in model_names])
    meta_test = np.column_stack([pred[m] for m in model_names])

    meta_models = {
        "ridge": RidgeCV(alphas=np.logspace(-4, 4, 50)),
        "elastic": ElasticNetCV(l1_ratio=[0.05, 0.15, 0.3, 0.5, 0.7], alphas=np.logspace(-4, 1, 35), cv=5, max_iter=20000, random_state=seed),
        "bayes": BayesianRidge(),
        "huber": HuberRegressor(alpha=0.0005, epsilon=1.25, max_iter=1000),
    }

    meta_oof = {}
    meta_pred = {}
    for name, model in meta_models.items():
        model.fit(meta_train, y)
        meta_oof[name] = np.clip(model.predict(meta_train), CLIP_MIN, CLIP_MAX)
        meta_pred[name] = np.clip(model.predict(meta_test), CLIP_MIN, CLIP_MAX)

    # Elle blend + meta blend: LB stabilitesi için
    manual_weights = {
        "cat1": 0.22, "cat2": 0.17,
        "lgb1": 0.16, "lgb2": 0.12,
        "xgb1": 0.10, "xgb2": 0.08,
        "hgb": 0.04, "extra": 0.03,
        "mlp1": 0.05, "mlp2": 0.03,
    }
    manual_oof = np.zeros(len(X))
    manual_pred = np.zeros(len(X_test))
    for m, w in manual_weights.items():
        manual_oof += oof[m] * w
        manual_pred += pred[m] * w
    manual_oof = np.clip(manual_oof, CLIP_MIN, CLIP_MAX)
    manual_pred = np.clip(manual_pred, CLIP_MIN, CLIP_MAX)

    final_oof = np.clip(
        0.38 * meta_oof["ridge"] +
        0.22 * meta_oof["bayes"] +
        0.15 * meta_oof["elastic"] +
        0.10 * meta_oof["huber"] +
        0.15 * manual_oof,
        CLIP_MIN, CLIP_MAX
    )
    final_pred = np.clip(
        0.38 * meta_pred["ridge"] +
        0.22 * meta_pred["bayes"] +
        0.15 * meta_pred["elastic"] +
        0.10 * meta_pred["huber"] +
        0.15 * manual_pred,
        CLIP_MIN, CLIP_MAX
    )

    report = {m: rmse(y, oof[m]) for m in model_names}
    report.update({f"meta_{k}": rmse(y, v) for k, v in meta_oof.items()})
    report["manual_blend"] = rmse(y, manual_oof)
    report["final_blend"] = rmse(y, final_oof)
    print(f"\nSeed {seed} FINAL CV RMSE: {report['final_blend']:.6f}")
    print(json.dumps({k: round(v, 6) for k, v in sorted(report.items(), key=lambda x: x[1])}, indent=2, ensure_ascii=False))

    return oof, pred, final_oof, final_pred, report


def main():
    print("\n🌟 DATATHON RMSE ULTIMATE SUPERBLEND BAŞLIYOR 🌟\n")
    train, test, test_id = prepare_data()
    y = train[TARGET].values

    all_final_preds = []
    all_final_oofs = []
    all_reports = []
    all_model_preds = {}
    all_model_oofs = {}

    for seed in RANDOM_SEEDS:
        oof, pred, final_oof, final_pred, report = run_seed(seed, train, test)
        all_final_oofs.append(final_oof)
        all_final_preds.append(final_pred)
        all_reports.append({"seed": seed, **report})
        for k, v in pred.items():
            all_model_preds.setdefault(k, []).append(v)
        for k, v in oof.items():
            all_model_oofs.setdefault(k, []).append(v)

    # Seed ortalaması
    final_oof_mean = np.mean(all_final_oofs, axis=0)
    final_pred_mean = np.mean(all_final_preds, axis=0)

    # Rank blend: public/private stabilitesi için opsiyonel ek harman
    model_pred_mean = {k: np.mean(v, axis=0) for k, v in all_model_preds.items()}
    model_oof_mean = {k: np.mean(v, axis=0) for k, v in all_model_oofs.items()}
    conservative_pred = np.clip(
        0.35 * model_pred_mean["cat1"] + 0.22 * model_pred_mean["cat2"] +
        0.18 * model_pred_mean["lgb1"] + 0.12 * model_pred_mean["lgb2"] +
        0.08 * model_pred_mean["xgb1"] + 0.05 * model_pred_mean["mlp1"],
        CLIP_MIN, CLIP_MAX
    )
    conservative_oof = np.clip(
        0.35 * model_oof_mean["cat1"] + 0.22 * model_oof_mean["cat2"] +
        0.18 * model_oof_mean["lgb1"] + 0.12 * model_oof_mean["lgb2"] +
        0.08 * model_oof_mean["xgb1"] + 0.05 * model_oof_mean["mlp1"],
        CLIP_MIN, CLIP_MAX
    )

    ultra_oof = np.clip(0.72 * final_oof_mean + 0.28 * conservative_oof, CLIP_MIN, CLIP_MAX)
    ultra_pred = np.clip(0.72 * final_pred_mean + 0.28 * conservative_pred, CLIP_MIN, CLIP_MAX)

    print("\n================ GENEL CV ================")
    print(f"Seed mean final CV:     {rmse(y, final_oof_mean):.6f}")
    print(f"Conservative blend CV:  {rmse(y, conservative_oof):.6f}")
    print(f"ULTRA final CV:         {rmse(y, ultra_oof):.6f}")
    print("==========================================\n")

    # Submission dosyaları
    pd.DataFrame({"id": test_id, TARGET: ultra_pred}).to_csv("submission_ultimate_superblend.csv", index=False)
    pd.DataFrame({"id": test_id, TARGET: final_pred_mean}).to_csv("submission_meta_seed_mean.csv", index=False)
    pd.DataFrame({"id": test_id, TARGET: conservative_pred}).to_csv("submission_conservative_tree_blend.csv", index=False)
    pd.DataFrame({"id": test_id, TARGET: model_pred_mean["cat1"]}).to_csv("submission_catboost_only.csv", index=False)
    pd.DataFrame({"id": test_id, TARGET: model_pred_mean["lgb1"]}).to_csv("submission_lgbm_only.csv", index=False)

    report_df = pd.DataFrame(all_reports)
    report_df.to_csv("oof_report.csv", index=False)

    # OOF tahminlerini sakla: sonraki blend denemeleri için altın değerinde
    oof_dump = pd.DataFrame({"target": y, "ultra_oof": ultra_oof, "final_oof_mean": final_oof_mean, "conservative_oof": conservative_oof})
    for k, v in model_oof_mean.items():
        oof_dump[k] = v
    oof_dump.to_csv("oof_predictions.csv", index=False)

    print("Hazır dosyalar:")
    print("1) submission_ultimate_superblend.csv  <-- ilk dene")
    print("2) submission_meta_seed_mean.csv")
    print("3) submission_conservative_tree_blend.csv")
    print("4) submission_catboost_only.csv")
    print("5) submission_lgbm_only.csv")
    print("6) oof_report.csv / oof_predictions.csv")


if __name__ == "__main__":
    main()
