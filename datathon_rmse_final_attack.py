# -*- coding: utf-8 -*-
"""
DATATHON RMSE FINAL ATTACK PIPELINE
VSCode klasörüne koyulacak dosyalar:
    train_temiz.csv
    test_temiz.csv
Opsiyonel:
    submission_ULTIMATE_GRANDMASTER_MLP.csv veya önceki iyi submission dosyaların

Çalıştırma:
    python datathon_rmse_final_attack.py

Not: RMSE için düşük skor iyidir. Kod çok ağır değil ama güçlü ensemble üretir.
"""

import os
import glob
import time
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV, BayesianRidge, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone

from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb

# =========================================================
# AYARLAR
# =========================================================
TRAIN_PATH = "train_temiz.csv"
TEST_PATH = "test_temiz.csv"
TARGET = "bilissel_performans_skoru"
ID_COL = "id"

N_SPLITS = 5
SEEDS = [42, 2026, 777]
CLIP_MIN, CLIP_MAX = 0.0, 10.0

# Çok yavaşlarsa önce sadece bunu değiştir:
# SEEDS = [42]
# N_SPLITS = 5

OUT_DIR = Path("submissions_final_attack")
OUT_DIR.mkdir(exist_ok=True)

np.random.seed(42)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def read_data():
    print("\n[1] Veriler okunuyor...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    if TARGET not in train.columns:
        raise ValueError(f"Train içinde hedef kolon yok: {TARGET}")

    if ID_COL in test.columns:
        test_id = test[ID_COL].copy()
    else:
        test_id = np.arange(len(test))

    if ID_COL in train.columns:
        train = train.drop(columns=[ID_COL])
    if ID_COL in test.columns:
        test = test.drop(columns=[ID_COL])

    print(f"Train: {train.shape} | Test: {test.shape}")
    return train, test, test_id


def add_features(train, test):
    print("[2] Feature engineering yapılıyor...")
    train = train.copy()
    test = test.copy()
    y = train[TARGET].copy()
    train_x = train.drop(columns=[TARGET])

    all_x = pd.concat([train_x, test], axis=0, ignore_index=True)

    cat_cols = all_x.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = all_x.select_dtypes(exclude=["object", "category"]).columns.tolist()

    # Kategorik temizliği
    for c in cat_cols:
        all_x[c] = all_x[c].astype(str).fillna("missing")

    # Sayısal eksik doldurma
    for c in num_cols:
        med = all_x[c].median()
        all_x[c] = all_x[c].fillna(med)

    eps = 1e-6

    # Ana sinyaller
    if {"rem_yuzdesi", "derin_uyku_yuzdesi"}.issubset(all_x.columns):
        all_x["uyku_kalite_toplam"] = all_x["rem_yuzdesi"] + all_x["derin_uyku_yuzdesi"]
        all_x["rem_derin_fark"] = all_x["rem_yuzdesi"] - all_x["derin_uyku_yuzdesi"]
        all_x["rem_derin_oran"] = all_x["rem_yuzdesi"] / (all_x["derin_uyku_yuzdesi"] + eps)
        all_x["uyku_kalite_kare"] = all_x["uyku_kalite_toplam"] ** 2

    if {"gecelik_uyanma_sayisi", "uykuya_dalma_suresi_dk"}.issubset(all_x.columns):
        all_x["uyku_bozulma_indeksi"] = all_x["gecelik_uyanma_sayisi"] * np.log1p(all_x["uykuya_dalma_suresi_dk"])
        all_x["uyanma_dalma_toplam"] = all_x["gecelik_uyanma_sayisi"] + all_x["uykuya_dalma_suresi_dk"] / 10.0

    if {"rem_yuzdesi", "derin_uyku_yuzdesi", "gecelik_uyanma_sayisi"}.issubset(all_x.columns):
        all_x["uyku_onarim_indeksi"] = (all_x["rem_yuzdesi"] + all_x["derin_uyku_yuzdesi"]) / (all_x["gecelik_uyanma_sayisi"] + 1.0)

    if {"stres_skoru", "gunluk_calisma_saati"}.issubset(all_x.columns):
        all_x["sinerjik_zihinsel_yuk"] = all_x["stres_skoru"] * all_x["gunluk_calisma_saati"]
        all_x["stres_calisma_oran"] = all_x["stres_skoru"] / (all_x["gunluk_calisma_saati"] + eps)
        all_x["stres_kare"] = all_x["stres_skoru"] ** 2
        all_x["calisma_kare"] = all_x["gunluk_calisma_saati"] ** 2

    if {"gunluk_adim_sayisi", "stres_skoru"}.issubset(all_x.columns):
        all_x["hareket_stres_orani"] = np.log1p(all_x["gunluk_adim_sayisi"]) / (all_x["stres_skoru"] + 1.0)
        all_x["adim_log"] = np.log1p(all_x["gunluk_adim_sayisi"])

    if {"uyku_oncesi_kafein_mg", "uyku_oncesi_ekran_suresi_dk"}.issubset(all_x.columns):
        all_x["kafein_ekran_etkilesim"] = np.log1p(all_x["uyku_oncesi_kafein_mg"]) * np.log1p(all_x["uyku_oncesi_ekran_suresi_dk"])
        all_x["kafein_ekran_toplam"] = all_x["uyku_oncesi_kafein_mg"] / 10.0 + all_x["uyku_oncesi_ekran_suresi_dk"]

    if {"dinlenik_nabiz_bpm", "stres_skoru"}.issubset(all_x.columns):
        all_x["nabiz_stres"] = all_x["dinlenik_nabiz_bpm"] * all_x["stres_skoru"]

    if {"oda_sicakligi_celsius"}.issubset(all_x.columns):
        all_x["ideal_sicaklik_sapma"] = np.abs(all_x["oda_sicakligi_celsius"] - 20.0)
        all_x["sicaklik_kare_sapma"] = all_x["ideal_sicaklik_sapma"] ** 2

    if {"hafta_sonu_uyku_farki_saat"}.issubset(all_x.columns):
        all_x["sosyal_jetlag_abs"] = np.abs(all_x["hafta_sonu_uyku_farki_saat"])
        all_x["sosyal_jetlag_kare"] = all_x["hafta_sonu_uyku_farki_saat"] ** 2

    if {"yas"}.issubset(all_x.columns):
        all_x["yas_kare"] = all_x["yas"] ** 2
        all_x["yas_log"] = np.log1p(all_x["yas"])
        all_x["yas_bin"] = pd.cut(all_x["yas"], bins=[0, 25, 35, 45, 55, 65, 120], labels=False).astype(str)

    if {"vucut_kitle_indeksi"}.issubset(all_x.columns):
        all_x["bmi_sapma_22"] = np.abs(all_x["vucut_kitle_indeksi"] - 22.0)
        all_x["bmi_kare"] = all_x["vucut_kitle_indeksi"] ** 2

    # Kategori frekansları ve ikili kombinasyonlar
    cat_cols = all_x.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols:
        freq = all_x[c].value_counts(normalize=True)
        all_x[f"{c}_freq"] = all_x[c].map(freq).astype(float)

    combo_pairs = [
        ("meslek", "ruh_sagligi_durumu"),
        ("meslek", "kronotip"),
        ("cinsiyet", "kronotip"),
        ("mevsim", "gun_tipi"),
        ("ulke", "meslek"),
    ]
    for a, b in combo_pairs:
        if a in all_x.columns and b in all_x.columns:
            name = f"{a}__{b}"
            all_x[name] = all_x[a].astype(str) + "_x_" + all_x[b].astype(str)
            freq = all_x[name].value_counts(normalize=True)
            all_x[f"{name}_freq"] = all_x[name].map(freq).astype(float)

    # Kategorilere göre sayısal ortalama/sapma. Hedef yok, leakage yok.
    group_cats = [c for c in ["meslek", "ruh_sagligi_durumu", "kronotip", "ulke"] if c in all_x.columns]
    key_nums = [c for c in ["stres_skoru", "gunluk_calisma_saati", "gunluk_adim_sayisi", "rem_yuzdesi", "derin_uyku_yuzdesi", "dinlenik_nabiz_bpm"] if c in all_x.columns]
    for gc in group_cats:
        for nc in key_nums:
            m = all_x.groupby(gc)[nc].transform("mean")
            all_x[f"{gc}_{nc}_mean"] = m
            all_x[f"{gc}_{nc}_diff"] = all_x[nc] - m

    # Sonsuz değer temizliği
    all_x = all_x.replace([np.inf, -np.inf], np.nan)
    for c in all_x.columns:
        if all_x[c].dtype.name not in ["object", "category"]:
            all_x[c] = all_x[c].fillna(all_x[c].median())
        else:
            all_x[c] = all_x[c].astype(str).fillna("missing")

    train_fe = all_x.iloc[:len(train_x)].reset_index(drop=True)
    test_fe = all_x.iloc[len(train_x):].reset_index(drop=True)

    cat_cols_final = train_fe.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols_final:
        train_fe[c] = train_fe[c].astype(str)
        test_fe[c] = test_fe[c].astype(str)

    print(f"Feature sayısı: {train_fe.shape[1]} | Kategorik: {len(cat_cols_final)}")
    return train_fe, test_fe, y.reset_index(drop=True), cat_cols_final


def get_models(seed):
    models = []

    models.append(("cat_main", "cat", CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=5000,
        learning_rate=0.025,
        depth=6,
        l2_leaf_reg=5.0,
        random_strength=0.7,
        bagging_temperature=0.4,
        border_count=254,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )))

    models.append(("cat_deep", "cat", CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=4000,
        learning_rate=0.022,
        depth=7,
        l2_leaf_reg=7.0,
        random_strength=1.0,
        bagging_temperature=0.7,
        border_count=254,
        random_seed=seed + 11,
        verbose=False,
        allow_writing_files=False,
    )))

    models.append(("cat_smooth", "cat", CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=6000,
        learning_rate=0.018,
        depth=5,
        l2_leaf_reg=9.0,
        random_strength=1.3,
        bagging_temperature=0.2,
        border_count=254,
        random_seed=seed + 22,
        verbose=False,
        allow_writing_files=False,
    )))

    models.append(("lgb_main", "lgb", lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=6000,
        learning_rate=0.018,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=35,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.04,
        reg_lambda=2.5,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )))

    models.append(("lgb_dartish", "lgb", lgb.LGBMRegressor(
        objective="regression_l1",
        metric="rmse",
        n_estimators=4500,
        learning_rate=0.015,
        num_leaves=45,
        max_depth=7,
        min_child_samples=45,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.08,
        reg_lambda=3.5,
        random_state=seed + 33,
        n_jobs=-1,
        verbosity=-1,
    )))

    models.append(("xgb_main", "xgb", xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=3500,
        learning_rate=0.018,
        max_depth=5,
        min_child_weight=5,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.03,
        reg_lambda=3.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=-1,
    )))

    return models


def prepare_for_lgb_xgb(X_train, X_valid, X_test, cat_cols):
    X_tr = X_train.copy()
    X_va = X_valid.copy()
    X_te = X_test.copy()
    for c in cat_cols:
        cats = pd.Index(pd.concat([X_tr[c], X_va[c], X_te[c]], axis=0).astype(str).unique())
        dtype = pd.CategoricalDtype(categories=cats)
        X_tr[c] = X_tr[c].astype(str).astype(dtype)
        X_va[c] = X_va[c].astype(str).astype(dtype)
        X_te[c] = X_te[c].astype(str).astype(dtype)
    return X_tr, X_va, X_te


def train_one_seed(seed, X, y, X_test, cat_cols):
    print(f"\n================ SEED {seed} ================")
    bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    model_defs = get_models(seed)
    model_names = [m[0] for m in model_defs]

    oof = {name: np.zeros(len(X)) for name in model_names}
    test_pred = {name: np.zeros(len(X_test)) for name in model_names}

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, bins), 1):
        print(f"Seed {seed} | Fold {fold}/{N_SPLITS}")
        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        for name, kind, model in model_defs:
            start = time.time()
            mdl = clone(model)

            if kind == "cat":
                mdl.fit(
                    X_tr, y_tr,
                    eval_set=(X_va, y_va),
                    cat_features=cat_cols,
                    early_stopping_rounds=300,
                    use_best_model=True,
                    verbose=False,
                )
                pred_va = mdl.predict(X_va)
                pred_te = mdl.predict(X_test)

            elif kind == "lgb":
                X_tr2, X_va2, X_te2 = prepare_for_lgb_xgb(X_tr, X_va, X_test, cat_cols)
                mdl.fit(
                    X_tr2, y_tr,
                    eval_set=[(X_va2, y_va)],
                    categorical_feature=cat_cols,
                    callbacks=[lgb.early_stopping(300, verbose=False)],
                )
                pred_va = mdl.predict(X_va2)
                pred_te = mdl.predict(X_te2)

            elif kind == "xgb":
                X_tr2, X_va2, X_te2 = prepare_for_lgb_xgb(X_tr, X_va, X_test, cat_cols)
                mdl.fit(X_tr2, y_tr, eval_set=[(X_va2, y_va)], verbose=False)
                pred_va = mdl.predict(X_va2)
                pred_te = mdl.predict(X_te2)

            else:
                raise ValueError(kind)

            pred_va = np.clip(pred_va, CLIP_MIN, CLIP_MAX)
            pred_te = np.clip(pred_te, CLIP_MIN, CLIP_MAX)

            oof[name][va_idx] = pred_va
            test_pred[name] += pred_te / N_SPLITS

            print(f"  {name:<12} RMSE={rmse(y_va, pred_va):.6f} | {time.time()-start:.1f}s")

    scores = {name: rmse(y, oof[name]) for name in model_names}
    print("\nBase model OOF skorları:")
    print(json.dumps(dict(sorted(scores.items(), key=lambda x: x[1])), indent=2))

    meta_train = np.column_stack([oof[n] for n in model_names])
    meta_test = np.column_stack([test_pred[n] for n in model_names])

    meta_outputs = {}

    ridge = RidgeCV(alphas=np.logspace(-4, 3, 30))
    ridge.fit(meta_train, y)
    meta_outputs["meta_ridge"] = np.clip(ridge.predict(meta_test), CLIP_MIN, CLIP_MAX)
    ridge_oof = np.clip(ridge.predict(meta_train), CLIP_MIN, CLIP_MAX)

    bayes = BayesianRidge()
    bayes.fit(meta_train, y)
    meta_outputs["meta_bayes"] = np.clip(bayes.predict(meta_test), CLIP_MIN, CLIP_MAX)
    bayes_oof = np.clip(bayes.predict(meta_train), CLIP_MIN, CLIP_MAX)

    huber = make_pipeline(StandardScaler(), HuberRegressor(alpha=0.0001, epsilon=1.35, max_iter=1000))
    huber.fit(meta_train, y)
    meta_outputs["meta_huber"] = np.clip(huber.predict(meta_test), CLIP_MIN, CLIP_MAX)
    huber_oof = np.clip(huber.predict(meta_train), CLIP_MIN, CLIP_MAX)

    # OOF üzerinden rastgele ağırlık araması
    rng = np.random.default_rng(seed)
    best_w = None
    best_score = 999
    for _ in range(25000):
        w = rng.dirichlet(np.ones(len(model_names)))
        p = np.clip(meta_train @ w, CLIP_MIN, CLIP_MAX)
        s = rmse(y, p)
        if s < best_score:
            best_score = s
            best_w = w

    weighted_oof = np.clip(meta_train @ best_w, CLIP_MIN, CLIP_MAX)
    weighted_test = np.clip(meta_test @ best_w, CLIP_MIN, CLIP_MAX)
    meta_outputs["meta_weight_search"] = weighted_test

    # Final güvenli blend: meta modeller + en iyi ağırlık araması
    final_oof = np.mean([ridge_oof, bayes_oof, huber_oof, weighted_oof], axis=0)
    final_test = np.mean([meta_outputs["meta_ridge"], meta_outputs["meta_bayes"], meta_outputs["meta_huber"], meta_outputs["meta_weight_search"]], axis=0)
    meta_outputs["final_seed_blend"] = np.clip(final_test, CLIP_MIN, CLIP_MAX)

    final_scores = {
        "meta_ridge": rmse(y, ridge_oof),
        "meta_bayes": rmse(y, bayes_oof),
        "meta_huber": rmse(y, huber_oof),
        "meta_weight_search": rmse(y, weighted_oof),
        "final_seed_blend": rmse(y, final_oof),
    }

    print("\nMeta skorları:")
    print(json.dumps(dict(sorted(final_scores.items(), key=lambda x: x[1])), indent=2))
    print("Ağırlık arama katsayıları:")
    for n, w in sorted(zip(model_names, best_w), key=lambda x: -x[1]):
        print(f"  {n:<12}: {w:.4f}")

    return oof, test_pred, meta_outputs, scores, final_scores


def save_submission(name, test_id, preds):
    preds = np.clip(np.asarray(preds), CLIP_MIN, CLIP_MAX)
    path = OUT_DIR / f"{name}.csv"
    pd.DataFrame({ID_COL: test_id, TARGET: preds}).to_csv(path, index=False)
    print(f"Kaydedildi: {path}")
    return path


def auto_blend_with_old_submissions(test_id, new_pred):
    print("\n[5] Önceki submission dosyalarıyla opsiyonel blend üretiliyor...")
    old_files = []
    for p in glob.glob("submission*.csv"):
        # Bu scriptin output klasöründekileri değil, ana klasördeki eski dosyaları al
        if os.path.isfile(p):
            old_files.append(p)

    if not old_files:
        print("Ana klasörde eski submission*.csv bulunamadı, bu adım geçildi.")
        return

    for p in old_files:
        try:
            df = pd.read_csv(p)
            if TARGET not in df.columns or len(df) != len(new_pred):
                continue
            old = df[TARGET].values
            base = Path(p).stem.replace("submission_", "old_")
            for w_new in [0.25, 0.40, 0.50, 0.60, 0.75]:
                blend = w_new * new_pred + (1 - w_new) * old
                save_submission(f"blend_{int(w_new*100)}new_OLD_{base}", test_id, blend)
        except Exception as e:
            print(f"Blend geçildi: {p} | {e}")


def main():
    total_start = time.time()
    train, test, test_id = read_data()
    X, X_test, y, cat_cols = add_features(train, test)

    all_seed_meta = {"meta_ridge": [], "meta_bayes": [], "meta_huber": [], "meta_weight_search": [], "final_seed_blend": []}
    all_seed_base = {}
    report = {}

    for seed in SEEDS:
        oof, test_pred, meta_outputs, base_scores, meta_scores = train_one_seed(seed, X, y, X_test, cat_cols)
        report[f"seed_{seed}_base"] = base_scores
        report[f"seed_{seed}_meta"] = meta_scores

        for k, v in meta_outputs.items():
            all_seed_meta[k].append(v)
        for k, v in test_pred.items():
            all_seed_base.setdefault(k, []).append(v)

    print("\n[4] Submission dosyaları yazılıyor...")

    # Seed ortalamaları
    final_candidates = {}
    for k, arrs in all_seed_meta.items():
        final_candidates[k] = np.mean(arrs, axis=0)
        save_submission(f"submission_{k}_seedmean", test_id, final_candidates[k])

    for k, arrs in all_seed_base.items():
        final_candidates[f"base_{k}"] = np.mean(arrs, axis=0)
        save_submission(f"submission_base_{k}_seedmean", test_id, final_candidates[f"base_{k}"])

    # En güvenli nihai dosyalar
    superblend = np.mean([
        final_candidates["meta_ridge"],
        final_candidates["meta_bayes"],
        final_candidates["meta_huber"],
        final_candidates["meta_weight_search"],
        final_candidates["final_seed_blend"],
    ], axis=0)
    save_submission("submission_FINAL_SUPERBLEND", test_id, superblend)

    conservative = np.mean([
        final_candidates["meta_ridge"],
        final_candidates["meta_bayes"],
        final_candidates["base_cat_main"],
        final_candidates["base_cat_deep"],
        final_candidates["base_cat_smooth"],
    ], axis=0)
    save_submission("submission_FINAL_CONSERVATIVE_CAT_META", test_id, conservative)

    cat_only = np.mean([
        final_candidates["base_cat_main"],
        final_candidates["base_cat_deep"],
        final_candidates["base_cat_smooth"],
    ], axis=0)
    save_submission("submission_FINAL_CAT_ONLY", test_id, cat_only)

    # Eski iyi submission ile otomatik blendler
    auto_blend_with_old_submissions(test_id, superblend)

    with open(OUT_DIR / "cv_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n====================================================")
    print("BİTTİ. İlk denenecek dosyalar:")
    print("1) submissions_final_attack/submission_FINAL_SUPERBLEND.csv")
    print("2) submissions_final_attack/submission_FINAL_CONSERVATIVE_CAT_META.csv")
    print("3) submissions_final_attack/submission_meta_bayes_seedmean.csv")
    print("4) submissions_final_attack/submission_meta_ridge_seedmean.csv")
    print("5) submissions_final_attack/submission_FINAL_CAT_ONLY.csv")
    print("6) Eski submission varsa blend_... dosyaları")
    print(f"Toplam süre: {(time.time() - total_start) / 60:.1f} dakika")
    print("====================================================")


if __name__ == "__main__":
    main()
