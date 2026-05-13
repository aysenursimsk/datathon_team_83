import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from hedef_1_1_pro import (
    TARGET,
    build_external_aligned_frame,
    build_feature_space,
)


SEED = 42


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def prepare_data():
    train_df = pd.read_csv("train_temiz.csv")
    test_df = pd.read_csv("test_temiz.csv")
    x_train, x_test, y = build_feature_space(train_df, test_df)

    external_df = build_external_aligned_frame()
    x_ext, _, y_ext = build_feature_space(
        external_df,
        external_df.drop(columns=[TARGET]).copy(),
    )
    return x_train, x_test, y, x_ext, y_ext


def train_domain_model(x_train, x_ext):
    common_cols = [c for c in x_train.columns if c in x_ext.columns]
    x_train_dom = x_train[common_cols].copy()
    x_ext_dom = x_ext[common_cols].copy()

    domain_x = pd.concat([x_train_dom, x_ext_dom], axis=0, ignore_index=True)
    domain_y = np.r_[np.ones(len(x_train_dom), dtype=int), np.zeros(len(x_ext_dom), dtype=int)]
    cat_cols = domain_x.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    oof = np.zeros(len(domain_x), dtype=np.float32)
    test_pred = np.zeros(len(domain_x), dtype=np.float32)

    for tr_idx, va_idx in skf.split(domain_x, domain_y):
        model = CatBoostClassifier(
            iterations=900,
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
        fold_pred = model.predict_proba(domain_x.iloc[va_idx])[:, 1]
        oof[va_idx] = fold_pred
        test_pred += model.predict_proba(domain_x)[:, 1] / skf.n_splits

    auc = roc_auc_score(domain_y, oof)
    contest_prob_train = test_pred[:len(x_train_dom)]
    contest_prob_ext = test_pred[len(x_train_dom):]
    return auc, common_cols, contest_prob_train, contest_prob_ext


def build_proxy_predictions(x_train, x_test, x_ext, y_ext, ext_weights=None):
    common_cols = [c for c in x_train.columns if c in x_ext.columns]
    x_ext_use = x_ext[common_cols].copy()
    x_train_use = x_train[common_cols].copy()
    x_test_use = x_test[common_cols].copy()
    cat_cols = x_ext_use.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    proxy = CatBoostRegressor(
        iterations=1800,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5.0,
        random_strength=0.8,
        bagging_temperature=0.2,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
    )
    fit_kwargs = {}
    if ext_weights is not None:
        fit_kwargs["sample_weight"] = ext_weights
    proxy.fit(x_ext_use, y_ext, cat_features=cat_cols, **fit_kwargs)

    train_pred = proxy.predict(x_train_use)
    test_pred = proxy.predict(x_test_use)
    return train_pred, test_pred


def evaluate_variant(name, x_train, y):
    cat_cols = x_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    y_bins = pd.qcut(y, q=12, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    oof_smooth = np.zeros(len(x_train), dtype=np.float32)
    oof_deep = np.zeros(len(x_train), dtype=np.float32)

    for tr_idx, va_idx in skf.split(x_train, y_bins):
        x_tr = x_train.iloc[tr_idx]
        x_va = x_train.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        smooth = CatBoostRegressor(
            iterations=2600,
            learning_rate=0.02,
            depth=5,
            l2_leaf_reg=8.0,
            random_strength=1.1,
            bagging_temperature=0.15,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False,
        )
        smooth.fit(
            x_tr,
            y_tr,
            cat_features=cat_cols,
            eval_set=(x_va, y_va),
            early_stopping_rounds=150,
        )
        oof_smooth[va_idx] = smooth.predict(x_va)

        deep = CatBoostRegressor(
            iterations=2200,
            learning_rate=0.022,
            depth=6,
            l2_leaf_reg=6.0,
            random_strength=0.8,
            bagging_temperature=0.35,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=SEED + 17,
            verbose=False,
            allow_writing_files=False,
        )
        deep.fit(
            x_tr,
            y_tr,
            cat_features=cat_cols,
            eval_set=(x_va, y_va),
            early_stopping_rounds=150,
        )
        oof_deep[va_idx] = deep.predict(x_va)

    best_score = None
    best_weight = None
    for w in np.linspace(0.0, 1.0, 41):
        blended = oof_smooth * (1.0 - w) + oof_deep * w
        score = rmse(y, blended)
        if best_score is None or score < best_score:
            best_score = score
            best_weight = w

    return {
        "name": name,
        "smooth_rmse": rmse(y, oof_smooth),
        "deep_rmse": rmse(y, oof_deep),
        "blend_weight": best_weight,
        "blend_rmse": best_score,
    }


def main():
    x_train, x_test, y, x_ext, y_ext = prepare_data()
    print(f"contest shape={x_train.shape} external shape={x_ext.shape}", flush=True)

    auc, _, contest_prob_train, contest_prob_ext = train_domain_model(x_train, x_ext)
    print(f"domain auc={auc:.6f}", flush=True)
    print(
        f"contest_prob_train mean={contest_prob_train.mean():.6f} | "
        f"contest_prob_ext mean={contest_prob_ext.mean():.6f}",
        flush=True,
    )

    ext_w_raw = np.clip(contest_prob_ext, 1e-3, 1.0)
    ext_w_scaled = 0.2 + 2.5 * ext_w_raw
    ext_keep = ext_w_raw >= np.quantile(ext_w_raw, 0.55)

    variants = []

    base_train_proxy, _ = build_proxy_predictions(x_train, x_test, x_ext, y_ext, None)
    x_base = x_train.copy()
    x_base["ext_proxy_cog"] = base_train_proxy
    variants.append(("base_proxy", x_base))

    weighted_train_proxy, _ = build_proxy_predictions(x_train, x_test, x_ext, y_ext, ext_w_scaled)
    x_weighted = x_train.copy()
    x_weighted["ext_proxy_cog_weighted"] = weighted_train_proxy
    variants.append(("weighted_proxy_only", x_weighted))

    x_dual = x_train.copy()
    x_dual["ext_proxy_cog"] = base_train_proxy
    x_dual["ext_proxy_cog_weighted"] = weighted_train_proxy
    x_dual["ext_similarity"] = contest_prob_train
    variants.append(("base_plus_weighted_plus_similarity", x_dual))

    filtered_train_proxy, _ = build_proxy_predictions(
        x_train,
        x_test,
        x_ext.loc[ext_keep].reset_index(drop=True),
        y_ext.loc[ext_keep].reset_index(drop=True),
        None,
    )
    x_filtered = x_train.copy()
    x_filtered["ext_proxy_cog_filtered"] = filtered_train_proxy
    variants.append(("filtered_proxy_only", x_filtered))

    x_combo = x_train.copy()
    x_combo["ext_proxy_cog"] = base_train_proxy
    x_combo["ext_proxy_cog_filtered"] = filtered_train_proxy
    x_combo["ext_similarity"] = contest_prob_train
    variants.append(("base_plus_filtered_plus_similarity", x_combo))

    results = []
    for name, x_variant in variants:
        print(f"\n{name} basliyor...", flush=True)
        result = evaluate_variant(name, x_variant, y)
        results.append(result)
        print(result, flush=True)

    result_df = pd.DataFrame(results).sort_values("blend_rmse")
    print("\nSirali sonuclar", flush=True)
    print(result_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
