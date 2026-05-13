import itertools

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

from hedef_1_1_pro import (
    TARGET,
    build_feature_space,
    train_external_proxy_features,
)


SEED = 42


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def forward_target(y):
    p = np.clip(np.asarray(y) / 10.0, 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def inverse_target(z):
    p = 1.0 / (1.0 + np.exp(-np.asarray(z)))
    return np.clip(p * 10.0, 0.0, 10.0)


def prepare():
    train_df = pd.read_csv("train_temiz.csv")
    test_df = pd.read_csv("test_temiz.csv")
    x_train, x_test, y = build_feature_space(train_df, test_df)
    train_proxy_dict, _, _, _, _ = train_external_proxy_features(x_train, x_test)
    for col, values in train_proxy_dict.items():
        x_train[col] = values
    return x_train, y


def eval_config(name, config, x_train, y, folds, cat_cols):
    oof = np.zeros(len(x_train), dtype=np.float32)
    for tr_idx, va_idx in folds:
        x_tr = x_train.iloc[tr_idx]
        x_va = x_train.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=config["iterations"],
            learning_rate=config["learning_rate"],
            depth=config["depth"],
            l2_leaf_reg=config["l2_leaf_reg"],
            random_strength=config["random_strength"],
            bagging_temperature=config["bagging_temperature"],
            random_seed=config["seed"],
            verbose=False,
            allow_writing_files=False,
        )

        if config.get("target_mode") == "logit":
            model.fit(
                x_tr,
                forward_target(y_tr.values),
                cat_features=cat_cols,
                eval_set=(x_va, forward_target(y_va.values)),
                early_stopping_rounds=150,
            )
            oof[va_idx] = inverse_target(model.predict(x_va))
        else:
            model.fit(
                x_tr,
                y_tr,
                cat_features=cat_cols,
                eval_set=(x_va, y_va),
                early_stopping_rounds=150,
            )
            oof[va_idx] = model.predict(x_va)

    return {"name": name, "rmse": rmse(y, oof), "oof": oof}


def best_pairwise(results, y):
    best = None
    for left, right in itertools.combinations(results, 2):
        for w in np.linspace(0.0, 1.0, 41):
            blend = left["oof"] * (1.0 - w) + right["oof"] * w
            score = rmse(y, blend)
            if best is None or score < best["rmse"]:
                best = {
                    "left": left["name"],
                    "right": right["name"],
                    "weight_right": float(w),
                    "rmse": score,
                }
    return best


def main():
    x_train, y = prepare()
    cat_cols = x_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    y_bins = pd.qcut(y, q=12, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    folds = list(skf.split(x_train, y_bins))

    configs = {
        "smooth_base": {
            "iterations": 2600,
            "learning_rate": 0.02,
            "depth": 5,
            "l2_leaf_reg": 8.0,
            "random_strength": 1.1,
            "bagging_temperature": 0.15,
            "seed": SEED,
        },
        "deep_base": {
            "iterations": 2200,
            "learning_rate": 0.022,
            "depth": 6,
            "l2_leaf_reg": 6.0,
            "random_strength": 0.8,
            "bagging_temperature": 0.35,
            "seed": SEED + 17,
        },
        "deep_more_iter": {
            "iterations": 3000,
            "learning_rate": 0.018,
            "depth": 6,
            "l2_leaf_reg": 6.0,
            "random_strength": 0.8,
            "bagging_temperature": 0.35,
            "seed": 2026,
        },
        "deep_stronger_reg": {
            "iterations": 2600,
            "learning_rate": 0.02,
            "depth": 6,
            "l2_leaf_reg": 8.5,
            "random_strength": 0.95,
            "bagging_temperature": 0.22,
            "seed": 777,
        },
        "deep_lower_rand": {
            "iterations": 2400,
            "learning_rate": 0.021,
            "depth": 6,
            "l2_leaf_reg": 5.0,
            "random_strength": 0.55,
            "bagging_temperature": 0.45,
            "seed": 888,
        },
        "wide_d7": {
            "iterations": 2200,
            "learning_rate": 0.02,
            "depth": 7,
            "l2_leaf_reg": 7.5,
            "random_strength": 0.8,
            "bagging_temperature": 0.25,
            "seed": 512,
        },
        "smooth_alt": {
            "iterations": 3200,
            "learning_rate": 0.017,
            "depth": 5,
            "l2_leaf_reg": 9.0,
            "random_strength": 1.25,
            "bagging_temperature": 0.08,
            "seed": 314,
        },
        "logit_model": {
            "iterations": 2600,
            "learning_rate": 0.02,
            "depth": 6,
            "l2_leaf_reg": 6.0,
            "random_strength": 0.8,
            "bagging_temperature": 0.20,
            "seed": 911,
            "target_mode": "logit",
        },
    }

    results = []
    for name, cfg in configs.items():
        print(f"\n{name} basliyor...", flush=True)
        result = eval_config(name, cfg, x_train, y, folds, cat_cols)
        results.append(result)
        print(f"{name} rmse={result['rmse']:.6f}", flush=True)

    results = sorted(results, key=lambda item: item["rmse"])
    print("\nTekil sonuclar", flush=True)
    for item in results:
        print(f"{item['name']:20s} {item['rmse']:.6f}", flush=True)

    top_results = results[:5]
    best_pair = best_pairwise(top_results, y)
    print("\nEn iyi ikili blend", flush=True)
    print(best_pair, flush=True)


if __name__ == "__main__":
    main()
