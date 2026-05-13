import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

from magic_moe_stack import TARGET, TRAIN_PATH, TEST_PATH, build_features


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_cfg(name, params, x, y, cat_cols, folds):
    oof = np.zeros(len(x), dtype=np.float32)
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        X_tr = x.iloc[tr_idx]
        X_va = x.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=params["iterations"],
            learning_rate=params["learning_rate"],
            depth=params["depth"],
            l2_leaf_reg=params["l2_leaf_reg"],
            random_strength=params["random_strength"],
            bagging_temperature=params["bagging_temperature"],
            border_count=254,
            random_seed=params["seed"],
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            X_tr,
            y_tr,
            cat_features=cat_cols,
            eval_set=(X_va, y_va),
            early_stopping_rounds=150,
        )
        preds = model.predict(X_va)
        oof[va_idx] = preds
        fold_scores.append(rmse(y_va, preds))

    return {
        "name": name,
        "rmse": rmse(y, oof),
        "folds": fold_scores,
    }


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    x, _, y, _ = build_features(train_df, test_df)
    cat_cols = x.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    y_bins = pd.qcut(y, q=12, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    folds = list(skf.split(x, y_bins))

    configs = {
        "base_d6": {
            "iterations": 2600,
            "learning_rate": 0.022,
            "depth": 6,
            "l2_leaf_reg": 6.0,
            "random_strength": 0.8,
            "bagging_temperature": 0.35,
            "seed": 42,
        },
        "shallower_d5": {
            "iterations": 3200,
            "learning_rate": 0.02,
            "depth": 5,
            "l2_leaf_reg": 8.0,
            "random_strength": 1.1,
            "bagging_temperature": 0.15,
            "seed": 42,
        },
        "deeper_d7": {
            "iterations": 2200,
            "learning_rate": 0.024,
            "depth": 7,
            "l2_leaf_reg": 5.0,
            "random_strength": 0.6,
            "bagging_temperature": 0.45,
            "seed": 42,
        },
        "smooth_d6": {
            "iterations": 3600,
            "learning_rate": 0.016,
            "depth": 6,
            "l2_leaf_reg": 10.0,
            "random_strength": 1.2,
            "bagging_temperature": 0.1,
            "seed": 2026,
        },
        "aggressive_d6": {
            "iterations": 1800,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 4.0,
            "random_strength": 0.4,
            "bagging_temperature": 0.6,
            "seed": 777,
        },
        "wide_d8": {
            "iterations": 1800,
            "learning_rate": 0.025,
            "depth": 8,
            "l2_leaf_reg": 7.0,
            "random_strength": 0.7,
            "bagging_temperature": 0.25,
            "seed": 777,
        },
    }

    results = []
    for name, params in configs.items():
        print(f"\n{name} basliyor...", flush=True)
        result = evaluate_cfg(name, params, x, y, cat_cols, folds)
        results.append(result)
        print(f"{name} RMSE: {result['rmse']:.5f} | foldler: {[round(s, 5) for s in result['folds']]}", flush=True)

    results = sorted(results, key=lambda item: item["rmse"])
    print("\nSirali sonuclar", flush=True)
    for item in results:
        print(f"{item['name']:14s} -> {item['rmse']:.5f}", flush=True)


if __name__ == "__main__":
    main()
