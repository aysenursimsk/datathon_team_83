from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor

from hedef_1_1_pro import (
    TARGET,
    get_test_id,
    build_feature_space,
    train_external_proxy_features,
)


OUT_DIR = Path("submission_aggressive_lab")
OUT_DIR.mkdir(exist_ok=True)


def load_submission(path):
    df = pd.read_csv(path)
    pred_col = [c for c in df.columns if c != "id"][0]
    return df[["id", pred_col]].rename(columns={pred_col: TARGET})


def clip_preds(preds):
    return np.clip(np.asarray(preds), 0.0, 10.0)


def save_submission(name, ids, preds):
    out_path = OUT_DIR / name
    pd.DataFrame({"id": ids, TARGET: clip_preds(preds)}).to_csv(out_path, index=False)
    return out_path


def weighted_average(sub_map):
    total = None
    for df, weight in sub_map:
        s = df[TARGET].values
        total = s * weight if total is None else total + s * weight
    return total


def train_full_catboost(train_x, y, test_x, sample_weight, config, cat_cols):
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
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(train_x, y, cat_features=cat_cols, **fit_kwargs)
    return model.predict(test_x)


def build_current_feature_space():
    train_df = pd.read_csv("train_temiz.csv")
    test_df = pd.read_csv("test_temiz.csv")
    test_id = get_test_id(test_df)
    train_x, test_x, y = build_feature_space(train_df, test_df)
    train_proxy_dict, test_proxy_dict, _, _, _ = train_external_proxy_features(train_x, test_x)
    for col, values in train_proxy_dict.items():
        train_x[col] = values
    for col, values in test_proxy_dict.items():
        test_x[col] = values
    return train_x, test_x, y, test_id


def main():
    print("Aggressive submission lab basliyor...", flush=True)
    train_x, test_x, y, test_id = build_current_feature_space()
    cat_cols = train_x.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    # Core submission family
    sub_best_local = load_submission("submission_BEST_LOCAL_CV_120058.csv")
    sub_best_safe = load_submission("submission_BEST_LB_SAFE_BLEND.csv")
    sub_diverse = load_submission("submission_blends/blend_diverse_anchor.csv")
    sub_trimmed = load_submission("submission_blends/blend_trimmed_mean.csv")
    sub_rank = load_submission("submission_blends/blend_rank_rescaled.csv")
    sub_scipy = load_submission("submission_SCIPY_OPTIMIZED_1_1X.csv")
    sub_magic = load_submission("submission_MAGIC_MOE_STACK.csv")
    sub_target022 = load_submission("submission_FINAL_TARGET_022.csv")

    # Stage-2 LB blends
    stage2_specs = {
        "submission_STAGE2_SAFE_PLUS.csv": [
            (sub_best_local, 0.45),
            (sub_best_safe, 0.35),
            (sub_diverse, 0.10),
            (sub_rank, 0.10),
        ],
        "submission_STAGE2_DIVERSE_PLUS.csv": [
            (sub_best_local, 0.35),
            (sub_best_safe, 0.20),
            (sub_diverse, 0.15),
            (sub_trimmed, 0.10),
            (sub_rank, 0.10),
            (sub_scipy, 0.10),
        ],
        "submission_STAGE2_RANK_MIX.csv": [
            (sub_best_local, 0.30),
            (sub_best_safe, 0.20),
            (sub_rank, 0.20),
            (sub_diverse, 0.10),
            (sub_magic, 0.10),
            (sub_target022, 0.10),
        ],
    }

    summary_rows = []
    for name, spec in stage2_specs.items():
        preds = weighted_average(spec)
        path = save_submission(name, test_id, preds)
        summary_rows.append(
            {
                "file": path.name,
                "kind": "stage2_blend",
                "mean": round(float(np.mean(preds)), 6),
                "std": round(float(np.std(preds)), 6),
            }
        )
        print(f"{name} kaydedildi", flush=True)

    # Pseudo-label family
    ensemble_stack = np.vstack([
        sub_best_local[TARGET].values,
        sub_best_safe[TARGET].values,
        sub_diverse[TARGET].values,
        sub_trimmed[TARGET].values,
        sub_rank[TARGET].values,
        sub_scipy[TARGET].values,
        sub_magic[TARGET].values,
    ])
    ensemble_mean = np.average(
        ensemble_stack,
        axis=0,
        weights=np.array([0.30, 0.20, 0.10, 0.10, 0.10, 0.10, 0.10]),
    )
    ensemble_std = ensemble_stack.std(axis=0)

    confidence = 1.0 - (ensemble_std - ensemble_std.min()) / (ensemble_std.max() - ensemble_std.min() + 1e-9)
    confidence = np.clip(confidence, 0.0, 1.0)

    pseudo_specs = [
        ("submission_PSEUDO_TOP15_W035.csv", 0.15, 0.35),
        ("submission_PSEUDO_TOP30_W025.csv", 0.30, 0.25),
        ("submission_PSEUDO_ALL_SOFT.csv", 1.00, 0.12),
    ]

    model_a_cfg = {
        "iterations": 3000,
        "learning_rate": 0.018,
        "depth": 6,
        "l2_leaf_reg": 6.0,
        "random_strength": 0.8,
        "bagging_temperature": 0.35,
        "seed": 2026,
    }
    model_b_cfg = {
        "iterations": 2200,
        "learning_rate": 0.022,
        "depth": 6,
        "l2_leaf_reg": 6.0,
        "random_strength": 0.8,
        "bagging_temperature": 0.35,
        "seed": 59,
    }

    for name, frac, base_w in pseudo_specs:
        if frac < 1.0:
            keep_n = int(len(test_x) * frac)
            keep_idx = np.argsort(ensemble_std)[:keep_n]
        else:
            keep_idx = np.arange(len(test_x))

        pseudo_x = test_x.iloc[keep_idx].copy()
        pseudo_y = pd.Series(ensemble_mean[keep_idx], index=pseudo_x.index)
        pseudo_weights = base_w * (0.35 + 0.65 * confidence[keep_idx])

        aug_x = pd.concat([train_x, pseudo_x], axis=0).reset_index(drop=True)
        aug_y = pd.concat([y.reset_index(drop=True), pseudo_y.reset_index(drop=True)], axis=0).reset_index(drop=True)
        sample_weight = np.r_[
            np.ones(len(train_x), dtype=np.float32),
            pseudo_weights.astype(np.float32),
        ]

        pred_a = train_full_catboost(aug_x, aug_y, test_x, sample_weight, model_a_cfg, cat_cols)
        pred_b = train_full_catboost(aug_x, aug_y, test_x, sample_weight, model_b_cfg, cat_cols)
        final_pred = 0.55 * pred_a + 0.45 * pred_b
        path = save_submission(name, test_id, final_pred)

        summary_rows.append(
            {
                "file": path.name,
                "kind": "pseudo_label",
                "mean": round(float(np.mean(final_pred)), 6),
                "std": round(float(np.std(final_pred)), 6),
                "pseudo_frac": frac,
                "pseudo_weight": base_w,
            }
        )
        print(
            f"{name} kaydedildi | pseudo_rows={len(keep_idx)} | "
            f"std_q_max={ensemble_std[keep_idx].max():.6f} | base_w={base_w}",
            flush=True,
        )

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "aggressive_summary.csv", index=False)
    print("Ozet: submission_aggressive_lab/aggressive_summary.csv", flush=True)


if __name__ == "__main__":
    main()
