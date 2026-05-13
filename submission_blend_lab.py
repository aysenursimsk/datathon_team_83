import os
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "bilissel_performans_skoru"
OUT_DIR = Path("submission_blends")
OUT_DIR.mkdir(exist_ok=True)


def load_submission(path):
    df = pd.read_csv(path)
    pred_col = [c for c in df.columns if c != "id"][0]
    return df[["id", pred_col]].rename(columns={pred_col: TARGET})


def clip_preds(series):
    return np.clip(series, 0.0, 10.0)


def save_submission(name, ids, preds):
    out_path = OUT_DIR / name
    pd.DataFrame({"id": ids, TARGET: clip_preds(preds)}).to_csv(out_path, index=False)
    return out_path


def weighted_blend(base_df, weight_map):
    total = None
    for path, weight in weight_map.items():
        preds = load_submission(path)[TARGET]
        total = preds * weight if total is None else total + preds * weight
    return total


def rank_blend_rescaled(frames, reference):
    ranks = []
    for frame in frames:
        ranks.append(frame[TARGET].rank(method="average", pct=True))
    avg_rank = sum(ranks) / len(ranks)
    sorted_ref = np.sort(reference[TARGET].values)
    idx = np.clip((avg_rank.values * (len(sorted_ref) - 1)).round().astype(int), 0, len(sorted_ref) - 1)
    mapped = pd.Series(sorted_ref[idx], index=reference.index)
    return mapped


def main():
    files = {
        "external": "submission_EXTERNAL_PROXY_CATBOOST_BLEND.csv",
        "scipy": "submission_SCIPY_OPTIMIZED_1_1X.csv",
        "magic": "submission_MAGIC_MOE_STACK.csv",
        "ultra": "submission_TEAM83_ULTRA_CLEAN.csv",
        "hedef": "submission_HEDEF_1_19_FINAL.csv",
        "target022": "submission_FINAL_TARGET_022.csv",
        "alt1200": "submission_1_200_alti_denemesi.csv",
        "log": "submission_LOG_TRANSFORM_Stacking.csv",
    }

    loaded = {key: load_submission(path) for key, path in files.items()}
    ids = loaded["external"]["id"]

    specs = [
        (
            "blend_safe_anchor.csv",
            {
                files["external"]: 0.60,
                files["scipy"]: 0.25,
                files["magic"]: 0.15,
            },
        ),
        (
            "blend_diverse_anchor.csv",
            {
                files["external"]: 0.55,
                files["scipy"]: 0.15,
                files["target022"]: 0.15,
                files["hedef"]: 0.15,
            },
        ),
        (
            "blend_wide_diversity.csv",
            {
                files["external"]: 0.50,
                files["scipy"]: 0.15,
                files["ultra"]: 0.15,
                files["target022"]: 0.10,
                files["log"]: 0.10,
            },
        ),
        (
            "blend_trimmed_mean.csv",
            {
                files["external"]: 0.40,
                files["scipy"]: 0.15,
                files["magic"]: 0.15,
                files["ultra"]: 0.10,
                files["hedef"]: 0.10,
                files["target022"]: 0.10,
            },
        ),
    ]

    print("Blends olusturuluyor...\n", flush=True)
    saved_paths = []
    for out_name, weight_map in specs:
        preds = weighted_blend(loaded["external"], weight_map)
        path = save_submission(out_name, ids, preds)
        saved_paths.append(path)
        print(out_name, flush=True)
        for src, w in weight_map.items():
            print(f"  {w:.2f}  {os.path.basename(src)}", flush=True)
        print(f"  mean={preds.mean():.6f} std={preds.std():.6f}\n", flush=True)

    rank_preds = rank_blend_rescaled(
        [
            loaded["external"],
            loaded["scipy"],
            loaded["target022"],
            loaded["hedef"],
            loaded["log"],
        ],
        loaded["external"],
    )
    rank_path = save_submission("blend_rank_rescaled.csv", ids, rank_preds)
    saved_paths.append(rank_path)
    print("blend_rank_rescaled.csv", flush=True)
    print("  rank blend of external + scipy + target022 + hedef + log", flush=True)
    print(f"  mean={rank_preds.mean():.6f} std={rank_preds.std():.6f}\n", flush=True)

    summary_rows = []
    for path in saved_paths:
        df = pd.read_csv(path)
        s = df[TARGET]
        summary_rows.append(
            {
                "file": path.name,
                "mean": round(float(s.mean()), 6),
                "std": round(float(s.std()), 6),
                "min": round(float(s.min()), 6),
                "max": round(float(s.max()), 6),
            }
        )

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "blend_summary.csv", index=False)
    print("Ozet kaydedildi: submission_blends/blend_summary.csv", flush=True)


if __name__ == "__main__":
    main()
