import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hedef_1_1_pro import TARGET, build_external_aligned_frame, build_feature_space, get_test_id


SEED = 42


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def weighted_from_neighbors(distances, indices, y_external, power=1.0):
    weights = 1.0 / np.maximum(distances, 1e-9) ** power
    neighbor_values = y_external.values[indices]
    return (weights * neighbor_values).sum(axis=1) / weights.sum(axis=1)


def build_retrieval_features(x_train, x_test, x_external, y_external):
    shared_cols = [c for c in x_train.columns if c in x_external.columns]
    train_shared = x_train[shared_cols].copy()
    test_shared = x_test[shared_cols].copy()
    external_shared = x_external[shared_cols].copy()

    cat_cols = train_shared.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_cols = [c for c in shared_cols if c not in cat_cols]

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        sparse_threshold=0.3,
    )

    train_matrix = preprocessor.fit_transform(train_shared)
    test_matrix = preprocessor.transform(test_shared)
    external_matrix = preprocessor.transform(external_shared)

    nn = NearestNeighbors(n_neighbors=5, metric="manhattan", algorithm="brute", n_jobs=-1)
    nn.fit(external_matrix)

    train_dist, train_idx = nn.kneighbors(train_matrix)
    test_dist, test_idx = nn.kneighbors(test_matrix)

    train_feat = pd.DataFrame(
        {
            "nn1_pred": y_external.values[train_idx[:, 0]],
            "nn3_pred": weighted_from_neighbors(train_dist[:, :3], train_idx[:, :3], y_external, 1.0),
            "nn5_pred": weighted_from_neighbors(train_dist[:, :5], train_idx[:, :5], y_external, 1.0),
            "nn3_pow2": weighted_from_neighbors(train_dist[:, :3], train_idx[:, :3], y_external, 2.0),
            "nn5_pow2": weighted_from_neighbors(train_dist[:, :5], train_idx[:, :5], y_external, 2.0),
            "d1": train_dist[:, 0],
            "d2": train_dist[:, 1],
            "d3": train_dist[:, 2],
            "d5_mean": train_dist.mean(axis=1),
            "d5_std": train_dist.std(axis=1),
        }
    )
    test_feat = pd.DataFrame(
        {
            "nn1_pred": y_external.values[test_idx[:, 0]],
            "nn3_pred": weighted_from_neighbors(test_dist[:, :3], test_idx[:, :3], y_external, 1.0),
            "nn5_pred": weighted_from_neighbors(test_dist[:, :5], test_idx[:, :5], y_external, 1.0),
            "nn3_pow2": weighted_from_neighbors(test_dist[:, :3], test_idx[:, :3], y_external, 2.0),
            "nn5_pow2": weighted_from_neighbors(test_dist[:, :5], test_idx[:, :5], y_external, 2.0),
            "d1": test_dist[:, 0],
            "d2": test_dist[:, 1],
            "d3": test_dist[:, 2],
            "d5_mean": test_dist.mean(axis=1),
            "d5_std": test_dist.std(axis=1),
        }
    )
    return train_feat, test_feat


def main():
    print("External 1NN retrieval pipeline basliyor...", flush=True)

    train_df = pd.read_csv("train_temiz.csv")
    test_df = pd.read_csv("test_temiz.csv")
    test_id = get_test_id(test_df)
    external_df = build_external_aligned_frame()

    x_train, x_test, y = build_feature_space(train_df, test_df)
    x_external, _, y_external = build_feature_space(
        external_df, external_df.drop(columns=[TARGET]).copy()
    )

    train_retrieval, test_retrieval = build_retrieval_features(
        x_train, x_test, x_external, y_external
    )

    print(f"Direct nn1 RMSE: {rmse(y, train_retrieval['nn1_pred']):.6f}", flush=True)
    print(f"Direct nn3 RMSE: {rmse(y, train_retrieval['nn3_pred']):.6f}", flush=True)
    print(f"Direct nn5 RMSE: {rmse(y, train_retrieval['nn5_pred']):.6f}", flush=True)

    all_cols = list(train_retrieval.columns)
    y_bins = pd.qcut(y, q=12, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    oof = np.zeros(len(train_retrieval), dtype=np.float32)
    for train_idx, val_idx in skf.split(train_retrieval, y_bins):
        model = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-6, 3, 20)),
        )
        model.fit(train_retrieval.iloc[train_idx][all_cols], y.iloc[train_idx])
        oof[val_idx] = model.predict(train_retrieval.iloc[val_idx][all_cols])

    print(f"Ridge-calibrated retrieval RMSE: {rmse(y, oof):.6f}", flush=True)

    final_model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-6, 3, 20)),
    )
    final_model.fit(train_retrieval[all_cols], y)
    final_pred = np.clip(final_model.predict(test_retrieval[all_cols]), 0.0, 10.0)
    pure_pred = np.clip(test_retrieval["nn1_pred"].values, 0.0, 10.0)

    pd.DataFrame({"id": test_id, TARGET: final_pred}).to_csv(
        "submission_EXTERNAL_1NN_RIDGE.csv", index=False
    )
    pd.DataFrame({"id": test_id, TARGET: pure_pred}).to_csv(
        "submission_EXTERNAL_1NN_PURE.csv", index=False
    )
    print("Kaydedildi: submission_EXTERNAL_1NN_RIDGE.csv", flush=True)
    print("Kaydedildi: submission_EXTERNAL_1NN_PURE.csv", flush=True)


if __name__ == "__main__":
    main()
