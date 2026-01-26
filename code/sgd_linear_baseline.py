import os
# 线程收敛：避免多库多线程 + 内存抖动导致黑屏
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "2"

from pathlib import Path
import numpy as np
import polars as pl
import pyarrow.parquet as pq

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor


# ======================
# paths & config
# ======================
BASE_DIR = Path(r"E:\work\wenbo")
DATASET_PATH = BASE_DIR / "data" / "used_dataset_filter.parquet"
SUMMARY_DIR = BASE_DIR / "output" / "ic_summary"
OUT_DIR = BASE_DIR / "output" / "model_pred_sgd"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "TARGET_RETURN_15"

# 文档要求：train < 2024-09-01, test >= 2024-09-01
TRAIN_END = pl.datetime(2024, 9, 1, time_zone="UTC")

# 可选：如果你仍然卡，就只评估更近段（不影响合规）
PRED_START_STR = "2025-05-01"  # 不想限制就设为 None
PRED_START = None if PRED_START_STR is None else pl.datetime(2025, 5, 1, time_zone="UTC")

# SGD：ElasticNet 最稳；想要 Ridge 等价就改成 penalty="l2"
SGD_PARAMS = dict(
    loss="squared_error",
    penalty="elasticnet",   # "l2" / "l1" / "elasticnet"
    alpha=1e-5,
    l1_ratio=0.2,
    learning_rate="invscaling",
    eta0=0.01,
    power_t=0.25,
    max_iter=1,             # 通过 partial_fit 多次喂数据
    tol=None,
    fit_intercept=True,
    random_state=42,
)

PRED_FILE_PREFIX = "sgd_pred_chunk"


# ======================
# 1) read factor list (33)
# ======================
final_candidates = pl.read_parquet(SUMMARY_DIR / "final_candidates.parquet")
feat_list = final_candidates["feat_name"].to_list()
print(f"[INFO] Using {len(feat_list)} factors")

need_cols = ["symbol", "trade_time", TARGET] + feat_list


# ======================
# 2) parquet row-group streaming iterator
# ======================
pf = pq.ParquetFile(str(DATASET_PATH))
n_rg = pf.num_row_groups
print(f"[INFO] Parquet row groups: {n_rg}")

def iter_rowgroups():
    for rg in range(n_rg):
        table = pf.read_row_group(rg, columns=need_cols)
        yield rg, table


# ======================
# 3) polars split by time (NO pyarrow time compare)
# ======================
def split_df_by_time(df: pl.DataFrame):
    # 统一 trade_time dtype：ms + UTC
    df = df.with_columns(
        pl.col("trade_time")
        .dt.cast_time_unit("ms")
        .dt.replace_time_zone("UTC")
    )

    train_df = df.filter(pl.col("trade_time") < TRAIN_END)
    test_df = df.filter(pl.col("trade_time") >= TRAIN_END)

    if PRED_START is not None:
        test_df = test_df.filter(pl.col("trade_time") >= PRED_START)

    return train_df, test_df


def df_to_xy(df: pl.DataFrame):
    df = df.drop_nulls([TARGET] + feat_list)
    if df.height == 0:
        return None, None, None

    X = df.select(feat_list).to_numpy()
    y = df[TARGET].to_numpy()
    meta = df.select(["symbol", "trade_time"])
    return X, y, meta


# ======================
# 4) Pass 1: fit scaler on TRAIN (streaming)
# ======================
scaler = StandardScaler(with_mean=True, with_std=True)

print("[INFO] Pass 1: fitting scaler on TRAIN (streaming)...")
train_rows = 0

for rg, table in iter_rowgroups():
    df = pl.from_arrow(table)
    train_df, _ = split_df_by_time(df)

    X, y, meta = df_to_xy(train_df)
    if X is None:
        continue

    scaler.partial_fit(X)
    train_rows += X.shape[0]

    if rg % 20 == 0:
        print(f"  scaler pass rg={rg}/{n_rg-1}, train_rows={train_rows}")

print(f"[INFO] Scaler fitted. total train rows used: {train_rows}")


# ======================
# 5) Pass 2: train SGDRegressor on TRAIN (streaming)
# ======================
model = SGDRegressor(**SGD_PARAMS)

print("[INFO] Pass 2: training SGDRegressor on TRAIN (streaming)...")
train_rows2 = 0

for rg, table in iter_rowgroups():
    df = pl.from_arrow(table)
    train_df, _ = split_df_by_time(df)

    X, y, meta = df_to_xy(train_df)
    if X is None:
        continue

    Xs = scaler.transform(X)
    model.partial_fit(Xs, y)
    train_rows2 += Xs.shape[0]

    if rg % 20 == 0:
        wnorm = float(np.linalg.norm(model.coef_))
        print(f"  train pass rg={rg}/{n_rg-1}, rows={train_rows2}, coef_norm={wnorm:.4f}")

print(f"[INFO] SGD training finished. total train rows used: {train_rows2}")
print(f"[INFO] coef_norm={float(np.linalg.norm(model.coef_)):.4f}")


# ======================
# 6) Pass 3: predict on TEST (streaming) and write chunks
# ======================
print("[INFO] Pass 3: predicting on TEST (streaming) and writing chunks...")

chunk_idx = 0
test_rows = 0
written = 0

for rg, table in iter_rowgroups():
    df = pl.from_arrow(table)
    _, test_df = split_df_by_time(df)

    X, y, meta = df_to_xy(test_df)
    if X is None:
        continue

    Xs = scaler.transform(X)
    pred = model.predict(Xs)

    out = meta.with_columns([
        pl.Series("pred", pred),
        pl.Series("target", y),
    ])

    out_path = OUT_DIR / f"{PRED_FILE_PREFIX}_{chunk_idx:05d}.parquet"
    out.write_parquet(out_path)

    chunk_idx += 1
    test_rows += out.height
    written += 1

    if rg % 20 == 0:
        print(f"  pred pass rg={rg}/{n_rg-1}, wrote={out_path.name}, test_rows={test_rows}")

print(f"[INFO] Prediction done. chunks_written={written}, test_rows={test_rows}")
print(f"[INFO] Output dir: {OUT_DIR}")
