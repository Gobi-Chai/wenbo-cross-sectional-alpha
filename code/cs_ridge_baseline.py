import os
# 线程收敛：防止 CPU 打满导致系统卡死
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import polars as pl
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge

# ======================
# paths & config
# ======================
BASE_DIR = Path(r"E:\work\wenbo")
DATASET_PATH = BASE_DIR / "data" / "used_dataset_filter.parquet"
SUMMARY_DIR = BASE_DIR / "output" / "ic_summary"
OUT_DIR = BASE_DIR / "output" / "model_pred"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "TARGET_RETURN_15"
TRAIN_END = "2024-09-01"
RIDGE_ALPHA = 1.0          # baseline，不调参
CHUNK_DAYS = 5             # 控制内存：按天分块处理

# ======================
# helpers
# ======================
def zscore_inplace(X: np.ndarray) -> np.ndarray:
    """按列 zscore（横截面），避免除零"""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    return (X - mu) / (sd + 1e-6)

def cs_ic_stats(df: pl.DataFrame, pred_col="pred", y_col="target") -> dict:
    ic_ts = (
        df.group_by("trade_time")
        .agg(pl.corr(pred_col, y_col).alias("ic"))
        .drop_nulls()
    )
    ic = ic_ts["ic"].to_numpy()
    if ic.size == 0:
        return {"n_days": 0, "ic_mean": np.nan, "ic_std": np.nan, "ic_ir": np.nan, "ic_pos_ratio": np.nan}

    return {
        "n_days": int(ic.size),
        "ic_mean": float(ic.mean()),
        "ic_std": float(ic.std(ddof=1)) if ic.size > 1 else 0.0,
        "ic_ir": float(ic.mean() / (ic.std(ddof=1) + 1e-6)) if ic.size > 1 else np.nan,
        "ic_pos_ratio": float((ic > 0).mean()),
    }

def global_ic(df: pl.DataFrame, pred_col="pred", y_col="target") -> float:
    x = df[pred_col].to_numpy()
    y = df[y_col].to_numpy()
    if x.size < 2:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

# ======================
# 1️⃣ 读取 33 个因子列表
# ======================
final_candidates = pl.read_parquet(SUMMARY_DIR / "final_candidates.parquet")
feat_list = final_candidates["feat_name"].to_list()
print(f"Using {len(feat_list)} factors")

# ======================
# 2️⃣ 读取数据（只取需要列）
# ======================
ds = (
    pl.scan_parquet(DATASET_PATH)
    .with_columns(
        pl.col("trade_time")
        .dt.cast_time_unit("ms")
        .dt.replace_time_zone("UTC")
    )
    .select(["symbol", "trade_time", TARGET] + feat_list)
)

PRED_START = "2025-05-01"
pred_cut = pl.lit(PRED_START).str.to_datetime(time_zone="UTC")

train_cut = pl.lit(TRAIN_END).str.to_datetime(time_zone="UTC")

# 取 train/test 的日期列表（小表）
train_days = (
    ds.select("trade_time")
    .filter(pl.col("trade_time") < train_cut)
    .unique()
    .collect()
    .sort("trade_time")
)["trade_time"].to_list()
test_days = (
    ds.select("trade_time")
    .filter(
        (pl.col("trade_time") >= train_cut) &
        (pl.col("trade_time") >= pred_cut)
    )
    .unique()
    .collect()
    .sort("trade_time")
)["trade_time"].to_list()



print(f"train days: {len(train_days)} | test days: {len(test_days)}")

# ======================
# 3️⃣ 在 train 集拟合一次 Ridge（采样天数可控，避免内存爆）
# ======================
# 你可以先用较少天数快速跑通，比如 40 天；确认没问题再放大
FIT_DAYS = min(60, len(train_days))  # 先保守，后面你再调大
fit_days = train_days[-FIT_DAYS:]    # 用最近的若干天拟合（更贴近部署）

print(f"Fitting ridge on last {len(fit_days)} train days...")

# fit_df = (
#     ds.filter(pl.col("trade_time").is_in(fit_days))
#     .collect()
#     .drop_nulls([TARGET] + feat_list)
# )

fit_start = fit_days[0]
fit_end = fit_days[-1]

fit_df = (
    ds.filter(
        (pl.col("trade_time") >= fit_start) &
        (pl.col("trade_time") <= fit_end)
    )
    .collect()
)


# 注意：这里是“全样本拟合一次”，不是逐日拟合
X_fit = fit_df.select(feat_list).to_numpy()
y_fit = fit_df[TARGET].to_numpy()

# 关键：这里用“横截面 zscore”的近似替代：
# 为了不逐日处理，我们直接在整体上做一次标准化（baseline 可接受）
X_fit = zscore_inplace(X_fit)

model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
model.fit(X_fit, y_fit)

print("Ridge fitted. coef norm:", float(np.linalg.norm(model.coef_)))

# ======================
# 4️⃣ 用冻结系数生成 pred（按天分块，防内存）
# ======================
def predict_days(day_list, tag: str) -> pl.DataFrame:
    preds = []
    for i in range(0, len(day_list), CHUNK_DAYS):
        chunk_days = day_list[i:i+CHUNK_DAYS]
        print(f"[{tag}] predicting days {i} ~ {i+len(chunk_days)-1}")


        chunk_start = chunk_days[0]
        chunk_end = chunk_days[-1]      

        chunk = (
            ds.filter(
                (pl.col("trade_time") >= chunk_start) &
                (pl.col("trade_time") <= chunk_end)
            )
            .collect()
        )
        if chunk.height == 0:
            continue

        X = chunk.select(feat_list).to_numpy()
        y = chunk[TARGET].to_numpy()

        # 用同一套（整体）zscore方式，保证 train/test一致
        X = zscore_inplace(X)

        pred = model.predict(X)

        preds.append(
            chunk.select(["symbol", "trade_time"])
            .with_columns([
                pl.Series("pred", pred),
                pl.Series("target", y),
            ])
        )

    return pl.concat(preds) if preds else pl.DataFrame()

train_pred = predict_days(train_days, "train")
test_pred = predict_days(test_days, "test")

# ======================
# 5️⃣ 评估：截面 IC + 全局 IC
# ======================
train_stat = cs_ic_stats(train_pred)
test_stat = cs_ic_stats(test_pred)

print("\n=== TRAIN (Frozen Ridge) ===")
print(train_stat)
print("train global IC:", global_ic(train_pred))

print("\n=== TEST (Frozen Ridge) ===")
print(test_stat)
print("test global IC:", global_ic(test_pred))

# ======================
# 6️⃣ 保存 pred（后续做加权 IC / 加权R2）
# ======================
train_path = OUT_DIR / "frozen_ridge_train_pred.parquet"
test_path = OUT_DIR / "frozen_ridge_test_pred.parquet"
train_pred.write_parquet(train_path)
test_pred.write_parquet(test_path)

print("\nSaved:")
print(train_path)
print(test_path)
