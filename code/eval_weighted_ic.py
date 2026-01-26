import polars as pl
from pathlib import Path
import numpy as np

BASE_DIR = Path(r"E:\work\wenbo")
PRED_DIR = BASE_DIR / "output" / "model_pred_sgd"
WEIGHT_PATH = BASE_DIR / "data" / "used_weight"

# lazy scan
pred = pl.scan_parquet(PRED_DIR / "sgd_pred_chunk_*.parquet")
weight = pl.scan_ipc(WEIGHT_PATH)

# ===== 关键修复 1：统一 symbol dtype =====
pred = pred.with_columns(
    pl.col("symbol").cast(pl.Utf8)
)

weight = weight.with_columns(
    pl.col("symbol").cast(pl.Utf8)
)

# ===== 关键修复 2：统一 trade_time dtype =====
pred = pred.with_columns(
    pl.col("trade_time")
    .dt.cast_time_unit("ms")
    .dt.replace_time_zone("UTC")
)

weight = weight.with_columns(
    pl.col("trade_time")
    .dt.cast_time_unit("ms")
    .dt.replace_time_zone("UTC")
)

# join 权重
df = (
    pred
    .join(
        weight,
        on=["trade_time", "symbol"],
        how="inner"
    )
    .filter(pl.col("joint_weight") > 0)
)

# ===== 加权 IC（按日）=====
ic_ts = (
    df.group_by("trade_time")
    .agg([
        (
            (pl.col("joint_weight") * pl.col("pred") * pl.col("target")).sum()
            - (pl.col("joint_weight") * pl.col("pred")).sum()
              * (pl.col("joint_weight") * pl.col("target")).sum()
              / pl.col("joint_weight").sum()
        ).alias("cov_xy"),

        (
            (pl.col("joint_weight") * pl.col("pred") * pl.col("pred")).sum()
            - (pl.col("joint_weight") * pl.col("pred")).sum() ** 2
              / pl.col("joint_weight").sum()
        ).alias("var_x"),

        (
            (pl.col("joint_weight") * pl.col("target") * pl.col("target")).sum()
            - (pl.col("joint_weight") * pl.col("target")).sum() ** 2
              / pl.col("joint_weight").sum()
        ).alias("var_y"),
    ])
    .with_columns(
        (pl.col("cov_xy") / (pl.col("var_x") * pl.col("var_y")).sqrt())
        .alias("w_ic")
    )
    .select("trade_time", "w_ic")
    .collect()
    .drop_nulls()
)

w_ic = ic_ts["w_ic"].to_numpy()

print("=== Weighted IC ===")
print("n_days:", w_ic.size)
print("mean  :", float(w_ic.mean()) if w_ic.size else np.nan)
print("std   :", float(w_ic.std(ddof=1)) if w_ic.size > 1 else np.nan)
print("IR    :", float(w_ic.mean() / (w_ic.std(ddof=1) + 1e-6)) if w_ic.size > 1 else np.nan)
print("pos_ratio:", float((w_ic > 0).mean()) if w_ic.size else np.nan)
