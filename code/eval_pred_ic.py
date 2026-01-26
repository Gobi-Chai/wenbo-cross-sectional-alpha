import polars as pl
from pathlib import Path
import numpy as np

BASE_DIR = Path(r"E:\work\wenbo")
PRED_DIR = BASE_DIR / "output" / "model_pred_sgd"

pred = pl.scan_parquet(PRED_DIR / "sgd_pred_chunk_*.parquet")

# 截面 IC（按 trade_time）
ic_ts = (
    pred.group_by("trade_time")
    .agg(pl.corr("pred", "target").alias("ic"))
    .collect()
    .drop_nulls()
)

ic = ic_ts["ic"].to_numpy()
print("n_days:", ic.size)
print("ic_mean:", float(ic.mean()) if ic.size else np.nan)
print("ic_std :", float(ic.std(ddof=1)) if ic.size > 1 else np.nan)
print("ic_ir  :", float(ic.mean() / (ic.std(ddof=1) + 1e-6)) if ic.size > 1 else np.nan)
print("pos_ratio:", float((ic > 0).mean()) if ic.size else np.nan)

# 全局 IC（不分组）
global_df = pred.select(["pred", "target"]).collect().drop_nulls()
x = global_df["pred"].to_numpy()
y = global_df["target"].to_numpy()
gic = float(np.corrcoef(x, y)[0, 1]) if x.size > 1 else np.nan
print("global_ic:", gic)
