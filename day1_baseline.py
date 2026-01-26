import polars as pl
from pathlib import Path

# =====================
# paths & load
# =====================
DATA_DIR = Path(r"E:\work\wenbo")
DATASET_PATH = DATA_DIR / "used_dataset_filter.parquet"

# ds = pl.scan_parquet(DATASET_PATH)

ds = (
    pl.scan_parquet(DATASET_PATH)
    .filter(pl.col("trade_time") >= pl.datetime(2024, 6, 1))
)


TARGET = "TARGET_RETURN_15"

# =====================
# 自动识别 FEAT 列
# =====================
feat_cols = [c for c in ds.columns if c.startswith("FEAT")]

# =====================
# 单因子 IC 扫描
# =====================
results = []

for feat in feat_cols:
    ic_stat = (
        ds
        .select([
            "trade_time",
            pl.col(feat).alias("pred"),
            pl.col(TARGET).alias("target"),
        ])
        .group_by("trade_time")
        .agg(
            pl.corr("pred", "target").alias("ic")
        )
        .select([
            pl.lit(feat).alias("feat_name"),
            pl.col("ic").count().alias("n_days"),
            pl.col("ic").mean().alias("ic_mean"),
            pl.col("ic").std().alias("ic_std"),
            (pl.col("ic").mean() / pl.col("ic").std()).alias("ic_ir"),
            pl.col("ic").median().alias("ic_median"),
            (pl.col("ic") > 0).cast(pl.Float32).mean().alias("ic_pos_ratio"),
        ])

    )

    results.append(ic_stat)
# =====================
# 触发执行（只 collect 175 行）
# =====================
ic_table = pl.concat(results).collect()






