import os
os.environ["POLARS_MAX_THREADS"] = "4"

import polars as pl
from pathlib import Path

# ======================
# paths & config
# ======================
BASE_DIR = Path(r"E:\work\wenbo")
DATASET_PATH = BASE_DIR / "data" / "used_dataset_filter.parquet"
SUMMARY_DIR = BASE_DIR / "output" / "ic_summary"

START_DATE = "2025-01-01"
CORR_THRESHOLD = 0.7
# ======================

# ======================
# 1️⃣ 读取 final candidates
# ======================
final_candidates = pl.read_parquet(
    SUMMARY_DIR / "final_candidates.parquet"
)

feat_list = (
    final_candidates
    .select("feat_name")
    .to_series()
    .to_list()
)

print(f"Loaded final candidates: {len(feat_list)} factors")

# ======================
# 2️⃣ 读取数据 + 横截面 rank（关键降维）
# ======================
print("Computing daily cross-sectional ranks...")

rank_df = (
    pl.scan_parquet(DATASET_PATH)
    .with_columns(
        pl.col("trade_time")
        .dt.cast_time_unit("ms")
        .dt.replace_time_zone("UTC")
    )
    .filter(
        pl.col("trade_time")
        >= pl.lit(START_DATE).str.to_datetime(time_zone="UTC")
    )
    .select(["trade_time"] + feat_list)
    .group_by("trade_time")
    .agg([
        pl.col(feat).rank(method="average").alias(feat)
        for feat in feat_list
    ])
    .collect()   # ⚠️ 这里 collect，但数据已极度压缩
)

print("Rank table shape:", rank_df.shape)
# 大概是 (交易日数, 33)

# ======================
# 3️⃣ 计算因子相关矩阵（小表，安全）
# ======================
print("Computing factor correlation matrix...")

corr_mat = rank_df.select([
    pl.corr(pl.col(f1), pl.col(f2)).alias(f"{f1}__{f2}")
    for i, f1 in enumerate(feat_list)
    for f2 in feat_list[i + 1:]
])

corr_df = corr_mat.collect()

# ======================
# 4️⃣ 整理成 long format
# ======================
corr_pairs = (
    corr_df
    .melt(variable_name="pair", value_name="corr")
    .with_columns([
        pl.col("pair").str.split("__").list.get(0).alias("feat_1"),
        pl.col("pair").str.split("__").list.get(1).alias("feat_2"),
    ])
    .select("feat_1", "feat_2", "corr")
)

# ======================
# 5️⃣ 筛选高度相关对
# ======================
high_corr = (
    corr_pairs
    .filter(pl.col("corr").abs() >= CORR_THRESHOLD)
    .sort("corr", descending=True)
)

print("Highly correlated pairs:", high_corr.height)

# ======================
# 6️⃣ join 因子质量信息
# ======================
dedup_table = (
    high_corr
    .join(
        final_candidates.select(
            "feat_name", "ic_ir", "rank_ic_ir", "rank_ic_pos_ratio"
        ),
        left_on="feat_1",
        right_on="feat_name",
        how="left"
    )
    .rename({
        "ic_ir": "ic_ir_1",
        "rank_ic_ir": "rank_ic_ir_1",
        "rank_ic_pos_ratio": "rank_ic_pos_ratio_1",
    })
    .drop("feat_name")
    .join(
        final_candidates.select(
            "feat_name", "ic_ir", "rank_ic_ir", "rank_ic_pos_ratio"
        ),
        left_on="feat_2",
        right_on="feat_name",
        how="left"
    )
    .rename({
        "ic_ir": "ic_ir_2",
        "rank_ic_ir": "rank_ic_ir_2",
        "rank_ic_pos_ratio": "rank_ic_pos_ratio_2",
    })
    .drop("feat_name")
)

# ======================
# 7️⃣ 保存
# ======================
OUT_PATH = SUMMARY_DIR / "factor_corr_dedup_table.parquet"
dedup_table.write_parquet(OUT_PATH)

print("=== DEDUP FINISHED ===")
print("saved to:", OUT_PATH)

print(
    dedup_table
    .select(
        "feat_1", "feat_2", "corr",
        "ic_ir_1", "rank_ic_ir_1",
        "ic_ir_2", "rank_ic_ir_2"
    )
    .head(10)
)
