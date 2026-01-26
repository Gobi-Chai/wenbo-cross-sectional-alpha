import os
os.environ["POLARS_MAX_THREADS"] = "4"

import polars as pl
from pathlib import Path

# ======================
# paths & config
# ======================
BASE_DIR = Path(r"E:\work\wenbo")
DATASET_PATH = BASE_DIR / "data" / "used_dataset_filter.parquet"
IC_RAW_DIR = BASE_DIR / "output" / "ic_raw"
IC_SUMMARY_DIR = BASE_DIR / "output" / "ic_summary"

IC_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "TARGET_RETURN_15"
BATCH_SIZE = 5

# ======================
#  读取 Raw IC，拿有效因子
# ======================
ic_table = pl.read_parquet(IC_RAW_DIR / "ic_batch_*.parquet")

valid_feats = (
    ic_table
    .filter(pl.col("ic_ir").is_not_nan() & (pl.col("ic_std") > 0))
    .select("feat_name")
    .to_series()
    .to_list()
)

print("valid factor count:", len(valid_feats))

# +=====================
# lazy scan + 时间锁定
# ======================
ds = (
    pl.scan_parquet(DATASET_PATH)
    .with_columns(
        pl.col("trade_time")
        .dt.cast_time_unit("ms")
        .dt.replace_time_zone("UTC")
    )
    .filter(
        pl.col("trade_time")
        >= pl.lit("2025-01-01").str.to_datetime(time_zone="UTC")
    )
    .select(["trade_time", TARGET] + valid_feats)
)

# ======================
# 3️⃣ RankIC 扫描（batch 计算，但不落 batch 文件）
# ======================
all_stats = []

for i in range(0, len(valid_feats), BATCH_SIZE):
    batch_feats = valid_feats[i:i + BATCH_SIZE]
    print(f"RankIC batch {i} ~ {i + len(batch_feats) - 1}")

    rank_ic_ts = (
        ds
        .group_by("trade_time")
        .agg([
            pl.corr(
                pl.col(feat).rank(),
                pl.col(TARGET).rank()
            ).alias(feat)
            for feat in batch_feats
        ])
    )

    for feat in batch_feats:
        stat = (
            rank_ic_ts
            .select([
                pl.lit(feat).alias("feat_name"),
                pl.col(feat).count().alias("n_days"),
                pl.col(feat).mean().alias("rank_ic_mean"),
                pl.col(feat).std().alias("rank_ic_std"),
                (pl.col(feat).mean() / pl.col(feat).std()).alias("rank_ic_ir"),
                (pl.col(feat) > 0)
                .cast(pl.Float32)
                .mean()
                .alias("rank_ic_pos_ratio"),
            ])
        )
        all_stats.append(stat)

# ======================
# 4️ 一次性 collect（只有 ~148 行，非常安全）
# ======================
rank_ic_table = pl.concat(all_stats).collect()

# ======================
# 5️写单一结果文件
# ======================
OUTPUT_PATH = IC_SUMMARY_DIR / "single_factor_rank_ic.parquet"
rank_ic_table.write_parquet(OUTPUT_PATH)

print("=== RankIC FINISHED ===")
print("result shape:", rank_ic_table.shape)
print(f"saved to: {OUTPUT_PATH}")
