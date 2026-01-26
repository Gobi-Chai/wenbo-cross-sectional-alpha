import os
os.environ["POLARS_MAX_THREADS"] = "4"

import polars as pl
from pathlib import Path

# =====================
# paths & config
# =====================
DATA_DIR = Path(r"E:\work\wenbo")
DATASET_PATH = DATA_DIR / "used_dataset_filter.parquet"

TARGET = "TARGET_RETURN_15"
BATCH_SIZE = 5        # 核心参数：控制内存（10~20 都安全）

# =====================
# lazy scan + time filter
# =====================
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
)

# 2️⃣ 再从 ds 里取 FEAT 列名
all_feats = [c for c in ds.columns if c.startswith("FEAT")]

# 3️⃣ 再 select（可选，但你现在已经写了）
ds = ds.select(["trade_time", TARGET] + all_feats)

# 4️⃣ feat_cols 其实和 all_feats 一样，用一个就够
feat_cols = all_feats

# =====================
# batch IC scan
# =====================


for i in range(0, len(feat_cols), BATCH_SIZE):
    batch_feats = feat_cols[i:i + BATCH_SIZE]

    print(f"Processing batch {i} ~ {i + len(batch_feats) - 1}")

    # 一次 group_by，只算这一批 FEAT
    ic_ts = (
        ds
        .group_by("trade_time")
        .agg([
            pl.corr(pl.col(feat), pl.col(TARGET)).alias(feat)
            for feat in batch_feats
        ])
    )

    # 每个 FEAT 立刻压缩成 1 行
    batch_stats = []

    for feat in batch_feats:
        stat = (
            ic_ts
            .select([
                pl.lit(feat).alias("feat_name"),
                pl.col(feat).count().alias("n_days"),
                pl.col(feat).mean().alias("ic_mean"),
                pl.col(feat).std().alias("ic_std"),
                (pl.col(feat).mean() / pl.col(feat).std()).alias("ic_ir"),
                pl.col(feat).median().alias("ic_median"),
                (pl.col(feat) > 0).cast(pl.Float32).mean().alias("ic_pos_ratio"),
            ])
        )
        batch_stats.append(stat)
    batch_df = pl.concat(batch_stats).collect()
    batch_df.write_parquet(DATA_DIR / f"ic_batch_{i}.parquet")
    print(f"Saved batch {i} ~ {i + len(batch_feats) - 1}")
    

# =====================
# final collect（≈175 行）
# =====================
ic_table = pl.read_parquet(DATA_DIR / "ic_batch_*.parquet")

print("=== IC SCAN FINISHED ===")
print("result shape:", ic_table.shape)

