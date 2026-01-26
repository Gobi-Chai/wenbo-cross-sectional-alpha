import polars as pl
from pathlib import Path


#======
#合并ic和icrank，筛选单因子
#======
BASE_DIR = Path(r"E:\work\wenbo")
SUMMARY_DIR = BASE_DIR / "output" / "ic_summary"

raw_ic = pl.read_parquet(SUMMARY_DIR / "single_factor_ic.parquet")
rank_ic = pl.read_parquet(SUMMARY_DIR / "single_factor_rank_ic.parquet")

# 合并
ic_compare = raw_ic.join(rank_ic, on="feat_name", how="inner")

ic_valid = ic_compare.filter(
    # Raw IC 必须有效
    pl.col("ic_ir").is_not_nan() &
    (pl.col("ic_std") > 0) &

    # RankIC 也必须有效
    pl.col("rank_ic_ir").is_not_nan() &
    (pl.col("rank_ic_std") > 0)
)

# print("after validity filter:", ic_valid.height)

ic_dir_consistent = ic_valid.filter(
    pl.col("ic_mean") * pl.col("rank_ic_mean") > 0
)

# print("after direction consistency:", ic_dir_consistent.height)


ic_stable = ic_dir_consistent.filter(
    pl.col("rank_ic_pos_ratio") > 0.52
)

# print("after rank stability:", ic_stable.height)


final_candidates = (
    ic_stable
    .sort("ic_ir", descending=True)
)

# print("final candidate count:", final_candidates.height)

final_candidates.select(
    "feat_name",
    "ic_mean",
    "ic_ir",
    "rank_ic_mean",
    "rank_ic_ir",
    "rank_ic_pos_ratio"
).head(10)

print("dir consistent:", ic_dir_consistent.height)
print("rank stable:", ic_stable.height)
print("final candidates:", final_candidates.height)

final_candidates.select(
    "feat_name",
    "ic_ir",
    "rank_ic_ir",
    "rank_ic_pos_ratio"
).head(5)

FINAL_DIR = BASE_DIR / "output" / "ic_summary"
FINAL_DIR.mkdir(exist_ok=True, parents=True)

final_candidates.write_parquet(
    FINAL_DIR / "final_candidates.parquet"
)
