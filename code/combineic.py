import polars as pl
from pathlib import Path

BASE_DIR = Path(r"E:\work\wenbo")
IC_RAW_DIR = BASE_DIR / "output" / "ic_raw"


#  读回所有 batch 结果
ic_table = pl.read_parquet(IC_RAW_DIR / "ic_batch_*.parquet")

valid_ic = ic_table.filter(
    pl.col("ic_ir").is_not_nan() &
    (pl.col("ic_std") > 0)
)

print("valid factor count:", valid_ic.height)
#=======
#IC_IR排序
#=======
top_factors = (
    valid_ic
    .sort("ic_ir", descending=True)
    .select([
        "feat_name",
        "n_days",
        "ic_mean",
        "ic_std",
        "ic_ir",
        "ic_median",
        "ic_pos_ratio",
    ])
    .head(10)
)

bottom_factors = (
    valid_ic
    .sort("ic_ir")
    .select([
        "feat_name",
        "n_days",
        "ic_mean",
        "ic_std",
        "ic_ir",
        "ic_median",
        "ic_pos_ratio",
    ])
    .head(10)
)

print(top_factors)
print(bottom_factors)


