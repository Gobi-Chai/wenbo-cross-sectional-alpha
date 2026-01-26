import polars as pl
from pathlib import Path

DATA_DIR = Path(r"E:\Work\wenbo")  #径

DATASET_PATH = DATA_DIR / "used_dataset_filter.parquet"
WEIGHT_PATH = DATA_DIR / "used_weight"

# lazy scan
ds = pl.scan_parquet(DATASET_PATH)
wt = pl.scan_ipc(WEIGHT_PATH)

# schema
# print("=== Dataset schema ===")
# print(ds.schema)

print("\n=== Weight schema ===")
print(wt.schema)

# FEAT 列
feat_cols = [c for c in ds.columns if c.startswith("FEAT")]
print("\n#FEAT columns:", len(feat_cols))

# 抽样
print(
    ds.select(
        ["symbol", "trade_time"]
        + feat_cols[:5]
        + ["TARGET_RETURN_15"]
    )
    .limit(5)
    .collect()
)

# trade_time dtype
print("\ntrade_time dtype:", ds.schema.get("trade_time"))
