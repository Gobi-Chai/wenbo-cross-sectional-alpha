import polars as pl
from pathlib import Path

BASE_DIR = Path(r"E:\work\wenbo")

IC_RAW_DIR = BASE_DIR / "output" / "ic_raw"
IC_SUMMARY_DIR = BASE_DIR / "output" / "ic_summary"
IC_SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

# 1 读所有 Raw IC batch
raw_ic = pl.read_parquet(IC_RAW_DIR / "ic_batch_*.parquet")

print("raw ic shape:", raw_ic.shape)  # 应该是 (175, 7)

#  写成单一汇总文件
raw_ic_path = IC_SUMMARY_DIR / "single_factor_ic.parquet"
raw_ic.write_parquet(raw_ic_path)

print(f"saved raw ic summary to: {raw_ic_path}")
