from pathlib import Path
import polars as pl

DATA_DIR = Path(r"E:\Work\wenbo")
WEIGHT_PATH = DATA_DIR / "used_weight"

wt = pl.scan_ipc(WEIGHT_PATH)

print("=== Weight schema ===")
print(wt.schema)
