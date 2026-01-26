import polars as pl
from pathlib import Path

BASE_DIR = Path(r"E:\work\wenbo")
PRED_DIR = BASE_DIR / "output" / "model_pred_sgd"

pred = (
    pl.scan_parquet(PRED_DIR / "sgd_pred_chunk_*.parquet")
    .select(["symbol", "trade_time", "pred"])
    .collect()
)

pred.write_parquet(BASE_DIR / "submission_pred.parquet")
print("saved submission_pred.parquet, shape:", pred.shape)
