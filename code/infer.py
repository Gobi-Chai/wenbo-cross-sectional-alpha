import argparse
from pathlib import Path
import os

# 防止多线程导致资源占用过高
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "2"

import polars as pl
import numpy as np
import joblib


def main():
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("--test_ipc", type=str, required=True, help="Path to test ipc file")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing model artifacts")
    parser.add_argument("--output", type=str, required=True, help="Output parquet path")

    args = parser.parse_args()

    test_ipc = Path(args.test_ipc)
    model_dir = Path(args.model_dir)
    output_path = Path(args.output)

    # ===== load model artifacts =====
    model = joblib.load(model_dir / "sgd_model.pkl")
    scaler = joblib.load(model_dir / "scaler.pkl")
    feat_list = joblib.load(model_dir / "feature_list.pkl")

    # ===== load test data =====
    df = pl.read_ipc(test_ipc)

    # dtype 对齐
    df = df.with_columns(
        pl.col("symbol").cast(pl.Utf8),
        pl.col("trade_time")
        .dt.cast_time_unit("ms")
        .dt.replace_time_zone("UTC")
    )

    # 只取需要的列
    df_feat = df.select(["symbol", "trade_time"] + feat_list)

    # 丢弃缺失
    df_feat = df_feat.drop_nulls(feat_list)

    if df_feat.height == 0:
        raise RuntimeError("No valid rows after drop_nulls")

    # ===== inference =====
    X = df_feat.select(feat_list).to_numpy()
    Xs = scaler.transform(X)
    pred = model.predict(Xs)

    # ===== output =====
    out = df_feat.select(["symbol", "trade_time"]).with_columns(
        pl.Series("pred", pred)
    )

    out.write_parquet(output_path)
    print(f"[INFO] inference finished, saved to {output_path}, shape={out.shape}")


if __name__ == "__main__":
    main()
