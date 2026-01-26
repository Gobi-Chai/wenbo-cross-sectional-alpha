import polars as pl

df = pl.read_parquet(r"E:\work\wenbo\submission_pred.parquet")
print(df.head())
print(df.schema)
