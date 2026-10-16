import pandas as pd
df = pd.read_parquet("data/open_h/Surgical/jhu/imerse/star_il/star_il/data/chunk-000/episode_000000.parquet")
print(df.columns.tolist())
print(len(df), "frames")
print("state:", len(df["observation.state"].iloc[0]), "action:", len(df["action"].iloc[0]))
print(df["instruction.text"].unique() if "instruction.text" in df else "no instruction column")