import polars as pl

ROOT = "data/open_h/Surgical/jhu/imerse/srt_tissue_lift"
df = pl.read_parquet(f"{ROOT}/data/chunk-000/episode_000000.parquet")

print("Shape: ", df.shape)                        # rows = frames of this episode
print("\n")
print("Columns: ",df.columns)                      # what is stored per frame
print("\n")
print("Instruction text: ", df["instruction.text"][0])       # the text
print("\n")
print("Observation state: ", df["observation.state"][0])      # 16 numbers
print("\n")
print("Action: ", df["action"][0])                 # 16 numbers
