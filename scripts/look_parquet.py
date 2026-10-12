import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = "data/open_h/Surgical/jhu/imerse/srt_tissue_lift"
PATH = f"{ROOT}/data/chunk-000/episode_000000.parquet"

# --- A: the schema, read without loading the data ---
print(pq.read_schema(PATH))

# --- B: load the table ---
df = pd.read_parquet(PATH)
print("shape:", df.shape)
print(df.dtypes)

# --- C: the five bookkeeping columns ---
book = ["index", "episode_index", "frame_index", "timestamp", "task_index"]
print(df[book].head())
print(df[book].tail())

# --- D: one row, up close ---
row = df.iloc[0]
print("action type:", type(row["action"]), "length:", len(row["action"]))
print("text:", row["instruction.text"])
print("tools:", row["observation.meta.tool.psm1"], "|", row["observation.meta.tool.psm2"])

# --- E: from a column of arrays to a matrix ---
actions = np.stack(df["action"].to_numpy())
states  = np.stack(df["observation.state"].to_numpy())
print("actions:", actions.shape, "| states:", states.shape)

# --- F: sanity checks ---
print("rows == length in episodes.jsonl:", len(df) == 122)
print("frame_index is 0..N-1:", (df["frame_index"].to_numpy() == np.arange(len(df))).all())
print("time step ≈ 1/30 s:", np.allclose(np.diff(df["timestamp"]), 1 / 30, atol=1e-3))

# --- G: is the action a target the state reaches later? ---
err_same = np.abs(actions[:-1] - states[:-1]).mean()   # action t vs state t
err_next = np.abs(actions[:-1] - states[1:]).mean()    # action t vs state t+1
print(f"|action_t - state_t|   = {err_same:.6f}")
print(f"|action_t - state_t+1| = {err_next:.6f}")