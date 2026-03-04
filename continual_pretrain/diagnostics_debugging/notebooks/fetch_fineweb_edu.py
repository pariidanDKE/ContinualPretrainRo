import os
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset, Dataset

# Load HF_TOKEN from .env (two levels up: notebooks/ -> diagnostics_debugging/ -> continual_pretrain/)
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)
HF_TOKEN = os.environ["HF_TOKEN"]

# ---------- stream fineweb-edu sample-100BT ----------
N_ROWS = 500_000
HF_REPO_NAME = "fineweb-edu-500k-sample"  # change to your preferred name

print(f"Streaming HuggingFaceFW/fineweb-edu (split=sample-100BT), taking {N_ROWS:,} rows …")

streamed = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-100BT",
    split="train",
    streaming=True,
    token=HF_TOKEN,
)

# Convert streamed iterable -> concrete Dataset without holding a full list
def row_generator():
    for i, row in enumerate(streamed):
        if i >= N_ROWS:
            break
        yield row

ds = Dataset.from_generator(row_generator)
print(f"Collected {len(ds):,} rows — columns: {ds.column_names}")

# ---------- push to your HF account ----------
print(f"Pushing to hub as {HF_REPO_NAME} …")
ds.push_to_hub(HF_REPO_NAME, token=HF_TOKEN, private=True)
print("Done.")
