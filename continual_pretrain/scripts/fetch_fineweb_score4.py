import json
import urllib.request
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / ".env")
HF_TOKEN = os.environ["HF_TOKEN"]

import duckdb

DATASET = "OpenLLM-Ro/fineweb2-ro-bert"
OUTPUT = Path(__file__).parent.parent / "data" / "fineweb2_ro_score4.parquet"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

print("Fetching parquet file URLs...")
resp = urllib.request.urlopen(
    f"https://datasets-server.huggingface.co/parquet?dataset={DATASET}"
)
urls = [f["url"] for f in json.load(resp)["parquet_files"]]
print(f"Found {len(urls)} parquet files")

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute(f"CREATE SECRET (TYPE HTTP, BEARER_TOKEN '{HF_TOKEN}');")
con.execute("SET threads = 8;")
con.execute("SET memory_limit='8GB';")
con.execute("SET temp_directory='/tmp/duckdb_spill';")

print("Querying score=4 rows across all files (this will take a while)...")
con.execute(f"""
COPY (
    SELECT * FROM read_parquet({urls})
    WHERE int_score = 4
) TO '{OUTPUT}' (FORMAT PARQUET, COMPRESSION ZSTD)
""")

result = con.execute(f"SELECT COUNT(*) FROM read_parquet('{OUTPUT}')").fetchone()
print(f"Done. Saved {result[0]:,} rows to {OUTPUT}")
