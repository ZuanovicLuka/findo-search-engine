import json
import os
import sqlite3
import time

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm import tqdm


def arrow_to_sqlite(arrow_path: str, sqlite_path: str, batch_size: int = 1000):
    """Transforms .arrow into sqlite database"""

    start_time = time.time()
    print(f"\nStarting conversion from {arrow_path} => {sqlite_path}")

    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            title TEXT,
            text TEXT,
            out_links TEXT,
            redirect INTEGER
        )
    """
    )

    inserted = 0
    next_id = 1

    def process_batch(batch: pa.RecordBatch):
        nonlocal inserted, next_id

        cols = {name: batch.column(name).to_pylist() for name in batch.schema.names}
        titles = cols.get("title", [])
        texts = cols.get("text", [])
        out_links = cols.get("out_links", [None] * len(titles))
        redirects = cols.get("redirect", [0] * len(titles))

        rows = [
            (
                next_id + i,
                titles[i],
                texts[i],
                json.dumps(out_links[i]) if out_links[i] else None,
                int(redirects[i] or 0),
            )
            for i in range(len(titles))
        ]
        next_id += len(rows)
        cur.executemany(
            "INSERT INTO documents (id, title, text, out_links, redirect) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        inserted += len(rows)

        if inserted % (batch_size * 10) == 0:
            conn.commit()

    if os.path.isdir(arrow_path):
        dataset = ds.dataset(arrow_path)
        scanner = dataset.scanner(columns=["title", "text", "out_links", "redirect"])
        print("Reading")
        batch_count = 0
        for batch in tqdm(scanner.to_batches(), desc="Converting", unit="batch"):
            process_batch(batch)
            batch_count += 1
        print(f"Processed {batch_count} batches in total")
    else:
        if arrow_path.endswith(".parquet"):
            parquet_file = pq.ParquetFile(arrow_path)
            total_row_groups = parquet_file.num_row_groups
            print(f"Parquet file with {total_row_groups} row groups.")
            for rg in tqdm(range(total_row_groups), desc="Converting", unit="group"):
                batch_table = parquet_file.read_row_group(
                    rg, columns=["title", "text", "out_links", "redirect"]
                )
                process_batch(batch_table.to_batches()[0])
        else:
            with pa.memory_map(arrow_path, "r") as source:
                reader = pa.ipc.open_file(source)
                total_batches = reader.num_record_batches
                print(f"Arrow file with {total_batches} record batches.")
                for i in tqdm(range(total_batches), desc="Converting", unit="batch"):
                    batch = reader.get_batch(i)
                    process_batch(batch)

    conn.commit()
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nDone! Inserted {inserted:,} rows into {sqlite_path}")
    print(f"Total time: {elapsed:.2f}s ({inserted / elapsed:.1f} rows/sec)")
    conn.close()
    print("SQLite connection closed")


if __name__ == "__main__":
    arrow_to_sqlite(
        "data/ptwiki-articles-with-redirects.arrow", "demo-index/database.db", batch_size=1000
    )
