import argparse
import json
import os
import time

import pyarrow.dataset as ds

from findo.core.limit_memory import start_memory_monitor
from findo.core.logging import setup_logging
from findo.entrypoints.indexer import SPIMIIndexer
from findo.entrypoints.tokenizer import Tokenizer

setup_logging()
start_memory_monitor(show_memory_updates=True)


def save_config(args, output_dir):
    """Saving global cofiguration of indexer, in the .json in output directory"""
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config_file.json")

    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    print(f"\nConfiguration saved to {config_path}")


def print_config(args):
    print()
    print("-" * 50)
    print(f"Starting indexing of: {args.file_path}")
    print("Configuration:")
    print(f" ° Memory limit: {args.memory_limit} MB")
    print(f" °  Min term frequency: {args.min_term_frequency}")
    print(f" °  Output directory: {args.output_directory}")
    print(f" °  Inverted index format: {args.inverted_index_format}")
    print(f" °  Forward index: {'enabled' if args.forward_index else 'disabled'}")
    print("  °  Tokenizer config:")
    print(
        f"      ° Separate alphanumeric: {'enabled' if args.separate_alphanumeric else 'disabled'}"
    )
    print(f"      ° Remove numbers: {'enabled' if args.remove_numbers else 'disabled'}")
    print(f"      ° Remove URLs: {'enabled' if args.remove_urls else 'disabled'}")
    print(f"      ° Min token length: {args.min_token_length}")
    print(f"      ° Lowercase: {'enabled' if args.lowercase else 'disabled'}")
    print(f"      ° Stemmer: {'enabled' if args.stemmer else 'disabled'}")
    print(f"      ° Stopwords: {'enabled' if args.stopwords else 'disabled'}")
    print("-" * 50)


def build_term_offsets(index_path: str, out_path: str):
    """Scan final_index.jsonl and record byte offsets for each term"""
    offsets = {}
    with open(index_path, "rb") as f:
        pos = 0
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            term = next(iter(data.keys()))
            offsets[term] = pos
            pos = f.tell()

    with open(out_path, "w", encoding="utf-8") as out:
        json.dump(offsets, out)

    print(f"Saved {len(offsets):,} term offsets to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Findo Indexer CLI")
    parser.add_argument("file_path", type=str, help="Path to the file to index")
    parser.add_argument("--memory-limit", type=int, default=1800, help="Memory limit in MB")
    parser.add_argument("--min-term-frequency", type=int, default=5, help="Minimum term frequency")
    parser.add_argument(
        "--output-directory", type=str, default="demo-index", help="Output directory"
    )
    parser.add_argument(
        "--inverted-index-format", type=str, default="json", help="Inverted index format"
    )
    parser.add_argument(
        "--forward-index", dest="forward_index", action="store_true", help="Enable forward index"
    )
    parser.add_argument(
        "--no-forward-index",
        dest="forward_index",
        action="store_false",
        help="Disable forward index",
    )
    parser.set_defaults(forward_index=True)

    parser.add_argument(
        "--separate-alphanumeric",
        dest="separate_alphanumeric",
        action="store_true",
        help="Separate alphanumeric tokens",
    )
    parser.add_argument(
        "--no-separate-alphanumeric", dest="separate_alphanumeric", action="store_false"
    )
    parser.set_defaults(separate_alphanumeric=True)

    parser.add_argument(
        "--remove-numbers", dest="remove_numbers", action="store_true", help="Remove numbers"
    )
    parser.add_argument("--no-remove-numbers", dest="remove_numbers", action="store_false")
    parser.set_defaults(remove_numbers=True)

    parser.add_argument(
        "--remove-urls", dest="remove_urls", action="store_true", help="Remove URLs"
    )
    parser.add_argument("--no-remove-urls", dest="remove_urls", action="store_false")
    parser.set_defaults(remove_urls=True)

    parser.add_argument("--min-token-length", type=int, default=3, help="Minimum token length")
    parser.add_argument(
        "--lowercase", dest="lowercase", action="store_true", help="Convert text to lowercase"
    )
    parser.add_argument("--no-lowercase", dest="lowercase", action="store_false")
    parser.set_defaults(lowercase=True)

    parser.add_argument("--stemmer", dest="stemmer", action="store_true", help="Enable stemming")
    parser.add_argument("--no-stemmer", dest="stemmer", action="store_false")
    parser.set_defaults(stemmer=True)

    parser.add_argument(
        "--stopwords", dest="stopwords", type=str, default=None, help="Enter path to stopwords file"
    )

    args = parser.parse_args()
    print_config(args)
    save_config(args, args.output_directory)
    dataset = ds.dataset(args.file_path, format="arrow")

    indexer = SPIMIIndexer(
        output_dir=args.output_directory,
        memory_limit_mb=args.memory_limit,
        min_term_freq=args.min_term_frequency,
    )

    tokenizer = Tokenizer(args)

    doc_id = 1
    skipped = 0
    start_time_total = time.time()
    for i, batch in enumerate(dataset.to_batches(batch_size=1000)):
        start_time = time.time()
        text_col = batch.column("text")

        print(f"\nProcessing batch {i + 1} with {batch.num_rows} rows")
        for j, value in enumerate(text_col):
            text = value.as_py()
            if not text or not text.strip():
                skipped += 1
                doc_id += 1
                continue

            tokens = tokenizer.tokenize(text)
            indexer.add_document(doc_id, tokens)

            doc_id += 1
        end_time = time.time()
        print(f"Batch {i+1} processed in {end_time - start_time:.2f} seconds")

    print(f"\nSkipped {skipped} empty documents")

    indexer.finalize()

    end_time_total = time.time()
    print(f"\nTotal indexing time: {end_time_total - start_time_total:.2f} seconds")
    print(f"Total documents indexed: {doc_id - 1}")
    print(f"Total tokens processed: {indexer.total_tokens}")

    start_time_of_merge = time.time()
    indexer._merge_blocks()
    end_time_of_merge = time.time()
    print(f"\nTotal merging time: {end_time_of_merge - start_time_of_merge:.2f} seconds")

    build_term_offsets(
        index_path="demo-index/final_index.jsonl",
        out_path="demo-index/term_offsets.json",
    )


if __name__ == "__main__":
    main()
