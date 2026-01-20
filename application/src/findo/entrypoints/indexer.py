import glob
import heapq
import json
import os
from collections import defaultdict
from contextlib import ExitStack


class SPIMIIndexer:
    """Single-Pass In-Memory Indexer (SPIMI)

    Builds an inverted index in blocks and merges them efficiently
    """

    INVALID_TOKENS = {"<STOPWORD>", "<NUMBER>", "<SHORT-TOKEN>", "<URL>", "URL", "url"}

    def __init__(self, output_dir="demo-index", memory_limit_mb=1800, min_term_freq=3):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.memory_limit = memory_limit_mb * 1024 * 1024
        self.min_term_freq = min_term_freq

        self.inverted_index = defaultdict(list)
        self.block_count = 0
        self.token_count = 0
        self.token_threshold = 4_500_000  # flush block after this many tokens
        self.total_tokens = 0

        self.doc_lengths = {}
        self.doc_count = 0
        self.doc_stats_path = os.path.join(self.output_dir, "doc_stats.jsonl")

    def add_document(self, doc_id: int, tokens: list[str]):
        """Add one documents tokens to the in-memory index"""
        term_freq = defaultdict(int)
        valid_tokens = 0

        for term in tokens:
            if term in self.INVALID_TOKENS:
                continue
            term_freq[term] += 1
            valid_tokens += 1
            self.total_tokens += 1

        self.doc_lengths[doc_id] = valid_tokens
        self.doc_count += 1
        with open(self.doc_stats_path, "a", encoding="utf-8") as stats_file:
            stats_file.write(json.dumps({"doc_id": doc_id, "length": valid_tokens}) + "\n")

        for term, freq in term_freq.items():
            self.inverted_index[term].append((doc_id, freq))
            self.token_count += 1

        if self.token_count >= self.token_threshold:
            self._write_block()
            self.inverted_index.clear()
            self.token_count = 0

    def _write_block(self):
        """Writes one sorted block to a .jsonl file (one term per line)"""
        self.block_count += 1
        block_path = os.path.join(self.output_dir, f"{self.block_count}.jsonl")
        sorted_index = dict(sorted(self.inverted_index.items()))

        with open(block_path, "w", encoding="utf-8") as f:
            for term, postings in sorted_index.items():
                f.write(json.dumps({term: postings}) + "\n")

        print(
            f"SPIMI wrote block {self.block_count} "
            f"with {len(self.inverted_index)} terms, {block_path}"
        )

    def finalize(self):
        """Flush any remaining in-memory index to disk"""
        if self.inverted_index:
            self._write_block()
            self.inverted_index.clear()
        print("SPIMI indexing complete")

        avg_doc_length = self.total_tokens / self.doc_count if self.doc_count > 0 else 0

        metadata = {
            "doc_count": self.doc_count,
            "total_tokens": self.total_tokens,
            "avg_doc_length": avg_doc_length,
            "doc_stats_file": os.path.basename(self.doc_stats_path),
        }

        meta_path = os.path.join(self.output_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        print(f"Metadata saved to {meta_path}")
        print(f"Document stats written to {self.doc_stats_path}")

    def _merge_blocks(self):
        """Merge all intermediate .jsonl blocks into a single final index"""
        block_files = sorted(
            f
            for f in glob.glob(os.path.join(self.output_dir, "*.jsonl"))
            if not f.endswith(("final_index.jsonl", "doc_stats.jsonl"))
        )

        if not block_files:
            print("No blocks to merge")
            return

        with ExitStack() as stack:
            block_handles = [
                stack.enter_context(open(path, encoding="utf-8")) for path in block_files
            ]
            block_iterators = [self._line_iterator(f) for f in block_handles]

            heap = []
            for block_id, iterator in enumerate(block_iterators):
                try:
                    term, postings = next(iterator)
                    heapq.heappush(heap, (term, postings, block_id))
                except StopIteration:
                    continue

            final_index_path = os.path.join(self.output_dir, "final_index.jsonl")
            temp_index_path = os.path.join(self.output_dir, "temp_index.jsonl")

            current_term = None
            current_postings = []
            term_count = 0

            with open(temp_index_path, "w", encoding="utf-8") as f:
                while heap:
                    term, postings, block_id = heapq.heappop(heap)

                    if term == current_term:
                        current_postings.extend(postings)
                    else:
                        if current_term:
                            merged_postings = self._merge_postings(current_postings)
                            if len(merged_postings) >= self.min_term_freq:
                                f.write(json.dumps({current_term: merged_postings}) + "\n")
                                term_count += 1

                        current_term = term
                        current_postings = postings

                    try:
                        next_term, next_postings = next(block_iterators[block_id])
                        heapq.heappush(heap, (next_term, next_postings, block_id))
                    except StopIteration:
                        continue

                if current_term:
                    merged_postings = self._merge_postings(current_postings)
                    if len(merged_postings) >= self.min_term_freq:
                        f.write(json.dumps({current_term: merged_postings}) + "\n")
                        term_count += 1

        os.rename(temp_index_path, final_index_path)
        print(f"Merged {len(block_files)} blocks, {term_count} terms written to {final_index_path}")

    @staticmethod
    def _merge_postings(postings):
        """Combine duplicate (docID, freq) pairs, summing frequencies per docID"""
        merged = defaultdict(int)
        for doc_id, freq in postings:
            merged[doc_id] += freq
        return sorted(merged.items())

    @staticmethod
    def _line_iterator(file_handle):
        """Generator that yields (term, postings) pairs from a .jsonl block"""
        for line in file_handle:
            if not line.strip():
                continue
            data = json.loads(line)
            term, postings = next(iter(data.items()))
            yield term, postings
