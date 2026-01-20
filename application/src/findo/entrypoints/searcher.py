import heapq
import json
import math
import re
import sqlite3
import time
from argparse import Namespace
from collections import Counter
from typing import Optional

from findo.core.model import Document
from findo.entrypoints.neural_reranker import NeuralReranker
from findo.entrypoints.tokenizer import Tokenizer

INVALID_TOKENS = {"<STOPWORD>", "<NUMBER>", "<SHORT-TOKEN>", "<URL>", "URL", "url"}


class Searcher:
    def __init__(
        self,
        index_path: str,
        config_path: str,
        doc_stats: str,
        metadata_path: str,
        sqlite_path: str,
        term_offsets_path: str,
        neural_reranker: Optional[NeuralReranker] = None,
        rerank_k: int = 50,
    ):
        self.index_path = index_path
        self.config_path = config_path
        self.doc_stats_path = doc_stats
        self.metadata = self._load_namespace(metadata_path)
        self.config = self._load_namespace(config_path)
        self.tokenizer = Tokenizer(self.config)

        self.sqlite_path = sqlite_path
        self.conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        self._postings_cache = {}
        self._df_cache = {}
        self._idf_cache = {}

        # BM25 parameters
        self.k = 1.2
        self.b = 0.75

        with open(term_offsets_path, encoding="utf-8") as f:
            self.term_offsets = json.load(f)

        # Reranker parameters
        self.reranker = neural_reranker
        self.rerank_k = rerank_k

    def _load_namespace(self, path):
        """Load a JSON file and convert its contents into an argparse.Namespace object"""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return Namespace(**data)

    def get_postings(self, token):
        """Get postings list for a token"""
        if token in self._postings_cache:
            return self._postings_cache[token]
        offset = self.term_offsets.get(token)
        if offset is None:
            return []
        with open(self.index_path, "rb") as f:
            f.seek(offset)
            line = f.readline().decode("utf-8")
            postings = next(iter(json.loads(line).values()))
        self._postings_cache[token] = postings
        return postings

    def get_doc_length(self, doc_id):
        """Return document length from doc_stats.jsonl"""
        if not hasattr(self, "_doc_lengths"):
            self._doc_lengths = {}
            with open(self.doc_stats_path, encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    self._doc_lengths[d["doc_id"]] = d["length"]
        return self._doc_lengths.get(doc_id, 0)

    def _idf(self, df):
        """Inverse document frequency (cached)"""
        if df in self._idf_cache:
            return self._idf_cache[df]
        N = self.metadata.doc_count
        val = math.log((N - df + 0.5) / (df + 0.5) + 1)
        self._idf_cache[df] = val
        return val

    def _get_df(self, token):
        """Fast DF lookup with caching"""
        if token in self._df_cache:
            return self._df_cache[token]
        postings = self.get_postings(token)
        df = len(postings)
        self._df_cache[token] = df
        return df

    def _distill_query_terms(self, text: str, max_terms: int = 32, df_cap: int = -1):
        """Keep only the most informative query terms by tf * idf"""

        toks = [t for t in self.tokenizer.tokenize(text) if t not in INVALID_TOKENS]
        if not toks:
            return []

        tf = Counter(toks)
        scored = []
        for term, f in tf.items():
            df = self._get_df(term)
            if df == 0:
                continue
            if df_cap != -1 and df > df_cap:
                continue
            idf = self._idf(df)
            scored.append((term, f * idf))

        top = heapq.nlargest(max_terms, scored, key=lambda x: x[1])
        return [t for t, _ in top]

    def get_title_content(self, document_ids):
        """Return title and content for specific document ids from sqlite"""
        if document_ids and isinstance(document_ids[0], (list, tuple)):
            document_ids = [doc_id for doc_id, _ in document_ids]

        placeholders = ",".join("?" * len(document_ids))
        query = f"SELECT id, title, text FROM documents WHERE id IN ({placeholders})"
        cur = self.conn.cursor()
        cur.execute(query, document_ids)
        rows = cur.fetchall()

        row_map = {row["id"]: row for row in rows}
        results = []
        for doc_id in document_ids:
            row = row_map.get(doc_id)
            if row and row["text"]:
                results.append(
                    Document(
                        id=row["id"],
                        title=row["title"],
                        content=row["text"],
                    )
                )

        print(f"\nFetched {len(results)} documents")
        return results

    def split_into_sentences(self, text: str):
        """Splits Portuguese text into sentences using punctuation heuristics"""
        text = text.replace("\n", " ").strip()
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕ])", text)
        return [s.strip() for s in sentences if len(s.strip()) > 0]

    def extract_semantic_snippet(self, query: str, content: str):
        """Generate a semantic snippet consisting of the top_k most relevant sentences in the doc"""

        MAX_CHARS = 400
        TOP_K = 3
        MAX_SENTENCES = 50

        sentences = self.split_into_sentences(content)
        if not sentences:
            return content[:MAX_CHARS] + ("..." if len(content) > MAX_CHARS else "")

        sentences = sentences[:MAX_SENTENCES]

        if self.reranker:
            scores = self.reranker.score_sentences(query, sentences)

        ranked_indices = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[
            :TOP_K
        ]
        ranked_indices.sort()

        snippet = " ".join(sentences[i] for i in ranked_indices)

        if len(snippet) > MAX_CHARS:
            snippet = snippet[:MAX_CHARS] + "..."

        return snippet

    def _bm25_scores(self, query: str, doc_id_search: int = -1):
        """Compute BM25 scores for all documents matching the query terms"""

        raw_tokens = [t for t in self.tokenizer.tokenize(query) if t not in INVALID_TOKENS]

        if len(raw_tokens) > 200:
            df_cap = int(self.metadata.doc_count * 0.05)
            tokens = self._distill_query_terms(query, max_terms=32, df_cap=df_cap)
        else:
            tokens = raw_tokens

        if not tokens:
            return {}

        scores = {}
        avg_len = self.metadata.avg_doc_length

        for token in tokens:
            postings = self.get_postings(token)
            df = len(postings)
            if df == 0:
                continue

            idf = self._idf(df)

            if df > int(self.metadata.doc_count * 0.2):
                continue

            for doc_id, tf in postings:
                if doc_id_search != -1 and doc_id == doc_id_search:
                    continue
                l_d = self.get_doc_length(doc_id)
                norm = tf + self.k * (1 - self.b + self.b * (l_d / avg_len))
                score = idf * (tf * (self.k + 1)) / norm
                scores[doc_id] = scores.get(doc_id, 0.0) + score

        return scores

    def search(self, query: str, num_results: int = 10, doc_id_search: int = -1):
        """Search for a query using BM25 + optional neural reranking

        If the reranker is provided, the results are first retrieved with
        BM25 and then re-ordered by a neural model
        """
        start_time = time.time()

        scores = self._bm25_scores(query, doc_id_search=doc_id_search)
        if not scores:
            print(f"\nNo results for '{query[:60]}'")
            return []

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print(f"\nBM25 produced {len(ranked)} candidate documents")

        if self.reranker:
            candidate_k = max(num_results, self.rerank_k)
            candidates = ranked[:candidate_k]
            print(f"Sending top-{len(candidates)} BM25 candidates to neural reranker")

            candidate_docs = self.get_title_content(candidates)
            reranked_docs = self.reranker.rerank(query, candidate_docs)
            final_docs = reranked_docs[:num_results]

            for doc in final_docs:
                doc.snippet = self.extract_semantic_snippet(query, doc.content)

            end_time = time.time()
            elapsed = end_time - start_time

            print(f"\nTop {num_results} results (after neural reranking):")
            for idx, doc in enumerate(final_docs, start=1):
                print(f"  {idx:>2}. Doc {doc.id:<8} Title: {doc.title[:60]!r}")

            print(f"\nSearch + reranking completed in {elapsed:.3f} seconds\n")
            return final_docs

        top_bm25 = ranked[:num_results]

        end_time = time.time()
        elapsed = end_time - start_time

        print(f"\nTop {num_results} results (BM25 only):")
        for doc_id, s in top_bm25:
            print(f"  Doc {doc_id:<8} Score: {s:.4f}")

        print(f"\nSearch completed in {elapsed:.3f} seconds\n")

        results = self.get_title_content(top_bm25)

        for doc in results:
            doc.snippet = doc.content[:400] + ("..." if len(doc.content) > 400 else "")

        return results

    def search_similar(self, doc_id: int, query, num_results: int = 10):
        """Search for documents similar to the doc_id"""
        results = self.search(query, num_results, doc_id_search=doc_id)
        return results

    def search_doc(self, doc_id: int):
        """Retrieve document by its ID"""
        results = self.get_title_content([doc_id])
        if results:
            searched_doc = results
            return searched_doc
        return []
