import csv
import os
from collections import defaultdict

import ir_measures
from ir_measures import nDCG
from sentence_transformers import CrossEncoder

DATA_DIR = "trec_dl_2019"
OUTPUT_FILE = "evaluation/ndcg.txt"

COLLECTION_TSV = os.path.join(DATA_DIR, "collection.tsv")
QUERIES_TSV = os.path.join(DATA_DIR, "msmarco-test2019-queries.tsv")
CANDIDATES_TSV = os.path.join(DATA_DIR, "msmarco-passagetest2019-top1000.tsv")
QRELS_FILE = os.path.join(DATA_DIR, "2019qrels-pass.txt")
RUNS_DIR = os.path.join(DATA_DIR, "runs")

os.makedirs(RUNS_DIR, exist_ok=True)

MODELS = {
    "mmarco_mMiniLMv2_L12_H384": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "mMiniLM_L6_pt": "unicamp-dl/mMiniLM-L6-v2-pt-v2",
    "TinyBERT_L2": "cross-encoder/ms-marco-TinyBERT-L2-v2",
    "MiniLM_L2": "cross-encoder/ms-marco-MiniLM-L2-v2",
    "MiniLM_L4": "cross-encoder/ms-marco-MiniLM-L4-v2",
    "MiniLM_L6": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "MiniLM_L12": "cross-encoder/ms-marco-MiniLM-L12-v2",
}


def load_queries(path):
    queries = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            qid, text = row[0], row[1]
            queries[qid] = text
    return queries


def load_collection(path):
    passages = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            pid, text = row[0], row[1]
            passages[pid] = text
    return passages


def load_candidates(path):
    candidates = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            qid, pid = row[0], row[1]
            candidates[qid].append(pid)
    return candidates


def rerank_and_write_run(
    model_name: str, model_tag: str, queries, passages, candidates, batch_size: int = 32
):
    print(f"\nLoading CrossEncoder: {model_name}")
    model = CrossEncoder(model_name, max_length=512)

    run_path = os.path.join(RUNS_DIR, f"run_{model_tag}.txt")

    total_queries = len(candidates)
    print(f"Reranking {total_queries} queries")

    with open(run_path, "w", encoding="utf-8") as out:
        for i, (qid, pids) in enumerate(candidates.items(), start=1):
            if qid not in queries:
                continue

            query_text = queries[qid]

            pairs = [(query_text, passages[pid]) for pid in pids if pid in passages]
            pid_list = [pid for pid in pids if pid in passages]

            scores = model.predict(pairs, batch_size=batch_size)

            ranked = sorted(zip(pid_list, scores), key=lambda x: float(x[1]), reverse=True)

            for rank, (pid, score) in enumerate(ranked, start=1):
                out.write(f"{qid} Q0 {pid} {rank} {score:.6f} {model_tag}\n")

            if i % 10 == 0 or i == total_queries:
                print(f"  Processed {i}/{total_queries} queries")

    print(f"Saved run to {run_path}")
    return run_path


def evaluate_run(run_path: str):
    qrels = ir_measures.read_trec_qrels(QRELS_FILE)
    run = ir_measures.read_trec_run(run_path)
    results = ir_measures.calc_aggregate([nDCG @ 10], qrels, run)
    return results[nDCG @ 10]


if __name__ == "__main__":
    print("Loading TREC DL 2019 data")

    queries = load_queries(QUERIES_TSV)
    passages = load_collection(COLLECTION_TSV)
    candidates = load_candidates(CANDIDATES_TSV)

    selected_qids = list(candidates.keys())[:50]
    queries = {qid: queries[qid] for qid in selected_qids if qid in queries}
    candidates = {qid: candidates[qid] for qid in selected_qids}

    print(
        f"Loaded {len(queries)} queries, "
        f"{len(passages)} passages, "
        f"{len(candidates)} candidate query sets."
    )

    scores = {}

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for tag, model_name in MODELS.items():
            run_path = rerank_and_write_run(
                model_name=model_name,
                model_tag=tag,
                queries=queries,
                passages=passages,
                candidates=candidates,
                batch_size=32,
            )

            ndcg10 = evaluate_run(run_path)
            scores[tag] = ndcg10

            block = (
                "\n=====================\n"
                f"Model: {model_name}\n"
                f"nDCG@10 (TREC DL 2019): {ndcg10:.4f}\n"
                "=====================\n"
            )

            print(block)

            out_f.write(block)

        summary_header = "\nSummary (sorted by nDCG@10):\n"
        print(summary_header)
        out_f.write(summary_header)

        for tag, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            line = f"{tag:20s}  nDCG@10 = {score:.4f}\n"
            print(line, end="")
            out_f.write(line)

    print(f"\nnDCG results saved to {OUTPUT_FILE}\n")
