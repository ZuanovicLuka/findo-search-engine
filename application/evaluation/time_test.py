import time

from findo.entrypoints.neural_reranker import NeuralReranker
from findo.entrypoints.searcher import Searcher

K = 50
OUTPUT_FILE = "evaluation/times.txt"

QUERIES = [
    "Quem é Cristiano Ronaldo?",
    "O que é inteligência artificial?",
    "História da Universidade de Coimbra",
    "Revolução dos Cravos resumo",
]

MODELS = {
    "model1": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "model2": "unicamp-dl/mMiniLM-L6-v2-pt-v2",
    "model3": "cross-encoder/ms-marco-TinyBERT-L2-v2",
    "model4": "cross-encoder/ms-marco-MiniLM-L2-v2",
    "model5": "cross-encoder/ms-marco-MiniLM-L4-v2",
    "model6": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "model7": "cross-encoder/ms-marco-MiniLM-L12-v2",
}


def measure_avg_rerank_time(model_name: str, queries: list[str], f):
    """Measures the average reranking time of a model by running multiple queries,

    recording execution speed, and logging model architecture and parameter details
    """
    print(f"\nEvaluating: {model_name}")

    reranker = NeuralReranker(model_name=model_name, max_length=512)
    config = reranker.model.model.config
    total_params = sum(p.numel() for p in reranker.model.model.parameters())

    searcher = Searcher(
        index_path="demo-index/final_index.jsonl",
        config_path="demo-index/config_file.json",
        doc_stats="demo-index/doc_stats.jsonl",
        sqlite_path="demo-index/database.db",
        metadata_path="demo-index/metadata.json",
        term_offsets_path="demo-index/term_offsets.json",
        neural_reranker=reranker,
        rerank_k=K,
    )

    candidates_per_query = []
    for q in queries:
        bm25_scores = searcher._bm25_scores(q)
        sorted_docs = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
        docs = searcher.get_title_content(sorted_docs[:K])
        candidates_per_query.append(docs)

    reranker.rerank(queries[0], candidates_per_query[0])
    total_time = 0
    runs = 3

    for r in range(runs):
        for q, docs in zip(queries, candidates_per_query):
            start = time.time()
            reranker.rerank(q, docs)
            elapsed = time.time() - start
            total_time += elapsed
            print(f"Query '{q}' processed in {elapsed:.4f} seconds")

    avg_time = total_time / (len(queries) * runs)

    block = f"""
            =====================
            Loaded CrossEncoder: {model_name}

            --- Architecture ---
            Model type:              {config.model_type}
            Hidden size:             {config.hidden_size}
            Number of layers:        {config.num_hidden_layers}
            Attention heads:         {config.num_attention_heads}
            Max position embeddings: {config.max_position_embeddings}

            Total parameters:        {total_params:,}

            Average time for {len(QUERIES)} queries: {avg_time:.4f} seconds
            =====================
            """

    print(block)
    f.write(block + "\n")

    return avg_time


if __name__ == "__main__":
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for key, model_name in MODELS.items():
            measure_avg_rerank_time(model_name, QUERIES, f)
            print(f"Saved results for {model_name}.\n")
