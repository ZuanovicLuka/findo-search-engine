import time

from findo.entrypoints.neural_reranker import NeuralReranker
from findo.entrypoints.searcher import Searcher

OUTPUT_FILE = "evaluation/rerank_results.txt"
K = 50
TARGET_ID = 85321

QUERIES = [
    "História da Academia Cristiano Ronaldo do Sporting Clube de Portugal",
    "Quando foi inaugurada a Academia de Alcochete e porque recebeu o nome de Cristiano Ronaldo?",
    "Infraestruturas desportivas e hoteleiras da Academia Sporting em Alcochete",
    "Ataque à Academia de Alcochete 2018 detalhes arguidos julgamento",
    "Quais jogadores famosos foram formados na Academia Sporting?",
    "Campos de treino e instalações da Academia do Sporting em Alcochete",
    "Quantos quartos existem no edifício central da Academia Cristiano Ronaldo?",
    "Ginásios, balneários e equipamentos da formação do Sporting CP",
    "Rede de Escolas Academia Cristiano Ronaldo para jovens entre 5 e 14 anos",
    "Como funciona a detecção de talentos nas Escolas Academia Sporting?",
    "Academia Sporting certificação ISO9001:2008 detalhes",
    "Por que a Academia do Sporting é considerada uma das melhores da Europa?",
    "Centro de estágio do Sporting utilizado no Euro 2004",
    "Complexos de treino de futebol em Portugal inaugurados em 2002",
    "Condições de treino para equipas profissionais no Sporting Clube de Portugal",
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


def evaluate_model(model_name, f):
    """Evaluates a reranking model by running multiple queries, measuring average rerank time,

    and checking how often a target document appears in rank 1 and in the top 10
    """
    print(f"\nEvaluating {model_name}")

    reranker = NeuralReranker(model_name=model_name, max_length=512)
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

    report = [
        f"\n=============================\nModel: {model_name}\n=============================\n"
    ]

    total_time = 0
    first_result = 0
    in_top_10 = 0

    for query in QUERIES:
        print(f'Processing query: "{query}"')
        bm25_scores = searcher._bm25_scores(query)
        sorted_docs = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
        docs = searcher.get_title_content(sorted_docs[:K])

        start = time.time()
        reranked = reranker.rerank(query, docs)
        elapsed = time.time() - start
        total_time += elapsed

        pos = None
        for idx, d in enumerate(reranked):
            if d.id == TARGET_ID:
                pos = idx + 1
                break

        if pos:
            if pos == 1:
                first_result += 1
            if pos <= 10:
                in_top_10 += 1

    avg_time = total_time / len(QUERIES)
    report.append(f"\nAverage rerank time for model: {avg_time:.4f} seconds\n")
    report.append(
        f"Number of times document {TARGET_ID} was first: {first_result} out of {len(QUERIES)}"
    )
    report.append(
        f"Number of times document {TARGET_ID} was in top-10: {in_top_10} out of {len(QUERIES)}"
    )

    f.write("\n".join(report) + "\n\n")
    print(f"Finished {model_name}")


if __name__ == "__main__":
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for key, model_name in MODELS.items():
            evaluate_model(model_name, f)

    print("\nAll results saved to rerank_results.txt")
