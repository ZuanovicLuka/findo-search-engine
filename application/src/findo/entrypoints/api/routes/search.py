"""Search endpoints."""

from fastapi import APIRouter

from findo.entrypoints.answer_generation import AnswerGenerator
from findo.entrypoints.api.model import SearchResponse
from findo.entrypoints.neural_reranker import NeuralReranker
from findo.entrypoints.query_corrector import QueryCorrector
from findo.entrypoints.searcher import Searcher
from findo.entrypoints.summarizer import Summarizer

router = APIRouter(tags=["search engine"])

MODELS = {
    "model1": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "model2": "unicamp-dl/mMiniLM-L6-v2-pt-v2",
    "model3": "cross-encoder/ms-marco-TinyBERT-L2-v2",
    "model4": "cross-encoder/ms-marco-MiniLM-L2-v2",
    "model5": "cross-encoder/ms-marco-MiniLM-L4-v2",
    "model6": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "model7": "cross-encoder/ms-marco-MiniLM-L12-v2",
}

neural_reranker = NeuralReranker(
    model_name=MODELS["model2"],
    max_length=512,
)

searchEngine = Searcher(
    index_path="demo-index/final_index.jsonl",
    config_path="demo-index/config_file.json",
    doc_stats="demo-index/doc_stats.jsonl",
    sqlite_path="demo-index/database.db",
    metadata_path="demo-index/metadata.json",
    term_offsets_path="demo-index/term_offsets.json",
    neural_reranker=neural_reranker,
    rerank_k=50,
)

searchEngine.search("portugal")

answerGenerator = AnswerGenerator()
summarizer = Summarizer()
queryCorrector = QueryCorrector()


@router.get("/correct_query")
def correct_query(query: str) -> str:
    """Correct the given query for spelling/typos"""
    corrected_query = queryCorrector.correct_query(query)
    return corrected_query


@router.get("/search")
def search(query: str, num_results: int = 10) -> SearchResponse:
    """Search for documents matching the given query"""
    results = searchEngine.search(query, num_results)
    return SearchResponse(results=results)


@router.get("/is_question")
def is_question(query: str) -> bool:
    """Check if the given query is a question"""
    answer = answerGenerator.is_question(query)
    return answer


@router.get("/summarize")
def summarize(text: str) -> str:
    """Summarize the given text"""
    summary = summarizer.summarize(text)
    return summary


@router.get("/generate_answer")
def generate_answer(query: str, document_content: str) -> str:
    """Generate an answer for the given query based on the document content"""
    answer = answerGenerator.generate_answer(query, document_content)
    if not answer:
        answer = "Desculpe, ocorreu um erro ao gerar a resposta."
    return answer


@router.get("/search_doc")
def search_doc(doc_id: int) -> SearchResponse:
    """Getting the document content based on the specific document ID"""
    return SearchResponse(results=searchEngine.search_doc(doc_id))


@router.get("/search_like")
def search_like(doc_id: int, query: str, num_results: int = 10) -> SearchResponse:
    """Search for documents similar to the given document ID"""
    return SearchResponse(results=searchEngine.search_similar(doc_id, query, num_results))
