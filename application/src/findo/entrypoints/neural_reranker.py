from __future__ import annotations

from sentence_transformers import CrossEncoder

from findo.core.model import Document


class NeuralReranker:
    def __init__(
        self,
        model_name: str = "unicamp-dl/mMiniLM-L6-v2-pt-v2",
        max_length: int = 512,
        device: str | None = None,
    ):
        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
            device=device,
        )

    def rerank(self, query: str, docs: list[Document]):
        """Return the same documents, reordered by neural relevance"""
        if not docs:
            return []

        pairs = [(query, doc.content) for doc in docs]

        scores = self.model.predict(pairs)

        doc_score_pairs = sorted(
            zip(docs, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        return [doc for doc, _ in doc_score_pairs]

    def score_sentences(self, query: str, sentences: list[str]):
        """Returns a list of relevance scores for each sentence"""
        if not sentences:
            return []

        pairs = [(query, s) for s in sentences]
        scores = self.model.predict(pairs)
        return scores
