"""Cross-encoder reranker as a native LangChain component.

LangChain's built-in rerankers need PyTorch (sentence-transformers) or a paid
API. This one wraps FastEmbed's ONNX cross-encoder, so it stays small and free,
and plugs into anything that accepts a ``BaseDocumentCompressor``.
"""

import math
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import PrivateAttr


class FastEmbedReranker(BaseDocumentCompressor):
    model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    cache_dir: str | None = None
    top_n: int = 4
    min_score: float = 0.05
    _model: Any = PrivateAttr(default=None)

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance of each text to the query as a probability in [0, 1]."""
        if self._model is None:  # lazy: load the model on first use
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(model_name=self.model_name, cache_dir=self.cache_dir)
        return [1.0 / (1.0 + math.exp(-float(x))) for x in self._model.rerank(query, texts)]

    def rank(self, documents: Sequence[Document], queries: list[str]) -> list[Document]:
        """Score every document against each query, keep the best score per document,
        drop those below ``min_score`` and return the ``top_n`` best."""
        if not documents:
            return []
        texts = [d.page_content for d in documents]
        per_query = [self.score(q, texts) for q in dict.fromkeys(queries)]  # dedupe, keep order
        scores = [max(column) for column in zip(*per_query, strict=True)]
        ranked = sorted(zip(documents, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [
            Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "relevance_score": round(score, 4)},
                id=doc.id,
            )
            for doc, score in ranked
            if score >= self.min_score
        ][: self.top_n]

    def compress_documents(
        self, documents: Sequence[Document], query: str, callbacks: Callbacks | None = None
    ) -> Sequence[Document]:
        return self.rank(documents, [query])
