"""A LangChain ``BaseRetriever``: hybrid search for candidates, then cross-encoder rerank.

Being a real LangChain retriever means it composes with any LCEL chain or
agent, and OpenTelemetry traces show it as a RETRIEVER span with the
documents it returned.
"""

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ragapp.query_expansion import expand_query
from ragapp.register_lookup import asked_addresses, keep_matching
from ragapp.reranker import FastEmbedReranker


class HybridRerankRetriever(BaseRetriever):
    store: Any  # KnowledgeStore
    reranker: FastEmbedReranker | None = None  # None = no reranking (ablation)
    mode: str = "hybrid"
    candidates_k: int = 20
    top_k: int = 4
    expand_queries: bool = True  # also search "CH1" for "channel 1" (ragapp/query_expansion.py)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        search_query: str | None = None,
        top_k: int | None = None,
        scopes: list[str] | None = None,
    ) -> list[Document]:
        """Search with ``search_query`` (an LLM rewrite, if any). A chunk counts as relevant
        if it matches the original question *or* the rewrite; the grounding check at the
        end of the graph guards against a rewrite drifting away from the user's intent.
        ``scopes`` = classification labels the caller may see (None = all), enforced by Qdrant.
        With ``expand_queries``, both are searched and reranked with the other spellings of
        numbered terms added ("channel 1" -> "CH1"), so BM25 and the cross-encoder see the
        words the document uses. A question about a register address ("register at 0x04")
        keeps only the chunks whose heading has that address (ragapp/register_lookup.py)."""
        limit = top_k or self.top_k
        text = search_query or query
        expanded = expand_query(text) if self.expand_queries else text
        asks_address = bool(asked_addresses(f"{query} {text}"))
        k = self.candidates_k * 3 if asks_address else self.candidates_k  # wider net, then filter
        candidates = self.store.search(expanded, k=k, mode=self.mode, scopes=scopes)
        if asks_address:
            candidates = keep_matching(f"{query} {text}", candidates)
        if self.reranker is None:
            # Fusion scores are not calibrated, so there is no threshold and no refusal.
            return candidates[:limit]
        return self.reranker.rank(candidates, [query, text, expanded])[:limit]