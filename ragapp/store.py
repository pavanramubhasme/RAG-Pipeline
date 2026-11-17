"""Qdrant through LangChain's ``QdrantVectorStore``: one collection, two vectors per chunk.

``dense`` holds the semantic embedding, ``bm25`` the sparse keyword vector.
In hybrid mode Qdrant runs both searches and fuses them with Reciprocal Rank
Fusion server-side, so we never merge result lists in Python.

Access control is a payload filter on ``metadata.classification`` applied *inside* the
search: chunks the caller may not see are never returned, not removed afterwards.
"""

import warnings

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, SparseEmbeddings
from qdrant_client import QdrantClient, models

from ragapp.config import Settings

DENSE = "dense"
SPARSE = "bm25"
SOURCE_KEY = "metadata.source"
CLASSIFICATION_KEY = "metadata.classification"


class KnowledgeStore:
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        embeddings: Embeddings,
        sparse_embeddings: SparseEmbeddings,
    ):
        self.client = client
        self.collection = collection
        self.embeddings = embeddings
        self.sparse_embeddings = sparse_embeddings
        self._ensure_collection()
        self._stores: dict[str, QdrantVectorStore] = {}

    @classmethod
    def from_settings(
        cls, s: Settings, embeddings: Embeddings, sparse_embeddings: SparseEmbeddings
    ) -> "KnowledgeStore":
        if s.qdrant_url:
            client = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=30)
        elif s.qdrant_path == ":memory:":
            client = QdrantClient(":memory:")
        else:
            client = QdrantClient(path=s.qdrant_path)  # embedded: one process at a time
        return cls(client, s.collection, embeddings, sparse_embeddings)

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            dim = len(self.embeddings.embed_query("dimension probe"))
            self.client.create_collection(
                self.collection,
                vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
                # IDF is computed by Qdrant over the whole collection: proper BM25, not just TF.
                sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
        with warnings.catch_warnings():  # embedded mode warns that indexes are no-ops
            warnings.simplefilter("ignore")
            for key in (SOURCE_KEY, CLASSIFICATION_KEY):  # idempotent; fast filtering on a server
                self.client.create_payload_index(self.collection, key, models.PayloadSchemaType.KEYWORD)

    def vector_store(self, mode: str = "hybrid") -> QdrantVectorStore:
        if mode not in self._stores:
            self._stores[mode] = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection,
                embedding=self.embeddings,
                sparse_embedding=self.sparse_embeddings,
                retrieval_mode=RetrievalMode(mode),
                vector_name=DENSE,
                sparse_vector_name=SPARSE,
            )
        return self._stores[mode]

    # ------------------------------------------------------------------ writes
    def add(self, documents: list[Document], ids: list[str], batch_size: int = 64) -> None:
        self.vector_store("hybrid").add_documents(documents, ids=ids, batch_size=batch_size)

    @staticmethod
    def _source_filter(source: str) -> models.Filter:
        return models.Filter(
            must=[models.FieldCondition(key=SOURCE_KEY, match=models.MatchValue(value=source))]
        )

    @staticmethod
    def _scope_filter(scopes: list[str] | None) -> models.Filter | None:
        """Only chunks whose classification is one of ``scopes``; None = no restriction."""
        if scopes is None:
            return None
        return models.Filter(
            must=[models.FieldCondition(key=CLASSIFICATION_KEY, match=models.MatchAny(any=list(scopes)))]
        )

    def delete_source(self, source: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(filter=self._source_filter(source)),
            wait=True,
        )

    # ------------------------------------------------------------------- reads
    def search(
        self, query: str, k: int, mode: str = "hybrid", scopes: list[str] | None = None
    ) -> list[Document]:
        if scopes is not None and not scopes:  # a role that may see nothing
            return []
        results = self.vector_store(mode).similarity_search_with_score(
            query, k=k, filter=self._scope_filter(scopes)
        )
        return [
            Document(page_content=d.page_content, metadata={**d.metadata, "search_score": s}, id=d.id)
            for d, s in results
        ]

    def get_doc_hash(self, source: str) -> str | None:
        points, _ = self.client.scroll(
            self.collection, scroll_filter=self._source_filter(source), limit=1, with_payload=True
        )
        return points[0].payload["metadata"].get("doc_hash") if points else None

    def chunk_texts(self, source: str) -> list[str]:
        """Every chunk of one document (the evaluation counts which ones contain the answer)."""
        texts: list[str] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                scroll_filter=self._source_filter(source),
                limit=256,
                offset=offset,
                with_payload=["page_content"],
            )
            texts.extend(p.payload.get("page_content", "") for p in points)
            if offset is None:
                return texts

    def list_sources(self, scopes: list[str] | None = None) -> list[str]:
        """Documents with at least one chunk the caller may see."""
        if scopes is not None and not scopes:
            return []
        sources: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                scroll_filter=self._scope_filter(scopes),
                limit=256,
                offset=offset,
                with_payload=[SOURCE_KEY],
            )
            sources.update(p.payload["metadata"]["source"] for p in points)
            if offset is None:
                return sorted(sources)

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def ping(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def close(self) -> None:
        self.client.close()
