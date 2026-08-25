"""LangChain retrieval and a LangGraph retrieve-then-generate workflow."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

from rag import Passage, read_document


Strategy = Literal["fixed", "semantic"]


def _documents_to_json(documents: list[Document]) -> str:
    return json.dumps([{"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents])


def _documents_from_json(payload: str) -> list[Document]:
    return [Document(page_content=item["page_content"], metadata=item["metadata"]) for item in json.loads(payload)]


class SQLiteRAGCache:
    """Persistent source, chunk, and embedding cache shared across app restarts."""

    def __init__(self, path: str = "rag_history.db") -> None:
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS rag_libraries "
            "(id TEXT PRIMARY KEY, name TEXT, documents TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS rag_chunks "
            "(library_id TEXT, strategy TEXT, config TEXT, documents TEXT NOT NULL, "
            "PRIMARY KEY (library_id, strategy, config))"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS rag_embedding_cache "
            "(namespace TEXT, kind TEXT, text_hash TEXT, vector TEXT NOT NULL, "
            "PRIMARY KEY (namespace, kind, text_hash))"
        )
        self.connection.commit()

    def save_library(self, library_id: str, name: str, documents: list[Document]) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO rag_libraries (id, name, documents) VALUES (?, ?, ?)",
            (library_id, name, _documents_to_json(documents)),
        )
        self.connection.commit()

    def libraries(self) -> list[tuple[str, str]]:
        return self.connection.execute(
            "SELECT id, name FROM rag_libraries ORDER BY created_at DESC"
        ).fetchall()

    def load_library(self, library_id: str) -> list[Document] | None:
        row = self.connection.execute(
            "SELECT documents FROM rag_libraries WHERE id = ?", (library_id,)
        ).fetchone()
        return _documents_from_json(row[0]) if row else None

    def get_chunks(self, library_id: str, strategy: Strategy, config: str) -> list[Document] | None:
        row = self.connection.execute(
            "SELECT documents FROM rag_chunks WHERE library_id = ? AND strategy = ? AND config = ?",
            (library_id, strategy, config),
        ).fetchone()
        return _documents_from_json(row[0]) if row else None

    def put_chunks(self, library_id: str, strategy: Strategy, config: str, documents: list[Document]) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO rag_chunks (library_id, strategy, config, documents) VALUES (?, ?, ?, ?)",
            (library_id, strategy, config, _documents_to_json(documents)),
        )
        self.connection.commit()


class SQLiteCachedEmbeddings(Embeddings):
    def __init__(self, delegate: Embeddings, cache: SQLiteRAGCache, namespace: str) -> None:
        self.delegate, self.cache, self.namespace = delegate, cache, namespace

    def _get(self, text: str, kind: str) -> list[float] | None:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        row = self.cache.connection.execute(
            "SELECT vector FROM rag_embedding_cache WHERE namespace = ? AND kind = ? AND text_hash = ?",
            (self.namespace, kind, digest),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def _put(self, text: str, kind: str, vector: list[float]) -> None:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self.cache.connection.execute(
            "INSERT OR REPLACE INTO rag_embedding_cache (namespace, kind, text_hash, vector) VALUES (?, ?, ?, ?)",
            (self.namespace, kind, digest, json.dumps(vector)),
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float] | None] = [self._get(text, "document") for text in texts]
        missing_indexes = [index for index, vector in enumerate(vectors) if vector is None]
        if missing_indexes:
            missing = [texts[index] for index in missing_indexes]
            generated = self.delegate.embed_documents(missing)
            for index, vector in zip(missing_indexes, generated):
                vectors[index] = vector
                self._put(texts[index], "document", vector)
            self.cache.connection.commit()
        return [vector or [] for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        cached = self._get(text, "query")
        if cached is not None:
            return cached
        vector = self.delegate.embed_query(text)
        self._put(text, "query", vector)
        self.cache.connection.commit()
        return vector


def documents_from_files(
    files: list[tuple[str, bytes]], ocr_client=None, ocr_model: str = "gemini-3.7-flash"
) -> list[Document]:
    """Parse uploaded files into page/location-aware LangChain documents."""
    documents: list[Document] = []
    for filename, data in files:
        for location, text in read_document(filename, data, ocr_client, ocr_model):
            if text.strip():
                documents.append(Document(page_content=text.strip(), metadata={"filename": filename, "location": location}))
    return documents


def fixed_chunks(documents: list[Document]) -> list[Document]:
    """Baseline: conventional character chunks with overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1_200, chunk_overlap=180,
        separators=["\n\n", ". ", "? ", "! ", "\n", " "], add_start_index=True,
    )
    return _number_chunks(splitter.split_documents(documents), "fixed")


def semantic_chunks(documents: list[Document], embeddings: Embeddings, percentile: float = 85) -> list[Document]:
    """Split where adjacent sentence embeddings show a semantic discontinuity."""
    splitter = SemanticChunker(
        embeddings, breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=percentile, min_chunk_size=120, add_start_index=True,
    )
    return _number_chunks(splitter.split_documents(documents), "semantic")


def _number_chunks(documents: list[Document], strategy: Strategy) -> list[Document]:
    for index, document in enumerate(documents):
        document.metadata = {**document.metadata, "chunk_id": f"{strategy}-{index}", "strategy": strategy}
    return documents


def as_passage(document: Document) -> Passage:
    return Passage(
        id=str(document.metadata["chunk_id"]),
        filename=str(document.metadata.get("filename", "source")),
        location=str(document.metadata.get("location", "document")),
        text=document.page_content,
    )


def format_model_response(content: object) -> str:
    """Normalize provider content blocks or JSON into readable Markdown."""
    if isinstance(content, list):
        parts = [str(block.get("text", "")) if isinstance(block, dict) else str(block) for block in content]
        content = "\n".join(part for part in parts if part.strip())
    if not isinstance(content, str):
        content = str(content)
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text
    parsed: object
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            return text
    if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
        return format_model_response(parsed)
    return _json_to_markdown(parsed)


def _json_to_markdown(value: object, level: int = 2) -> str:
    if isinstance(value, dict):
        return "\n\n".join(
            f"{'#' * min(level, 6)} {str(key).replace('_', ' ').title()}\n\n{_json_to_markdown(item, level + 1)}"
            for key, item in value.items()
        )
    if isinstance(value, list):
        return "\n".join(f"- {_json_to_markdown(item, level + 1)}" for item in value)
    return "Not provided." if value is None else str(value)


class GraphState(TypedDict, total=False):
    question: str
    strategy: Strategy
    context: list[Document]
    answer: str


class LangChainRAG:
    """Two comparable retrievers sharing embeddings, plus a LangGraph workflow."""

    def __init__(self, documents: list[Document], embedding_model: str = "models/gemini-embedding-001",
                 chat_model: str = "gemini-2.5-flash", embeddings: Embeddings | None = None,
                 llm=None, top_k: int = 6, api_key: str | None = None,
                 cache_path: str | None = None, library_id: str | None = None) -> None:
        base_embeddings = embeddings or GoogleGenerativeAIEmbeddings(
            model=embedding_model, google_api_key=api_key
        )
        self.cache = SQLiteRAGCache(cache_path) if cache_path else None
        self.embeddings = (
            SQLiteCachedEmbeddings(base_embeddings, self.cache, embedding_model)
            if self.cache else base_embeddings
        )
        self.llm = llm or ChatGoogleGenerativeAI(
            model=chat_model, temperature=0.2, google_api_key=api_key
        )
        self.top_k = top_k
        chunk_config = "fixed-1200-180|semantic-percentile-85-min120-v1"
        self.chunks: dict[Strategy, list[Document]] = {}
        for strategy in ("fixed", "semantic"):
            cached = self.cache.get_chunks(library_id, strategy, chunk_config) if self.cache and library_id else None
            chunks = cached or (fixed_chunks(documents) if strategy == "fixed" else semantic_chunks(documents, self.embeddings))
            self.chunks[strategy] = chunks
            if cached is None and self.cache and library_id:
                self.cache.put_chunks(library_id, strategy, chunk_config, chunks)
        self.stores: dict[Strategy, InMemoryVectorStore] = {}
        for strategy, chunks in self.chunks.items():
            store = InMemoryVectorStore(self.embeddings)
            store.add_documents(chunks)
            self.stores[strategy] = store
        self.graph = self._build_graph()

    def retrieve(self, question: str, strategy: Strategy = "semantic") -> list[Document]:
        if strategy not in self.stores:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")
        return self.stores[strategy].similarity_search(question, k=self.top_k)

    def ask(self, question: str, strategy: Strategy = "semantic") -> dict:
        return self.graph.invoke({"question": question, "strategy": strategy})

    def _build_graph(self):
        def retrieve_node(state: GraphState) -> dict:
            return {"context": self.retrieve(state["question"], state.get("strategy", "semantic"))}

        def generate_node(state: GraphState) -> dict:
            sources = "\n\n".join(
                f"[Source {index}: {doc.metadata.get('filename')}, {doc.metadata.get('location')}]\n{doc.page_content}"
                for index, doc in enumerate(state["context"], 1)
            )
            system = (
                "You are Clio, a careful history tutor. Answer only from the supplied sources and cite every "
                "factual claim as [Source N]. If evidence is insufficient, say what is missing. "
                "Return readable Markdown, never JSON. Use these sections: 'Short answer' (2-3 sentences), "
                "'Key points' (concise bullets), and 'Evidence digest' (one brief bullet per source actually used)."
            )
            response = self.llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=f"SOURCES\n{sources}\n\nQUESTION\n{state['question']}"),
            ])
            return {"answer": format_model_response(response.content)}

        builder = StateGraph(GraphState)
        builder.add_node("retrieve", retrieve_node)
        builder.add_node("generate", generate_node)
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)
        return builder.compile()


@dataclass(frozen=True)
class EvaluationQuery:
    question: str
    relevant_text: tuple[str, ...]


def compare_retrieval(pipeline: LangChainRAG, queries: list[EvaluationQuery], k: int | None = None) -> dict[str, dict[str, float]]:
    """Compare retrievers on identical labelled queries using Hit@k and MRR."""
    cutoff = k or pipeline.top_k
    results: dict[str, dict[str, float]] = {}
    for strategy in ("fixed", "semantic"):
        hits = 0
        reciprocal_rank = 0.0
        for query in queries:
            retrieved = pipeline.retrieve(query.question, strategy)[:cutoff]
            rank = next((index for index, doc in enumerate(retrieved, 1)
                         if any(label.casefold() in doc.page_content.casefold() for label in query.relevant_text)), None)
            if rank is not None:
                hits += 1
                reciprocal_rank += 1 / rank
        count = len(queries)
        results[strategy] = {
            f"hit_rate@{cutoff}": hits / count if count else 0.0,
            "mrr": reciprocal_rank / count if count else 0.0,
            "query_count": float(count),
            "chunk_count": float(len(pipeline.chunks[strategy])),
        }
    return results
