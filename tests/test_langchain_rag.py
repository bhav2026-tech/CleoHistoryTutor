from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_rag import (
    EvaluationQuery, LangChainRAG, SQLiteRAGCache, compare_retrieval, fixed_chunks,
    format_model_response, semantic_chunks,
)


class KeywordEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        vector = [
            float(lowered.count("revolution") + 1),
            float(lowered.count("treaty") + 1),
            float(lowered.count("agriculture") + 1),
        ]
        magnitude = sum(value * value for value in vector) ** 0.5
        return [value / magnitude for value in vector]


def test_both_chunkers_preserve_source_metadata():
    text = (
        "The revolution changed political institutions and public authority. " * 8
        + "The treaty established a new border and formally ended the conflict. " * 8
    )
    source = [Document(page_content=text, metadata={"filename": "history.txt", "location": "document"})]

    for chunks in (fixed_chunks(source), semantic_chunks(source, KeywordEmbeddings())):
        assert chunks
        assert all(chunk.metadata["filename"] == "history.txt" for chunk in chunks)
        assert all("chunk_id" in chunk.metadata for chunk in chunks)


def test_comparison_uses_identical_queries_and_computes_mrr():
    relevant = Document(page_content="The treaty was signed in 1919.", metadata={})
    irrelevant = Document(page_content="Agricultural output increased.", metadata={})

    class Pipeline:
        top_k = 2
        chunks = {"fixed": [irrelevant, relevant], "semantic": [relevant, irrelevant]}

        def retrieve(self, question, strategy):
            return self.chunks[strategy]

    report = compare_retrieval(
        Pipeline(), [EvaluationQuery("When was the treaty signed?", ("signed in 1919",))]
    )
    assert report["fixed"]["hit_rate@2"] == 1
    assert report["fixed"]["mrr"] == 0.5
    assert report["semantic"]["mrr"] == 1


def test_provider_blocks_and_json_are_rendered_as_markdown():
    content = [{"type": "text", "text": '{"short_answer":"The treaty ended the war.","key_points":["Signed in 1919","Changed borders"]}'}]
    result = format_model_response(content)
    assert "## Short Answer" in result
    assert "- Signed in 1919" in result
    assert "{'type': 'text'" not in result


def test_stringified_provider_blocks_hide_signature_metadata():
    content = "[{'type': 'text', 'text': 'Being bipedal means walking on two legs [Source 1].', 'extras': {'signature': 'secret'}}]"
    result = format_model_response(content)
    assert result == "Being bipedal means walking on two legs [Source 1]."
    assert "signature" not in result


def test_library_and_index_are_reused_from_sqlite(tmp_path):
    class CountingEmbeddings(KeywordEmbeddings):
        calls = 0

        def embed_documents(self, texts):
            self.calls += 1
            return super().embed_documents(texts)

    database = str(tmp_path / "rag.db")
    document = Document(
        page_content=("The revolution changed government. " * 10) + ("The treaty ended conflict. " * 10),
        metadata={"filename": "history.txt", "location": "document"},
    )
    cache = SQLiteRAGCache(database)
    cache.save_library("library-1", "History", [document])
    assert cache.load_library("library-1")[0].page_content == document.page_content

    embeddings = CountingEmbeddings()
    first = LangChainRAG(
        [document], embeddings=embeddings, llm=object(), cache_path=database, library_id="library-1"
    )
    calls_after_first_build = embeddings.calls
    second = LangChainRAG(
        [document], embeddings=embeddings, llm=object(), cache_path=database, library_id="library-1"
    )
    assert embeddings.calls == calls_after_first_build
    assert len(second.chunks["semantic"]) == len(first.chunks["semantic"])
