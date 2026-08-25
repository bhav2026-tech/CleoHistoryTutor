from __future__ import annotations

import hashlib
import os
import re

import streamlit as st
from dotenv import load_dotenv
from google import genai

from langchain_rag import (
    LangChainRAG, SQLiteRAGCache, as_passage, documents_from_files, format_model_response,
)
from rag import Passage


load_dotenv()
st.set_page_config(page_title="Clio · History Tutor", page_icon="🏛️", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #f7f2e8; color: #25211c; }
    [data-testid="stSidebar"] { background: #e9dfcb; }
    .hero { padding: 1.2rem 0 .7rem; border-bottom: 1px solid #bda98b; margin-bottom: 1rem; }
    .eyebrow { color: #8a482d; font-size: .78rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }
    .hero h1 { font-family: Georgia, serif; font-size: 3rem; margin: .15rem 0; color: #31261d; }
    .hero p { color: #6d5d4c; font-size: 1.08rem; max-width: 760px; }
    .source-card { border-left: 3px solid #a75838; padding: .3rem .8rem; margin: .5rem 0; background: rgba(255,255,255,.35); }
    </style>
    <div class="hero"><div class="eyebrow">Source-grounded learning</div><h1>Clio</h1>
    <p>Your private history tutor. Add primary sources, textbooks, or notes, then ask questions answered strictly from that collection.</p></div>
    """,
    unsafe_allow_html=True,
)


def chunk_digest(text: str, limit: int = 320) -> str:
    """Return a compact preview without making another model request."""
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    digest = " ".join(sentences[:2]).strip()
    if len(digest) > limit:
        digest = digest[:limit].rsplit(" ", 1)[0] + "…"
    return digest


def show_sources(sources: list[Passage]) -> None:
    for index, source in enumerate(sources, 1):
        with st.container(border=True):
            st.markdown(f"**Source {index} · {source.filename}, {source.location}**")
            st.markdown(chunk_digest(source.text))
            with st.expander("Read full chunk"):
                st.markdown(source.text)


def show_comparison(comparison: dict) -> None:
    columns = st.columns(2)
    for column, strategy_name in zip(columns, ("fixed", "semantic")):
        result = comparison[strategy_name]
        with column:
            st.subheader("Fixed chunks" if strategy_name == "fixed" else "Semantic chunks")
            st.markdown(format_model_response(result["answer"]))
            with st.expander(f"Retrieved evidence ({len(result['sources'])} chunks)"):
                show_sources(result["sources"])


@st.cache_resource
def get_rag_cache() -> SQLiteRAGCache:
    return SQLiteRAGCache("rag_history.db")


def create_pipeline(documents, library_id: str, embedding_model: str, chat_model: str, api_key: str):
    normalized_model = embedding_model if embedding_model.startswith("models/") else f"models/{embedding_model}"
    return LangChainRAG(
        documents, embedding_model=normalized_model, chat_model=chat_model, api_key=api_key,
        cache_path="rag_history.db", library_id=library_id,
    )

for key, default in {"passages": [], "messages": [], "library_key": None, "pipeline": None}.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.header("Your source library")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        st.caption("🔒 Gemini API key configured")
    else:
        st.error("Gemini API key is not configured. Add GEMINI_API_KEY to your .env file.")
    uploads = st.file_uploader(
        "Upload sources", type=["pdf", "txt", "md", "docx"], accept_multiple_files=True
    )
    chat_model = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.7-flash")
    embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    ocr_model = os.getenv("GEMINI_OCR_MODEL", chat_model)
    strategy = st.segmented_control(
        "Retrieval view", ["compare", "semantic", "fixed"],
        format_func=lambda value: {
            "compare": "Compare both", "semantic": "Semantic", "fixed": "Fixed",
        }[value],
        default="compare",
    )

    cache = get_rag_cache()
    saved_libraries = cache.libraries()
    if saved_libraries:
        st.subheader("Saved libraries")
        saved_by_id = {library_id: name for library_id, name in saved_libraries}
        selected_library = st.selectbox(
            "Load an indexed library", list(saved_by_id),
            format_func=lambda library_id: saved_by_id[library_id],
        )
        if st.button("Load saved library", width="stretch", disabled=not api_key):
            try:
                with st.spinner("Loading cached index…"):
                    documents = cache.load_library(selected_library)
                    if not documents:
                        raise ValueError("The saved library has no readable documents.")
                    pipeline = create_pipeline(
                        documents, selected_library, embedding_model, chat_model, api_key
                    )
                    st.session_state.pipeline = pipeline
                    st.session_state.passages = [as_passage(doc) for doc in pipeline.chunks["semantic"]]
                    st.session_state.library_key = selected_library
                    st.session_state.messages = []
                st.rerun()
            except Exception as error:
                st.error(f"Could not load the saved library: {error}")

    if uploads:
        upload_data = [(item.name, item.getvalue()) for item in uploads]
        digest = hashlib.sha256()
        for name, data in upload_data:
            digest.update(name.encode("utf-8"))
            digest.update(data)
        library_key = digest.hexdigest()
        if library_key != st.session_state.library_key:
            if not api_key:
                st.warning("Add your API key to index these files.")
            elif st.button("Build source library", type="primary", width="stretch"):
                try:
                    with st.spinner("Reading and indexing your sources…"):
                        client = genai.Client(api_key=api_key)
                        documents = documents_from_files(upload_data, ocr_client=client, ocr_model=ocr_model)
                        if not documents:
                            raise ValueError("No readable text was found in these files.")
                        cache.save_library(library_key, ", ".join(name for name, _ in upload_data), documents)
                        pipeline = create_pipeline(
                            documents, library_key, embedding_model, chat_model, api_key
                        )
                        st.session_state.pipeline = pipeline
                        st.session_state.passages = [as_passage(doc) for doc in pipeline.chunks["semantic"]]
                        st.session_state.library_key = library_key
                        st.session_state.messages = []
                    st.success(f"Ready: {len(uploads)} files, {len(st.session_state.passages)} passages")
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not build the library: {error}")

    if st.session_state.passages:
        passages: list[Passage] = st.session_state.passages
        names = sorted({passage.filename for passage in passages})
        fixed_count = len(st.session_state.pipeline.chunks["fixed"])
        semantic_count = len(st.session_state.pipeline.chunks["semantic"])
        st.success(f"{len(names)} sources · {fixed_count} fixed / {semantic_count} semantic chunks")
        for name in names:
            st.caption(f"• {name}")
        if st.button("Clear conversation", width="stretch"):
            st.session_state.messages = []
            st.rerun()

if not st.session_state.passages:
    st.info("Upload one or more PDF, DOCX, TXT, or Markdown files in the sidebar to begin.")
    col1, col2, col3 = st.columns(3)
    col1.markdown("**Ask for evidence**\n\n“What caused this uprising?”")
    col2.markdown("**Compare accounts**\n\n“How do these authors disagree?”")
    col3.markdown("**Study smarter**\n\n“Create a timeline from these notes.”")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("comparison"):
            show_comparison(message["comparison"])
        else:
            st.markdown(format_model_response(message["content"]))
        if message.get("sources"):
            with st.expander("View retrieved evidence"):
                show_sources(message["sources"])

question = st.chat_input("Ask a question about your sources…", disabled=st.session_state.pipeline is None)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the archive…"):
                if strategy == "compare":
                    comparison = {}
                    for strategy_name in ("fixed", "semantic"):
                        result = st.session_state.pipeline.ask(question, strategy_name)
                        comparison[strategy_name] = {
                            "answer": format_model_response(result["answer"]),
                            "sources": [as_passage(doc) for doc in result["context"]],
                        }
                else:
                    result = st.session_state.pipeline.ask(question, strategy)
                    context = [as_passage(doc) for doc in result["context"]]
                    answer = format_model_response(result["answer"])
            if strategy == "compare":
                show_comparison(comparison)
                st.session_state.messages.append({"role": "assistant", "comparison": comparison})
            else:
                st.markdown(answer)
                with st.expander("View retrieved evidence"):
                    show_sources(context)
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": context})
        except Exception as error:
            st.error(f"I couldn't answer that question: {error}")
