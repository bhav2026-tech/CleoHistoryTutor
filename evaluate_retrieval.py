"""Run a labelled, apples-to-apples retrieval comparison from the command line."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_rag import EvaluationQuery, LangChainRAG, compare_retrieval, documents_from_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="JSON file containing documents and labelled queries")
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args()
    load_dotenv()
    payload = json.loads(args.dataset.read_text(encoding="utf-8"))
    base = args.dataset.parent
    files = [(name, (base / name).read_bytes()) for name in payload["documents"]]
    queries = [EvaluationQuery(item["question"], tuple(item["relevant_text"])) for item in payload["queries"]]
    model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    if not model.startswith("models/"):
        model = f"models/{model}"
    pipeline = LangChainRAG(
        documents_from_files(files), embedding_model=model, top_k=args.k,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    print(json.dumps(compare_retrieval(pipeline, queries, args.k), indent=2))


if __name__ == "__main__":
    main()
