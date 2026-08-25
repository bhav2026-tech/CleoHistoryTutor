"""Small, inspectable retrieval pipeline for the History Tutor app."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, TypeVar

from docx import Document
from google import genai
from google.genai import types
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}
T = TypeVar("T")


def is_retryable_api_error(error: Exception) -> bool:
    details = str(error).upper()
    return any(
        marker in details
        for marker in ("429", "500", "502", "503", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED")
    )


def call_with_retry(operation: Callable[[], T], attempts: int = 4) -> T:
    """Retry transient Gemini capacity and service errors with short backoff."""
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as error:
            if attempt == attempts - 1 or not is_retryable_api_error(error):
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Gemini request failed after retries.")


@dataclass(frozen=True)
class Passage:
    id: str
    filename: str
    location: str
    text: str
    embedding: list[float] | None = None


def ocr_pdf(client: genai.Client, data: bytes, model: str) -> list[tuple[str, str]]:
    """Extract page-numbered text from an image-only PDF with Gemini."""
    contents = [
        types.Part.from_bytes(data=data, mime_type="application/pdf"),
        "Transcribe all readable text in this PDF exactly. Preserve paragraph order. "
        "Return every page, including pages with no text. Do not summarize, explain, or add facts.",
    ]
    config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
        response_json_schema={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "page": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["page", "text"],
            },
        },
    )
    models = list(dict.fromkeys([model, "gemini-3.6-flash"]))
    response = None
    for index, candidate in enumerate(models):
        try:
            response = call_with_retry(
                lambda candidate=candidate: client.models.generate_content(
                    model=candidate,
                    contents=contents,
                    config=config,
                )
            )
            break
        except Exception as error:
            if index == len(models) - 1 or not is_retryable_api_error(error):
                raise
    if response is None:
        raise RuntimeError("Gemini OCR was unavailable.")
    try:
        pages = json.loads(response.text or "[]")
        sections = [
            (f"page {int(page['page'])}", str(page["text"]).strip())
            for page in pages
            if str(page.get("text", "")).strip()
        ]
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise RuntimeError("Gemini returned an unreadable OCR response.") from error
    if not sections:
        raise ValueError("Gemini OCR could not find readable text in this PDF.")
    return sections


def read_document(
    filename: str,
    data: bytes,
    ocr_client: genai.Client | None = None,
    ocr_model: str = "gemini-3.7-flash",
) -> list[tuple[str, str]]:
    """Return (location, text) sections from one supported document."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")

    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        sections = [(f"page {i}", page.extract_text() or "") for i, page in enumerate(reader.pages, 1)]
        if any(text.strip() for _, text in sections):
            return sections
        if ocr_client is None:
            return sections
        return ocr_pdf(ocr_client, data, ocr_model)
    if suffix == ".docx":
        document = Document(io.BytesIO(data))
        paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
        return [("document", "\n".join(paragraphs))]

    text = data.decode("utf-8", errors="replace")
    return [("document", text)]


def chunk_text(text: str, chunk_size: int = 1_200, overlap: int = 180) -> list[str]:
    """Split text near paragraph/sentence boundaries with a small overlap."""
    clean = re.sub(r"[ \t]+", " ", text)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    if not clean:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        if end < len(clean):
            candidates = [clean.rfind(mark, start + chunk_size // 2, end) for mark in ("\n\n", ". ", "? ", "! ")]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + (2 if clean[boundary : boundary + 2] in {". ", "? ", "! "} else 0)
        piece = clean[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(clean):
            break
        start = max(end - overlap, start + 1)
    return chunks


def passages_from_files(
    files: Iterable[tuple[str, bytes]],
    ocr_client: genai.Client | None = None,
    ocr_model: str = "gemini-3.7-flash",
) -> list[Passage]:
    passages: list[Passage] = []
    for filename, data in files:
        file_hash = hashlib.sha256(data).hexdigest()[:16]
        for location, text in read_document(filename, data, ocr_client, ocr_model):
            for index, chunk in enumerate(chunk_text(text), 1):
                passage_id = f"{file_hash}:{location}:{index}"
                passages.append(Passage(passage_id, filename, location, chunk))
    return passages


class EmbeddingCache:
    def __init__(self, path: str = "rag_history.db") -> None:
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (id TEXT, model TEXT, vector TEXT, PRIMARY KEY (id, model))"
        )
        self.connection.commit()

    def get(self, passage_id: str, model: str) -> list[float] | None:
        row = self.connection.execute(
            "SELECT vector FROM embeddings WHERE id = ? AND model = ?", (passage_id, model)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, passage_id: str, model: str, vector: list[float]) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO embeddings (id, model, vector) VALUES (?, ?, ?)",
            (passage_id, model, json.dumps(vector)),
        )
        self.connection.commit()


def embed_passages(
    client: genai.Client, passages: list[Passage], model: str, cache: EmbeddingCache
) -> list[Passage]:
    vectors: dict[str, list[float]] = {}
    missing: list[Passage] = []
    for passage in passages:
        cached = cache.get(passage.id, model)
        if cached is None:
            missing.append(passage)
        else:
            vectors[passage.id] = cached

    for offset in range(0, len(missing), 100):
        batch = missing[offset : offset + 100]
        contents = [
            types.Content(role="user", parts=[types.Part.from_text(text=item.text)])
            for item in batch
        ]
        response = call_with_retry(
            lambda: client.models.embed_content(
                model=model,
                contents=contents,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=768,
                ),
            )
        )
        for item, result in zip(batch, response.embeddings or []):
            vector = list(result.values or [])
            vectors[item.id] = vector
            cache.put(item.id, model, vector)

    if any(item.id not in vectors for item in passages):
        raise RuntimeError("Gemini did not return embeddings for every source passage.")

    return [Passage(p.id, p.filename, p.location, p.text, vectors[p.id]) for p in passages]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def retrieve(client: genai.Client, question: str, passages: list[Passage], model: str, limit: int = 6) -> list[Passage]:
    response = call_with_retry(
        lambda: client.models.embed_content(
            model=model,
            contents=question,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768,
            ),
        )
    )
    if not response.embeddings:
        raise RuntimeError("Gemini did not return an embedding for the question.")
    query = list(response.embeddings[0].values or [])
    ranked = sorted(
        passages,
        key=lambda passage: cosine_similarity(query, passage.embedding or []),
        reverse=True,
    )
    return ranked[:limit]


def answer_question(client: genai.Client, question: str, context: list[Passage], model: str) -> str:
    sources = "\n\n".join(
        f"[Source {index}: {item.filename}, {item.location}]\n{item.text}"
        for index, item in enumerate(context, 1)
    )
    instructions = """You are Clio, a careful and encouraging history tutor.
Answer using only the supplied sources. Explain cause, consequence, chronology, and historical context when useful.
Cite every factual claim with inline citations like [Source 1]. You may combine citations.
If the sources do not contain enough information, say exactly what is missing; never fill gaps from memory.
Distinguish facts in the sources from interpretations, and mention conflicts between sources.
Keep the answer clear and suitable for a student, then end with one optional follow-up question that promotes deeper thinking."""
    response = call_with_retry(
        lambda: client.models.generate_content(
            model=model,
            contents=f"SOURCES\n{sources}\n\nQUESTION\n{question}",
            config=types.GenerateContentConfig(
                system_instruction=instructions,
                temperature=0.2,
            ),
        )
    )
    return response.text or "I couldn't generate an answer."
