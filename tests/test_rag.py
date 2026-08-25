import json
from types import SimpleNamespace

from rag import chunk_text, cosine_similarity, ocr_pdf, passages_from_files


def test_short_text_stays_whole():
    assert chunk_text("The Congress of Vienna met in 1814.") == ["The Congress of Vienna met in 1814."]


def test_long_text_splits_with_overlap():
    chunks = chunk_text("First sentence. " * 200, chunk_size=200, overlap=30)
    assert len(chunks) > 2
    assert all(chunks)


def test_text_file_becomes_citable_passage():
    passages = passages_from_files([("notes.txt", b"The treaty was signed in 1919.")])
    assert passages[0].filename == "notes.txt"
    assert passages[0].location == "document"


def test_cosine_similarity():
    assert cosine_similarity([1, 0], [1, 0]) == 1
    assert cosine_similarity([1, 0], [0, 1]) == 0


def test_ocr_pdf_preserves_page_numbers():
    response = SimpleNamespace(text=json.dumps([
        {"page": 1, "text": "A declaration was issued."},
        {"page": 2, "text": "The assembly convened."},
    ]))
    models = SimpleNamespace(generate_content=lambda **kwargs: response)
    client = SimpleNamespace(models=models)

    assert ocr_pdf(client, b"fake pdf", "test-model") == [
        ("page 1", "A declaration was issued."),
        ("page 2", "The assembly convened."),
    ]
