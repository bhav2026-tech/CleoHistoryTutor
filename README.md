# Clio — source-grounded history tutor

Clio is a compact Retrieval-Augmented Generation (RAG) app. Upload history sources, ask a question, and receive an answer grounded only in the retrieved passages with inline citations.

## Features

- PDF, DOCX, Markdown, and plain-text uploads
- Automatic Gemini OCR fallback for scanned/image-only PDFs
- Page-aware citations for PDFs
- LangChain vector stores with Gemini embeddings
- Fixed-overlap and embedding-based semantic chunking modes
- A LangGraph `retrieve -> generate` workflow
- Source-only tutor prompt that admits when evidence is missing
- Expandable retrieved evidence beneath every answer
- A labelled retrieval benchmark comparing both strategies with Hit@k and MRR
- Side-by-side fixed versus semantic answers and retrieved evidence
- Persistent SQLite libraries, chunks, and embedding cache across app restarts

## Run locally

Python 3.10+ is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Put your Gemini API key from Google AI Studio in .env.
streamlit run app.py
```

## How the RAG flow works

1. `read_document` extracts text while retaining filename and page/location metadata. Image-only PDFs automatically fall back to Gemini OCR.
2. LangChain builds both a fixed 1,200-character baseline and semantic chunks. Semantic boundaries are selected where adjacent sentence embeddings have unusually high distance (85th percentile).
3. Each strategy gets its own in-memory vector store backed by the same Gemini embedding model.
4. LangGraph executes explicit retrieval and grounded-generation nodes.
5. The six closest passages are sent to the tutor prompt, which must cite them as `[Source N]`.

## Compare retrieval quality

Copy `evaluation.example.json`, list the source documents relative to that JSON file, and add several queries. Each `relevant_text` entry should be a distinctive excerpt from evidence that is relevant to its query. Then run:

```powershell
python evaluate_retrieval.py evaluation.json --k 6
```

Both retrievers receive exactly the same queries and cutoff. The report includes Hit@k (whether relevant evidence appeared) and MRR (how highly the first relevant passage ranked). Meaningful comparison requires multiple representative queries with human-labelled evidence; an unlabelled visual comparison cannot establish retrieval quality.

The app also defaults to **Compare both**, which displays the two answers and their retrieved evidence in parallel. Indexed libraries are saved in `rag_history.db`; after restarting the app, use **Load saved library** instead of uploading and indexing the files again.

Searchable files are parsed locally. Image-only PDFs are sent to Gemini for OCR; extracted passages and questions are then sent to Gemini for embedding and answer generation.

## Test

```powershell
python -m pip install pytest
python -m pytest -q
```
