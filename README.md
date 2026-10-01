# RAG Pipeline: built from scratch (Week 5)

A retrieval-augmented generation (RAG) service built **without a framework**, so every step is visible: upload a document, split it into chunks, embed it locally, store the vectors in Postgres, then ask questions and get **cited, verified answers**.

| Day | What was built | Status |
|---|---|---|
| 28 | Parse, clean and chunk documents; store with metadata | Done |
| 29 | Local embeddings (fastembed) + pgvector column and index | Done |
| 30 | Top-k retrieval with metadata filters (`POST /retrieve`) | Done |
| 31 | Grounded answers with verified citations (`POST /ask`) | Done |
| 32-34 | Evaluation set, chunk-size / top-k experiments, failure analysis | Next |

## Architecture

```
INDEXING (once per document)
  upload -> parse -> clean -> chunk -> embed (local) -> store in Postgres + pgvector

QUERY (every question)
  question -> embed (same model) -> top-k nearest chunks (+ filters)
           -> numbered prompt -> LLM -> answer -> verify citations -> response
```

| Piece | Where it runs | Needs a key? |
|---|---|---|
| Embeddings (`BAAI/bge-small-en-v1.5`, 384 dims) | Locally, via fastembed / ONNX Runtime | No |
| Vector store | Postgres 16 + pgvector (Docker) | No |
| Answer generation | OpenAI-compatible chat API (Groq by default) | **Yes**, `LLM_API_KEY` |

## Quick start

**Requirements:** Docker Desktop.

1. Create your `.env` file next to `docker-compose.yml`:

   ```dotenv
   LLM_API_KEY=gsk_your_full_groq_key
   LLM_BASE_URL=https://api.groq.com/openai/v1
   LLM_MODEL=openai/gpt-oss-20b
   ```

   Get a key at console.groq.com. `.env` is git-ignored: never commit it or paste it anywhere public.

2. Start everything:

   ```bash
   docker compose up --build
   ```

3. Open:
   - **http://127.0.0.1:8000/** : the *chunk lab* (live chunk-size playground and file upload)
   - **http://127.0.0.1:8000/docs** : interactive API docs (easiest way to try every endpoint)

The **first upload is slow (10-30 s)**: fastembed downloads its ~130 MB model once. It is cached in a Docker volume, so later uploads and restarts are fast. Without a valid `LLM_API_KEY` everything works except `/ask`.

After changing `.env`, restart so the container picks it up:

```bash
docker compose down
docker compose up --build
```

## Try it

**Bash / macOS / Linux**

```bash
# 1. upload
curl -F "file=@notes.txt" "http://127.0.0.1:8000/documents?chunk_size=200&chunk_overlap=20"

# 2. retrieve the closest chunks (no LLM involved)
curl -X POST http://127.0.0.1:8000/retrieve -H "Content-Type: application/json" \
  -d '{"query": "what does the chunk lab do?", "top_k": 3}'

# 3. ask a question (retrieval + LLM)
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"query": "what does the chunk lab do?", "top_k": 3}'
```

**Windows PowerShell** (plain `curl` is an alias for something else, so use `curl.exe`)

```powershell
curl.exe -F "file=@notes.txt" "http://127.0.0.1:8000/documents?chunk_size=200&chunk_overlap=20"

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/retrieve -ContentType "application/json" `
  -Body '{"query":"what does the chunk lab do?","top_k":3}'

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask -ContentType "application/json" `
  -Body '{"query":"what does the chunk lab do?","top_k":3}'
```

The file must exist in the folder you run the command from (or use a full path).

**Example `/ask` response**

```json
{
  "query": "What is this document about?",
  "answer": "The document is a README for a software project ... [1][2][3]",
  "citations": [
    {"number": 1, "chunk_id": 4, "document_id": 1, "filename": "README.md",
     "page_number": null, "distance": 0.399}
  ],
  "all_citations_valid": true,
  "retrieved_count": 3
}
```

## API reference

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/documents` | Upload and ingest a `.txt`, `.md` or `.pdf` (max 10 MB) |
| `GET` | `/documents` | List documents |
| `GET` | `/documents/{id}` | One document |
| `GET` | `/documents/{id}/chunks` | Its chunks, with page numbers and offsets |
| `DELETE` | `/documents/{id}` | Delete a document and its chunks (`204`) |
| `POST` | `/chunk-preview` | Chunk text without saving (powers the chunk lab) |
| `POST` | `/retrieve` | Top-k nearest chunks, with filters |
| `POST` | `/ask` | Full pipeline: retrieve, generate, verify citations |
| `GET` | `/ping` | Health check |

**`POST /documents`** query parameters: `chunk_size` (default 500, range 5-4000) and `chunk_overlap` (default 50, must be smaller than `chunk_size`). The upload is a multipart form field named `file`.

**`POST /retrieve` and `POST /ask`** request body:

| Field | Type | Default | Notes |
|---|---|---|---|
| `query` | string | required | 1-2000 characters |
| `top_k` | int | 5 | 1-50 |
| `document_id` | int | none | Only search this document |
| `source_type` | string | none | `"pdf"`, `"txt"` or `"md"` |

`/retrieve` returns `results[]` with `chunk_id`, `document_id`, `filename`, `chunk_index`, `page_number`, `content` and `distance`. **`distance` is cosine distance: smaller is more relevant** (0 = identical direction, 2 = opposite). It is deliberately not converted to a "confidence" score.

`/ask` returns `answer`, `citations[]`, `all_citations_valid` and `retrieved_count`.

### Errors

Every failure uses one shape: `{"error": {"code": "...", "message": "...", "details": {...}}}`.

| Status | `code` | When |
|---|---|---|
| 404 | `not_found` | Unknown document id |
| 409 | `duplicate_document` | Same file already ingested with the same chunk settings (`details.existing_document_id`) |
| 413 | `payload_too_large` | File over 10 MB |
| 415 | `unsupported_file_type` | Not `.txt`, `.md` or `.pdf` |
| 422 | `invalid_document` | Empty file, unreadable or encrypted PDF, or no extractable text (scanned PDFs need OCR) |
| 422 | `invalid_chunk_params` | `chunk_overlap >= chunk_size` |
| 502 | `embedding_error` | The local embedding model failed |
| 502 | `provider_error` | The LLM call failed (`details.reason` has the provider's message) |

## Configuration

Set these in `.env`. Settings are read by the app (`app/core/config.py`).

| Variable | Default | Meaning |
|---|---|---|
| `LLM_API_KEY` | placeholder | Key for the chat provider |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | Any OpenAI-compatible endpoint |
| `LLM_MODEL` | `llama-3.1-8b-instant` in code | **Set `openai/gpt-oss-20b`** (see Troubleshooting) |
| `LLM_TIMEOUT_SECONDS` | 30 | Per-request timeout |
| `LLM_MAX_RETRIES` | 2 | Retries on timeouts, connection errors, 429 and 5xx |
| `LLM_RETRY_BACKOFF_SECONDS` | 0.5 | Doubles each retry |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model |
| `EMBEDDING_DIMENSIONS` | 384 | Must match the model **and** the database column |
| `EMBEDDING_BATCH_SIZE` | 64 | Texts per embedding batch |
| `DEFAULT_CHUNK_SIZE` / `DEFAULT_CHUNK_OVERLAP` | 500 / 50 | Used when the upload does not specify them |
| `MAX_UPLOAD_BYTES` | 10 MB | Upload cap |

In Docker, `docker-compose.yml` forwards only `DATABASE_URL`, `LLM_API_KEY`, `LLM_BASE_URL` and `LLM_MODEL` into the container. To change any other variable in Docker, add it under `api: environment:` in the compose file.

## Project structure

```
app/
  main.py                    FastAPI app, routers, error handler
  core/
    config.py                settings (env / .env)
    db.py                    async SQLAlchemy engine and session
    embeddings.py            the ONLY place that generates embeddings
    llm_client.py            the ONLY place that calls the chat LLM (retries live here)
    errors.py                domain errors -> one JSON error shape
  ingestion/                 pure Python: no HTTP, no database
    parsers.py               bytes -> pages (PDF keeps page numbers)
    cleaning.py              noise removal before chunking
    chunking.py              sentence-packed chunks with overlap and offsets
  models/models.py           Document and Chunk tables (+ pgvector column)
  repositories/              database access
  services/
    ingestion_service.py     validate, dedupe, parse, chunk, embed, store
    retrieval_service.py     embed query, nearest-neighbour search, filters
    generation_service.py    prompt, LLM call, citation verification
  routers/                   HTTP layer (documents, retrieval, ask, lab)
  static/chunk-lab.html      chunk playground UI
alembic/versions/            schema migrations (pgvector extension + index)
tests/                       60 tests, no Docker or API key needed
```

## How it works

**Parse** (`ingestion/parsers.py`): file bytes to pages. PDFs keep their page number; txt/md are one page with no number.

**Clean** (`ingestion/cleaning.py`): runs *before* chunking, so stored offsets refer to the text actually kept. It undoes hard line-wrapping, repairs words split by a hyphen at a line break, folds ligatures and non-breaking spaces (Unicode NFKC) and strips control characters. Noise left here would end up inside embeddings and in the LLM prompt.

**Chunk** (`ingestion/chunking.py`): packs whole sentences (or list/heading lines) into chunks of up to `chunk_size` tokens, repeating the last `chunk_overlap` tokens at the start of the next chunk. A single sentence larger than `chunk_size` falls back to being split by words. Chunks never span pages, so a citation's page number is exact.

**Embed** (`core/embeddings.py`): every chunk is embedded locally with `BAAI/bge-small-en-v1.5` (384 dimensions) and stored on its row. fastembed is synchronous and CPU-bound, so it runs in a worker thread (`asyncio.to_thread`) instead of blocking the event loop. The model is loaded once per process.

**Store** (`models/models.py`, Alembic migration `8a1c2f9d4b3e`): the embedding is a real pgvector column with an `ivfflat` cosine-distance index. Under SQLite (tests only) the same column falls back to a JSON array via `Vector(...).with_variant(JSON(), "sqlite")`.

**Retrieve** (`services/retrieval_service.py`): the query is embedded with the same model, then Postgres returns the `top_k` chunks with the smallest cosine distance using pgvector's operator. On the SQLite test path the candidates are ranked in Python instead.

**Generate** (`services/generation_service.py`): retrieved chunks are numbered and labelled, for example `[1] (handbook.pdf, page 4)`. The system prompt tells the model to answer only from that context, say "I don't know based on the provided documents." when it cannot, and cite every claim as `[N]`.

**Verify**: a model told never to invent a citation can still do it, so every `[N]` in the answer is checked against the chunks that were really retrieved. An invalid number is dropped from `citations` and sets `all_citations_valid` to `false`. If nothing is retrieved, the LLM is **not called at all** and the refusal is deterministic.

## Design decisions and trade-offs

- **Local embeddings instead of a hosted API.** No key, no billing, no outage to retry against, no network call at request time. Cost: somewhat lower quality than a large hosted model, a one-time ~130 MB download, and CPU load on your own server.
- **Embed before saving; one failure rejects the whole document.** A half-embedded document is worse than a clean retry. Cost: a large upload takes longer and cannot resume partway.
- **Embeddings and the LLM are separate, single-purpose modules.** One place to swap a provider and one place to mock in tests.
- **Retrieval is its own endpoint with no LLM.** Whether the right chunk came back can be evaluated independently of whether the model used it well.
- **Distance, not a similarity score.** It is honest about what the metric is.
- **ivfflat vs exact search.** ivfflat is approximate: it trades a little recall for speed at scale. `lists = 100` is a placeholder to revisit once real data volume is known (Day 33).
- **`SET LOCAL ivfflat.probes = 100`.** ivfflat probes only 1 of its `lists` clusters by default. On a small corpus that one cluster can miss every real match, so the query succeeds and returns *nothing*, with no error. Setting probes to 100 (equal to `lists`) fixes it. `SET LOCAL` applies to the current transaction only, so it cannot leak onto a pooled connection. With probes equal to `lists`, the search effectively scans every cluster; for a larger corpus, lower it and measure recall.
- **Chunk size.** Small chunks are precise but may lack context. Large chunks carry context but blur several topics into one embedding and use more of the LLM's context window. Overlap protects an answer that straddles a boundary, at the cost of more chunks and near-duplicate results.
- **Duplicate protection.** The key is the SHA-256 of the file plus the chunk settings, so the same file can be ingested at different chunk sizes (needed for the Day 33 experiments) but not twice at the same size.

## Tests

```bash
pip install -r requirements.txt
pytest -v
```

60 tests, no Docker, API key or model download needed. The suite runs on SQLite with the embedding model and the LLM mocked.

**Known test gap:** the pgvector SQL path (the distance operator, the ivfflat index, `SET LOCAL ivfflat.probes`) is not exercised by the suite, because SQLite has no vector operators. Verify it manually against the running Postgres, or add an integration test against the compose Postgres.

**Keep tests offline:** `mock_embeddings` in `tests/conftest.py` must patch **both** call sites. Add this line next to the existing patch, otherwise tests that call `/retrieve` or `/ask` without `install_topic_embeddings` load the real model and need internet:

```python
monkeypatch.setattr("app.services.retrieval_service.embed_texts", fake_embed_texts)
```

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| PowerShell: `A parameter cannot be found that matches parameter name 'F'` | `curl` is a PowerShell alias. Use `curl.exe`, or `Invoke-RestMethod`. |
| `curl: (26) Failed to open/read local data from file` | The file does not exist in the current folder. Create it or pass a full path. |
| `/ask` returns `provider_error` with `401 Invalid API Key` | The key is wrong, truncated or revoked. Put the full key in `.env`, then `docker compose down` and `up --build`. Check what the container received: `docker compose exec api printenv LLM_API_KEY`. |
| `/ask` returns `provider_error` with `404 model_not_found` | The model is retired or unavailable to your key. Groq retired `llama-3.1-8b-instant` for developer-tier use; set `LLM_MODEL=openai/gpt-oss-20b`. To list models your key can use: `docker compose exec api python -c "import os,json,urllib.request as u; r=u.Request('https://api.groq.com/openai/v1/models',headers={'Authorization':'Bearer '+os.environ['LLM_API_KEY'],'User-Agent':'curl/8'}); print([m['id'] for m in json.loads(u.urlopen(r).read())['data']])"` |
| First upload takes 10-30 s | Normal: the embedding model downloads once and is then cached. |
| `/retrieve` returns zero results on a populated database | The ivfflat probes setting (see Design decisions). It is already set to 100 in the code. |
| `409 duplicate_document` | That file was already ingested with these chunk settings. Use the id in `details.existing_document_id`, or upload with different settings. |
| `curl: not found` inside the container | The image has no `curl`. Use `python -c ...` as shown above. |

## Known limitations

These are real gaps, listed so they are not a surprise:

- **No context token budget.** Retrieved chunks are sent to the LLM without any limit. A large `top_k` times a large `chunk_size` could exceed the model's context or rate limits and surface as a `502`.
- **Retrieved text is not fenced off as untrusted.** A document containing instructions ("ignore the above...") could try to steer the model. Delimit the context clearly and tell the model to treat it as data.
- **No relevance threshold.** Retrieval always returns the `top_k` nearest chunks, even for an off-topic question, so the model is then relied on to refuse. A maximum-distance cutoff would let the app refuse deterministically.
- **Sampling is not set on the LLM call** (no `temperature` or `max_tokens`), so answers can vary between runs. A low temperature suits grounded Q&A.
- **Approximate token counting.** `chunk_size` is counted with a simple word-and-punctuation counter, which undercounts real model tokens. The embedding model accepts roughly 512 tokens, so a default 500-"token" chunk may be truncated when embedded. Prefer a smaller `chunk_size` or switch to the model's real tokenizer.
- **`lists = 100` is created on an empty table.** ivfflat builds its clusters from the data present at creation, so create or rebuild the index after loading data (`REINDEX`) and size `lists` to the corpus.
- **Parsing and chunking run on the event loop** (embedding already runs in a thread). Large files can slow other requests.
- **Embedding happens inside the upload request.** Fine for small files; large ones belong in a background job.
- **No OCR** (scanned PDFs are rejected), **no table or multi-column layout handling**, and **no authentication**.

## Security

- Keep `LLM_API_KEY` only in `.env`; it is git-ignored. If a key is ever pasted into chat, a screenshot or a commit, revoke it and create a new one.
- Treat uploaded documents as untrusted input (see the limitations above).
- Postgres is not published on a host port; only the API container can reach it.

## Next (Days 32-34)

1. A 30-50 question evaluation set, labelled by document, page and answer span (not chunk id, which changes with chunk settings).
2. Metrics: Precision@k, Recall@k and MRR, plus tokens and latency per query.
3. Experiment grid: chunk size 250 / 500 / 1000 against top-k 3 / 5 / 10.
4. Failure analysis and the 10-minute oral defence.
