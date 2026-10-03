# RAG Pipeline — Week 5 (built from scratch, complete)

Week 5 builds retrieval-augmented generation without a framework, so every
step is visible. **Days 28-34 are all done** — ingestion through evaluation,
plus the Day 34 failure analysis and oral defense write-up at
[`docs/WEEK5_DEFENSE.md`](docs/WEEK5_DEFENSE.md).

```
Document -> parse -> clean -> chunk -> embed -> pgvector index      <- Day 29
Question -> embed -> retrieve top-k (+ metadata filters)            <- Day 30
         -> build prompt -> LLM -> cited, verified answer           <- Day 31
Eval     -> 30-question frozen set -> Precision@k, Recall@k, MRR     <- Day 32
Grid     -> chunk_size x top_k, 9 runs, one variable at a time       <- Day 33
Defense  -> failure analysis + 10-minute oral defense script         <- Day 34 (this commit)
```

## Run it

No API key needed -- embeddings run locally (see below).

```bash
docker compose up --build
```

The first document you upload will be slow (10-30s): fastembed downloads its
~130MB model on first use. It's cached in a Docker volume afterward, so
every upload after that is fast, and restarting the containers doesn't
re-download it.

Open **http://127.0.0.1:8000/** — the *chunk lab*. Paste text and change chunk
size / overlap to see chunks re-cut live (highlighted text is the overlap), or
upload a `.txt`, `.md` or `.pdf` to run the real pipeline: parse, clean, chunk,
embed, and store. The page is served by the API itself, so there is no CORS
setup and nothing to build. API docs are at `/docs`.

Tests need no Docker or API key: `pip install -r requirements.txt && pytest -v`
(the embedding provider is mocked, same "fast fake" approach as every other
provider call in this sprint).

## What each stage does, and why

**Parse** (`ingestion/parsers.py`) — file bytes to pages. PDFs keep their page
number; txt/md are a single page with no number.

**Clean** (`ingestion/cleaning.py`) — runs *before* chunking, so stored offsets
refer to the text we actually keep. It undoes hard line-wrapping, repairs
words split by a hyphen at a line break, folds ligatures (`ﬁ` -> `fi`) and
non-breaking spaces with Unicode NFKC, and strips control characters. Noise left
in here would end up inside embeddings and in the LLM's prompt.

**Chunk** (`ingestion/chunking.py`) — packs whole sentences (or list/heading
lines) into chunks up to `chunk_size` tokens, repeating the last
`chunk_overlap` tokens at the start of the next chunk. A single sentence
bigger than `chunk_size` falls back to being split by words.

**Embed** (`core/embeddings.py`, Day 29) — every chunk's text is embedded
LOCALLY via `fastembed` (ONNX Runtime, model `BAAI/bge-small-en-v1.5`, 384
dimensions) and stored on the chunk row. This is the ONLY place in the app
that generates embeddings, mirroring `llm_client.py` from the Week 4
project: one place to swap the model, one place to mock in tests. Since
fastembed is synchronous and CPU-bound, it runs in a worker thread
(`asyncio.to_thread`) rather than blocking the event loop. A model failure
is raised as `EmbeddingError` (502) — nothing is persisted for a document
whose embeddings couldn't be generated.

**Why local instead of a hosted embeddings API:** no API key, no billing,
no provider outage to handle, and no network dependency at request time —
at the cost of somewhat lower embedding quality than a large hosted model,
and a one-time ~130MB model download on first use (Groq, notably, has no
embeddings endpoint at all, despite some third-party client libraries
implying otherwise — chat, audio, and TTS only).

**Store** (`models/models.py`, `alembic/versions/8a1c2f9d4b3e_...py`) — the
embedding lives in a real `pgvector` column on Postgres, with an `ivfflat`
cosine-distance index built by the Day 29 migration
(`CREATE EXTENSION vector`, then the index). In tests, the same column falls
back to a plain JSON array under SQLite (`Vector(...).with_variant(JSON(),
"sqlite")`), since pgvector has no SQLite backend — same data, no real
Postgres or API key needed to run the suite.

**Retrieve** (`services/retrieval_service.py`, `routers/retrieval.py`, Day 30)
— `POST /retrieve` embeds the query with the same local model used at
ingestion, then returns the `top_k` chunks with the smallest cosine
distance. Metadata filters (`document_id`, `source_type`) narrow the
candidate set *before* ranking, not after — so `top_k` comes from the
filtered set instead of being padded out by irrelevant documents that
happened to rank low. Deliberately no LLM call in this endpoint: keeping
retrieval separate means its quality (did the right chunk come back at
all?) can be evaluated independently of answer quality (Day 31+).

On Postgres, the distance is computed *inside the database* via pgvector's
cosine-distance operator, using the Day 29 `ivfflat` index — an approximate
nearest-neighbor search, not a full scan, which is what makes this viable
at real corpus sizes. The test suite (SQLite) has no pgvector operator to
call, so it fetches the (test-scale) candidate rows and ranks them in
Python instead — correct for a handful of rows, and a direct illustration
of why the Postgres path needs an index rather than doing the same thing
at scale.

**A real bug caught and fixed while building this:** the first version
returned zero results against a real, populated Postgres database, with
no error at all. The cause was `ivfflat`'s `lists`/`probes` trade-off: the
Day 29 index splits the table into `lists = 100` clusters and, by
default, a search probes only **1** of them. On a small corpus (a handful
of documents, nowhere near 100 rows worth of real clustering), that one
probed cluster can easily contain none of the actual nearest vectors —
so the query runs successfully and returns nothing, which is a much
harder failure to notice than an error would be. Fixed with
`SET LOCAL ivfflat.probes = 100` before the query, scoped to that one
transaction so it can't leak onto a pooled connection reused by another
request. The trade-off: more probes means slower (but more accurate)
search — the right number for a real corpus depends on its size and
would be revisited once actual data volume is known, exactly the kind of
tuning the Day 33 chunk-size/top-k experiment log is for.

```bash
curl -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "what does the chunk lab do?", "top_k": 3}'
```

**Answer with citations** (`services/generation_service.py`, `routers/ask.py`,
Day 31) — `POST /ask` runs the full pipeline: retrieve top-k chunks, build a
prompt that numbers and labels each one with its source (`[1] (handbook.pdf,
page 4)`), instruct the model to answer only from that numbered context and
cite every claim, then generate the answer.

This needs a real LLM, unlike embeddings — set `LLM_API_KEY` in `.env` (see
`.env.example`; defaults to Groq's chat endpoint, which is a genuinely
supported Groq feature, unlike their nonexistent embeddings endpoint). If
retrieval finds nothing, the LLM is never called at all: the refusal
("I don't know based on the provided documents.") is deterministic, not
hoped-for from the model.

**The citations are verified, not trusted.** A model told "never cite a
number that isn't in the context" can still do it anyway, so every `[N]` in
the answer is checked against the chunks that were actually retrieved.
A fabricated citation is dropped from the structured `citations` list and
flips `all_citations_valid` to `false` — the response tells you plainly
whether the model's citations can be trusted, rather than assuming they
can.

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "what does the chunk lab do?", "top_k": 3}'
```

**Evaluation** (`eval/`, Day 32) — `eval/eval_set.json` is a frozen set of 30
questions (10 factoid, 6 paraphrased, 6 loosely/ambiguously worded, 8
no-answer) against a small synthetic 3-document corpus
(`eval/corpus/`: a leave policy, a product FAQ, and a page of science
facts — picked specifically so the three are topically disjoint and
cross-document retrieval mistakes are obvious). Each labeled question is
tagged with its correct source document and an exact short phrase
(`gold_span`) that must appear in a correct retrieval.

`eval/run_eval.py` ingests the corpus, runs retrieval for every labeled
question, checks whether `gold_span` shows up in the top-k results, and
prints Precision@k, Recall@k, and MRR — overall and broken down by
question type. No-answer questions are reported separately (there's no
single "correct chunk" to score them against); instead it prints each
one's top-1 distance, so you can eyeball whether there's a gap that could
become a similarity threshold later.

```bash
docker compose cp eval api:/code/eval
docker compose exec api python eval/run_eval.py
```

**Why a single-relevant-item simplification, and why that's named
explicitly:** each question is scored against exactly one correct
passage, found by exact substring match, rather than a human-labeled set
of every passage that could answer it. This keeps the eval set cheap to
build and exactly reproducible, at the cost of not capturing questions
with multiple valid answers. Every number this script prints should be
read with that trade-off in mind — which is itself a legitimate thing to
say out loud in an interview, rather than presenting the metrics as more
rigorous than they are.

`run_eval.py` takes `--chunk-size`, `--chunk-overlap`, and `--top-k` as
command-line flags specifically so Day 33's chunk-size × k experiment
grid can call this exact script repeatedly with different values, rather
than being written from scratch. It also tracks retrieval latency
(average and p95) and an estimated prompt token count per question
(`system prompt + question + every retrieved chunk's stored token_count`),
both written into `eval/results_latest.json` alongside the quality metrics.

**Chunk-size × top-k grid** (`eval/run_grid.py`, Day 33) — calls
`run_eval.py`'s `run()` nine times, once per combination of
`chunk_size ∈ {250, 500, 1000}` × `top_k ∈ {3, 5, 10}`, holding overlap at
a fixed 10% of chunk_size so it never becomes a hidden third variable.
Everything else — the eval set, the embedding model, the prompt — stays
identical across all nine runs, which is the one rule Day 33 actually
insists on: change one thing at a time (or the full grid at once), never
let two variables move together, or a result can't be attributed to a
cause.

```bash
docker compose cp eval api:/code/eval
docker compose exec api python eval/run_grid.py
```

Prints a single summary table (Precision@k, Recall@k, MRR, average prompt
tokens, p95 latency, for all nine cells) and writes `eval/grid_results.json`
for later comparison. Each document gets re-ingested fresh at every new
chunk_size — Day 28's dedupe key is content hash *plus* chunk settings, so
a different chunk_size is correctly treated as a new ingestion, not a
duplicate; this is the dedupe design working as intended, not a special
case for the grid.

**A known artifact in this specific grid, named rather than hidden:** the
eval corpus is 3 short synthetic documents. At `chunk_size=1000`, each
document becomes only 1-2 chunks — fewer than `top_k=10` can even return.
Precision@10 on that row is mechanically capped at (total chunks)/10, not
a genuine signal about retrieval quality at that setting; the script
prints a `NOTE` line whenever this applies, and the grid summary repeats
the reminder at the bottom. Reading a metric without checking whether its
denominator assumption holds is a very easy mistake to make, and worth
calling out explicitly rather than presenting a misleading number
cleanly.

## The trade-offs (the interview-relevant part)

- **Chunk size.** Small chunks are precise but may lack the context to answer
  anything. Large chunks carry context but blur several topics into one
  embedding, so a match no longer tells you *which* part was relevant; they also
  spend more of the LLM's context window per retrieved chunk.
- **Overlap.** Protects against an answer straddling a boundary. Costs storage
  and embedding work (more chunks) and can return near-duplicate results.
- **Batching embeddings.** Fewer HTTP round-trips and lower latency per
  document, at the cost of one failure (after retries) discarding the whole
  document's ingestion rather than partially embedding it. Chosen because a
  half-embedded document is worse than a clean retry.
- **ivfflat vs exact search.** ivfflat is approximate — it trades a small
  amount of recall for speed at scale, unlike a brute-force scan over every
  vector. `lists = 100` is a placeholder; the right value depends on corpus
  size and gets revisited once real data volume is known (Day 30 retrieval
  evaluation touches this directly).
