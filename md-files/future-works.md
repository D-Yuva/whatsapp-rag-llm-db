## Future Work

- [ ] **Add optional time halo (context expansion)**

  - **Why:** Some key hints live a few minutes around a hit (e.g., “they also asked time complexity” right after “sieve”).
  - **Design:** After picking top vector hits, include any messages within ±N minutes of those hits.
  - **Config:** `HALO_MINUTES` (default 0 = off), `HALO_MAX_LINES` (cap to avoid bloat).
  - **Acceptance:** When halo is on, answers show extra nearby lines; when off, output remains identical to current behavior.

- [ ] **Better query translation (planning)**

  - **Why:** Increase recall/precision without more tokens.
  - **Plan:**
    - **Course-level (code-planned):** If user asks for a known course (e.g., “ACC”), generate 8–12 queries directly from `topics.json` (no LLM planning call).
    - **Paraphrase burst:** For each seed query, add 2–3 paraphrases (e.g., “round up decimals” → “ceil value”, “round to next int”).
    - **Cap & fuse:** Limit to ≤12 queries; use max-fusion over scores.
  - **Config:** `PLAN_MODE={auto|code|llm}`, `MAX_PLAN_QUERIES=12`, `PER_QUERY_K=20`, `TOP_K=60`.
  - **Acceptance:** More specific lines retrieved (e.g., “3 primes sum to n”) without extra junk.

- [ ] **Handle larger data via ANN (HNSW)**
  - **Why:** Keep latency snappy as messages grow beyond ~100k.
  - **Design:** Keep SQLite as metadata; attach a FAISS HNSW index for vectors.
  - **Plan:**
    - Add optional FAISS backend: `faiss.IndexHNSWFlat(dim, M=32); index.hnsw.efConstruction=200; efSearch configurable`.
    - Persist to `vectors/faiss_hnsw.idx` and maintain an `id <-> row_id` mapping table.
    - On upsert, add new vectors to FAISS; rebuild index only if needed.
  - **Config:** `USE_FAISS=1`, `FAISS_BACKEND=hnsw`, `HNSW_M=32`, `HNSW_EF_SEARCH=64`.
  - **Install:** `pip install faiss-cpu` (or `faiss-gpu` on supported setups).
  - **Acceptance:** Similar or better top-k quality vs brute-force, with significant speedup at large N.
