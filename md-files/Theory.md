# Why `whatsapp-rag-llm-db`?

A local, privacy-first assistant that turns messy WhatsApp chats into **answerable knowledge**.  
It embeds your chat exports once, then—on demand—lets an LLM create smart queries, runs **semantic vector search** over your messages, and answers **strictly** from what was actually said (with verbatim receipts).

---

## The Problem

WhatsApp groups explode during exams, projects, and events:
- Important info is **buried** across thousands of short, noisy messages.
- Keyword search misses phrasing variants (e.g., “round up decimals” vs “ceil it”).
- Scrolling for **proof** (who said what, when) is slow and error-prone.
- Cloud tools are overkill, **not private**, or rate-limited.
- Exported `.txt` logs are unstructured and hard to mine.

---

## What This Project Solves

- **Precise retrieval** from your chats, not the web.  
  Ask: “Give me everything on ACC,” “primes only — strict,” or “ACC on 2025-08-21.”
- **Semantic matching** (not just keywords).  
  Finds “3 primes sum to n,” “round up decimals,” “time complexity,” etc., even with different wording.
- **Verbatim receipts** for trust.  
  Outputs exact lines like `[2025-08-21 11:51] Sakthi: MCQs are mixed, including basic Java, code correction, and time complexity`.
- **Topic-aware expansion** from your own portions file (`topics.json`).  
  The LLM turns “ACC” into many focused queries (e.g., “Simple Sieve,” “rounding decimals,” “pattern printing”) so you don’t miss subtopics.
- **Local & private by design.**  
  Uses **Ollama** and **SQLite**—no cloud uploads, no API keys required.

---

## Who Is It For?

- **Students / Exam prep**  
  Extract what matters from frantic study groups. Get question types, algorithms, and tips with exact proof lines.
- **University clubs & cohorts**  
  Summarize decisions, deadlines, and resources shared across large chats.
- **Project teams** (hackathons, small startups)  
  Recover requirements, blockers, and decisions posted informally in WhatsApp.
- **Anyone who needs receipts**  
  “Who shared the link?”, “When was the change announced?”, “What exactly did they ask?”

---

## Why This Approach (and not alternatives)?

- **Local Vector DB (SQLite + embeddings)**  
  - Fast, incremental, and durable; filter by chat and time **before** scoring.  
  - One row per message → always know `[time] sender: text`.  
  - No servers, no ops.
- **Topic-driven LLM planning**  
  - LLM reads `topics.json` and generates **multi-queries**; Python does retrieval.  
  - The final answer is grounded in `<RAG_CONTEXT>` only (no hallucinations).
- **Simple to operate**  
  - `ollama serve`, one `.env`, and a `topics.json`. That’s it.

**Why not…**
- **Keyword-only search?** Misses phrasing variations and synonyms.
- **ChatGPT on entire logs?** Too many tokens, slow, and ungrounded.
- **Cloud vector DBs?** Overkill + privacy concerns; local works great at your scale.

---

## Typical Use Cases

- “**Give me ACC topics discussed on 2025-08-21**”  
  → Returns the exact lines (complexity MCQs, code correction, etc.) with timestamps.
- “**ACC primes only — strict**”  
  → Focuses on “Simple Sieve”, “3 primes sum to n”-type lines; filters out unrelated chatter.
- “**Everything about Compiler Design between 09:00 and 18:00**”  
  → Filters by time first, then semantic retrieval, then concise summary + receipts.

---

## Design Principles

- **Privacy first:** all local; no external APIs by default.
- **Deterministic receipts:** every claim has a verbatim line.
- **Lightweight runtime:** two small LLM calls per question (plan → answer), no growing history.
- **Config-free topics:** edit `topics/topics.json`; no code changes needed.

---

## Limitations & When to Extend

- If your corpus grows past ~100k messages, add **FAISS** (IVF-PQ/HNSW) for faster ANN search.  
- If you want better precision on subtle lines, add a tiny **cross-encoder reranker** for the final 50 results.  
- For exact phrase/URL recall, add **hybrid search** (BM25 + vectors).

---

## One-line Summary

**`whatsapp-rag-llm-db`** is the simplest way to turn chaotic WhatsApp exports into **searchable, trustworthy answers**—locally, privately, and with receipts.

# Why SQL ?

	•	Local, file-based DB — no server to run, fully offline & private.
	•	One row per message: chat, sender, ts, text, dim, vec(BLOB) → everything needed to print exact [time] sender: message.
	•	Filter-before-score: fast WHERE chat=… AND ts BETWEEN … to shrink the candidate set before vector math.
	•	Dedup & upserts: keyed by sha1(text) so re-runs don’t re-embed the same message.
	•	Crash-safe, atomic writes: WAL mode keeps the DB consistent during incremental indexing.
	•	Incremental indexing: only new messages are embedded and inserted (quick startup after first run).
	•	Scales well for laptops