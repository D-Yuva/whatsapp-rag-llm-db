## FAQ / Common Questions

- **Where is my data stored?**  
  All data stays local. The vector database is a single SQLite file at `vectors/embeddings.sqlite`. No cloud uploads.

- **What exactly is saved?**  
  One row per message: `chat, sender, ts (UTC), text, dim, vec (float32 blob)`. This lets the app print exact `[time] sender: message`.

- **Can I add more chats later?**  
  Yes. Run again with `--data "exports/*.txt"`. Only **new** messages (by `sha1(text)`) are embedded and inserted.

- **I changed `EMB_MODEL`. Do I need to reindex?**  
  Yes. Either delete `vectors/embeddings.sqlite` or set a new `VECTORS_DIR` so the app builds a fresh index.

- **How do I switch models in Ollama?**  
  Change `OLLAMA_MODEL` in `.env` (e.g., `qwen2.5:14b-instruct`, `llama3.1:8b-instruct`) and restart the app.

- **Does it work with multiple languages?**  
  Yes. The default embedder `intfloat/multilingual-e5-small` is multilingual and robust for short chat messages.

- **How do I restrict to a subtopic (e.g., “primes only”)?**  
  Ask for it explicitly (e.g., “ACC primes only — strict”). The LLM sets a semantic `restrict_to` filter so only closely related hits are returned.

- **How do I limit by date/time?**  
  Include it in your ask (e.g., “on 2025-08-21” or “between 09:00 and 18:00”). The app filters by `ts` before vector scoring.

- **Why no extra surrounding messages?**  
  By design there’s **no halo** (time-window expansion). You’ll get only the direct vector hits. Ask broader or increase `top_k` in your query if needed.

- **How do I update topics/portions?**  
  Edit `topics/topics.json`. The app reads it each run—no code changes or rebuilds needed.

- **Performance tips?**  
  Use `qwen2.5:14b-instruct` or `llama3.1:8b-instruct`, keep each turn to two small LLM calls, and avoid huge time ranges. Apple Silicon uses MPS automatically.

- **Can I back up or move the index easily?**  
  Yes. Copy the `vectors/` folder. That contains everything needed for retrieval.

- **What if retrieval misses an obvious line?**  
  Try a broader phrasing (“ACC coding questions”), add/adjust items in `topics.json`, or increase recall (e.g., ask for “higher top_k”).

- **I get `Ollama: address already in use`.**  
  Ollama is already running. That’s fine. Verify with `curl http://localhost:11434/api/tags`.

- **I get `LLM error: timed out`.**  
  First-token can be slow on big models. Use a smaller model or set `OLLAMA_TIMEOUT_SEC=600` in `.env`.

- **How do I reset everything?**  
  Delete `vectors/embeddings.sqlite` to reindex; or change `VECTORS_DIR` to build a fresh DB next to the old one.
