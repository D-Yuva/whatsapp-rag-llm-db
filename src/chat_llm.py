import os, re, json, sqlite3
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
from dateutil import parser as dateparser
from dotenv import load_dotenv, find_dotenv
import httpx
import numpy as np
from pathlib import Path

from src.emb_store import EmbeddingStore
from src.whatsapp_chat_cleaning import load_exports

# ----------------------- ENV & LLM CLIENTS (GROQ or OLLAMA) -----------------------

load_dotenv(find_dotenv(), override=True)

# ---- Backend selection (default to Ollama since you don't use Groq) ----
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower().strip()  # "ollama" or "groq"


# ---- Ollama (local) ----
OLLAMA_BASE  = os.getenv("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct")
OLLAMA_THINK = os.getenv("OLLAMA_THINK", "false")  # disable chain-of-thought by default
OLLAMA_TIMEOUT_SEC = int(os.getenv("OLLAMA_TIMEOUT_SEC", "600"))  # allow slow first-token models

class LLMError(RuntimeError):
    pass

class BaseLLMClient:
    """Minimal interface so the rest of the app doesn't care which backend is used."""
    def chat(self,
             messages: List[Dict[str, str]],
             temperature: float = 0.2,
             max_tokens: int = 800) -> str:
        raise NotImplementedError

class GroqClient(BaseLLMClient):
    """Groq OpenAI-compatible chat client."""
    def chat(self,
             messages: List[Dict[str, str]],
             temperature: float = 0.2,
             max_tokens: int = 800) -> str:
        if not GROQ_API_KEY:
            raise LLMError("Set GROQ_API_KEY in your environment/.env (no quotes or spaces).")
        url = f"{GROQ_BASE.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        # simple retry for transient 429/5xx
        for attempt in range(3):
            try:
                with httpx.Client(timeout=60.0) as client:
                    r = client.post(url, headers=headers, json=payload)
                if r.status_code >= 400:
                    if r.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                        import time; time.sleep(1.5 * (attempt + 1))
                        continue
                    raise LLMError(f"Groq error {r.status_code}: {r.text}")
                data = r.json()
                return data["choices"][0]["message"]["content"]
            except httpx.HTTPError as e:
                if attempt < 2:
                    import time; time.sleep(1.0 * (attempt + 1))
                    continue
                raise LLMError(f"HTTP error: {e}")
        raise LLMError("Groq chat failed after retries")

class OllamaClient(BaseLLMClient):
    """Local Ollama chat client (no API key, default localhost:11434)."""
    def chat(self,
             messages: List[Dict[str, str]],
             temperature: float = 0.2,
             max_tokens: int = 800) -> str:
        url = f"{OLLAMA_BASE.rstrip('/')}/api/chat"
        # Ollama uses num_predict instead of max_tokens; None/0 -> small cap
        options: Dict[str, Any] = {
            "temperature": float(temperature),
            "num_predict": int(max_tokens) if max_tokens and max_tokens > 0 else 256,
        }
        # disable chain-of-thought so Ollama returns content instead of 'thinking'
        if OLLAMA_THINK.lower() in ("0", "false", "no"):
            options["think"] = False
        payload: Dict[str, Any] = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        # simple retry for transient local errors (e.g., model loading)
        for attempt in range(3):
            try:
                with httpx.Client(timeout=OLLAMA_TIMEOUT_SEC) as client:
                    r = client.post(url, json=payload)
                if r.status_code >= 400:
                    if r.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                        import time; time.sleep(1.5 * (attempt + 1))
                        continue
                    raise LLMError(f"Ollama error {r.status_code}: {r.text}")
                data = r.json()
                msg = data.get("message", {})
                content = msg.get("content")
                if not content:
                    raise LLMError(f"Ollama unexpected response: {data}")
                return content
            except httpx.HTTPError as e:
                if attempt < 2:
                    import time; time.sleep(1.0 * (attempt + 1))
                    continue
                raise LLMError(f"HTTP error: {e}")
        raise LLMError("Ollama chat failed after retries")

def get_llm_client() -> BaseLLMClient:
    if LLM_BACKEND == "groq":
        return GroqClient()
    # default
    return OllamaClient()

# ----------------------- RETRIEVAL ORCHESTRATION -----------------------

IST = timezone(timedelta(hours=5, minutes=30))

# Once the LLM gets the <RAG_CONTEXT> -> Retrieved messages
SYSTEM_PROMPT = (
    "Important: There are NO tools available. Do NOT emit tool_calls.\n"
    "Do NOT include chain-of-thought or 'thinking'; output only the fenced rag_query and the final answer.\n"
    "You are a focused WhatsApp retrieval assistant.\n"
    "You have a second system message named TOPICS_DB containing a JSON object of course headings mapped to lists of items (portions).\n"
    "\n"
    "Your task is to:\n"
    "1) Read and understand the TOPICS_DB JSON.\n"
    "2) Decide whether the user is asking a COURSE-LEVEL question (matches a heading exactly) or a SUBTOPIC-LEVEL question (matches one or more items).\n"
    "3) In the SUBTOPIC-LEVEL case, scan all headings and items until you find a match. Use the matching heading as 'topic' and set 'restrict_to' to the relevant items or phrases that focus the query.\n"
    "\n"
    "Decision logic before retrieval:\n"
    "a) If the user ask matches a TOPICS_DB heading (e.g., 'ACC', 'Compiler Design'), treat it as a COURSE-LEVEL request.\n"
    "b) Else, if the ask matches one or more TOPICS_DB items (e.g., 'primes', 'sieve', 'round up decimals'), treat it as a SUBTOPIC-LEVEL request.\n"
    "c) If neither exact match is found, scan ALL headings and items in TOPICS_DB for fuzzy textual overlap; choose the most plausible heading(s) and/or item(s). If still uncertain, fall back to user phrasing only.\n"
    "\n"
    "Emit one fenced block exactly like this for retrieval:\n"
    "```rag_query\n"
    "{\n"
    "  \"topic\": \"<heading-or-null>\",            \n"
    "  \"topic_candidates\": [\"<optional list>\"] ,\n"
    "  \"queries\": [\"<query 1>\", \"<query 2>\", \"<query 3>\", \"<...>\"],\n"
    "  \"restrict_to\": [\"<only for subtopic focus>\"] ,\n"
    "  \"restrict_threshold\": 0.28,\n"
    "  \"chat\": null,\n"
    "  \"start\": null,\n"
    "  \"end\": null,\n"
    "  \"per_query_k\": 40,\n"
    "  \"top_k\": 120\n"
    "}\n"
    "```\n"
    "\n"
    "Rules:\n"
    "- COURSE-LEVEL (e.g., 'ACC'): set topic to that heading; build MANY queries from the items under that heading in TOPICS_DB plus any user hints; keep restrict_to empty.\n"
    "- SUBTOPIC-LEVEL (e.g., 'primes', 'sieve'): identify the heading(s) where the item appears; set topic to the best heading and topic_candidates to any alternates; build queries centered on the subtopic; set restrict_to to focused forms (e.g., ['prime','primes','3 primes sum to n','sieve']).\n"
    "- If uncertain: propose topic_candidates from headings/items you scanned; still emit queries and (if appropriate) restrict_to.\n"
    "\n"
    "After I provide <RAG_CONTEXT>, answer strictly from those lines:\n"
    "- Summarize concisely and include a 'Verbatim receipts' section with exact messages [timestamp] sender: message.\n"
    "- If the user asks for 'everything' or 'just the messages', output only the verbatim list.\n"
    "Never fabricate content outside <RAG_CONTEXT>."
)

RAG_RE = re.compile(r"```rag_query\s*(\{.*?\})\s*```", re.DOTALL) # Regex to capture JSON inside ```rag_query ... ```

# Converts string into datetime, convert date and time into UTC
def to_epoch_utc(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    dt = dateparser.parse(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    return int(dt.astimezone(timezone.utc).timestamp())

# Converts UTC into IST
def pretty_ist(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(IST).strftime("%Y-%m-%d %H:%M")

# Detects and extracts a JSON rag_query block (LLM query → JSON for retrieval)
def extract_rag_query(text: str) -> Optional[Dict[str, Any]]:
    blocks = RAG_RE.findall(text or "")
    if not blocks:
        return None
    # If multiple, take the last one (LLM will create multiple queries and only the last which is the refined one)
    for js in reversed(blocks):
        try:
            return json.loads(js)
        except Exception:
            continue
    return None

def run_retrieval(store: EmbeddingStore, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Uses ONLY what the LLM asked for:
      - spec["queries"] : list of queries generated by the LLM (recommended)
      - spec["restrict_to"] : optional list of focus terms (subtopic-only mode)
    Vector-search is done for all queries jointly; scores are fused (max).
    """
    # ---- queries from LLM ----
    queries: List[str] = []
    if spec.get("queries"):
        queries = [q for q in spec["queries"] if isinstance(q, str) and q.strip()]
    elif spec.get("q"):  # backward-compat if model emits a single 'q'
        queries = [str(spec["q"]).strip()]

    # dedupe while preserving order
    seen = set()
    queries = [q for q in queries if not (q.lower() in seen or seen.add(q.lower()))]
    if not queries:
        return []

    # ---- strict focus (subtopic) ----
    restrict_terms = [t for t in (spec.get("restrict_to") or []) if isinstance(t, str) and t.strip()]
    restrict_threshold = float(spec.get("restrict_threshold") or 0.28)

    per_query_k   = int(spec.get("per_query_k") or 40)
    top_k_overall = int(spec.get("top_k") or 120)

    # (halo logic removed)

    chat     = spec.get("chat")
    start_ts = to_epoch_utc(spec.get("start"))
    end_ts   = to_epoch_utc(spec.get("end"))

    # ---- fetch all candidate message vectors once ----
    X, meta = store._fetch_vectors(chat, start_ts, end_ts)  # X: (n,d) normalized
    if X.size == 0:
        return []

    # ---- encode all queries and score ----
    M = store.model()
    Q = M.encode(queries, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)  # (m,d)
    scores = (Q @ X.T)  # (m,n) cosine similarities

    # ---- top per query -> candidate set ----
    cand = set()
    for qi in range(scores.shape[0]):
        topq = np.argsort(-scores[qi])[: max(1, per_query_k)]
        cand.update(int(i) for i in topq)

    # ---- fuse by max across queries ----
    fused: Dict[int, float] = {i: float(np.max(scores[:, i])) for i in cand}
    top_idx = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)[: max(1, top_k_overall)]

    # (halo expansion removed)

    # ---- strict subtopic filter (semantic) ----
    if restrict_terms:
        F = M.encode(restrict_terms, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)  # (r,d)
        kept = []
        for i in top_idx:
            v = X[i]  # (d,)
            foc_score = float(np.max(F @ v))  # cosine to any focus term
            if foc_score >= restrict_threshold:
                kept.append(i)
        top_idx = kept

    # ---- build rows to inject as <RAG_CONTEXT> ----
    rows: List[Dict[str, Any]] = []
    for i in top_idx:
        r = meta[i]
        rows.append({
            "ts": int(r["ts"]),
            "ts_str": pretty_ist(int(r["ts"])),
            "sender": r["sender"],
            "text": r["text"],
            "chat": r.get("chat"),
            "score": float(fused.get(i, 0.0)),
        })
    return rows

def build_context_block(rows: List[Dict[str, Any]]) -> str:
    lines = [f"[{row['ts_str']}] {row['sender']}: {row['text']}" for row in rows]
    return "<RAG_CONTEXT>\n" + "\n".join(lines) + "\n</RAG_CONTEXT>"

def ensure_index_if_empty(store: EmbeddingStore, data_globs: List[str]) -> None:
    con = sqlite3.connect(store.db_path)
    cur = con.cursor()

    # Checks if DB already has rows?
    cur.execute("SELECT COUNT(1) FROM msg")
    n = cur.fetchone()[0] # Checks if there is a row
    con.close()

    # If rows exists then don't do anything
    if n>0:
        return 
    
    # If row doesn't exist it will get from the .txt file
    rows = load_exports(data_globs)

    # If no rows exist
    if not rows:
        print("No messages found to index. Provide --data with paths to WhatsApp exports.")
        return 
    
    # if rows found, index them
    added = store.upsert_messages(rows)
    print(f"Indexed {added} messages (initial).")

 # --- Topics file (portions that the LLM can read) ---
def load_topics() -> Dict[str, Any]:
    """
    Loads topics from either:
      1) TOPICS_DB_JSON env (JSON string), or
      2) TOPICS_PATH file (default: topics.json), searched in CWD and project root.
    Returns {} on failure.
    """
    raw = os.getenv("TOPICS_DB_JSON")
    if raw:
        try:
            return json.loads(raw)
        except Exception as e:
            print(f"[topics] Failed to parse TOPICS_DB_JSON: {e}")
    path = os.getenv("TOPICS_PATH", "topics.json")
    p = Path(path)
    if not p.is_absolute():
        proj_root = Path(__file__).resolve().parents[1]
        for c in (Path.cwd() / p, proj_root / p):
            if c.exists():
                p = c
                break
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[topics] File not found: {p}")
    except Exception as e:
        print(f"[topics] Failed to load {p}: {e}")
    return {}

# ----------------------- MAIN LOOP -----------------------
def main():
    import argparse

    ap = argparse.ArgumentParser(description="LLM chat that retrieves from your WhatsApp vectors on demand")
    ap.add_argument("--data", nargs="+", help="Paths or globs to WhatsApp .txt exports (used only if DB is empty)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=800)
    args = ap.parse_args()

    store = EmbeddingStore(
        vectors_dir=os.getenv("VECTORS_DIR", "vectors"),
        model_name=os.getenv("EMB_MODEL")  # default in emb_store is e5-small
    )
    if args.data:
        ensure_index_if_empty(store, args.data)

    topics_db = load_topics()

    history: List[Dict[str, str]] = [
        {"role": "system", "name": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "name": "TOPICS_DB", "content": "TOPICS_DB:\\n" + json.dumps(topics_db, ensure_ascii=False)}
    ]
    llm = get_llm_client()

    print("Chat ready. Type your message. (Ctrl+C to quit)\n")
    try:
        while True:
            user = input("You: ").strip()
            if not user:
                continue
            history.append({"role": "user", "content": user})

            try:
                draft = llm.chat(history, temperature=args.temperature, max_tokens=args.max_tokens)
            except LLMError as e:
                print(f"LLM error: {e}")
                continue

            spec = extract_rag_query(draft)
            if spec:
                rows = run_retrieval(store, spec)
                ctx_block = build_context_block(rows)
                # Provide the retrieved messages as system context
                history.append({"role": "system", "content": ctx_block})
                try:
                    final = llm.chat(history, temperature=args.temperature, max_tokens=args.max_tokens)
                except LLMError as e:
                    print(f"LLM error: {e}")
                    continue
                print(f"\nAssistant:\n{final}\n")
                print(f"[context lines: {len(rows)}]\n")
                history.append({"role": "assistant", "content": final})
            else:
                # Model answered directly without retrieval
                print(f"\nAssistant:\n{draft}\n")
                history.append({"role": "assistant", "content": draft})

    except KeyboardInterrupt:
        print("\nBye!")

if __name__ == "__main__":
    main()
