import os, sqlite3, hashlib
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

def device_name() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"

#Provides a unique, fixed-length ID for each message text.
def sha1_text(s: str) -> str: 
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

# ------------------EMBEDDING STORE-----------------------
class EmbeddingStore:
    """Local vector store backed by SQLite. One row per message text."""

    def __init__(self, vectors_dir: str = "vectors", model_name: Optional[str] = None):
        self.vectors_dir = Path(os.getenv("VECTORS_DIR", vectors_dir))
        self.vectors_dir.mkdir(parents = True, exist_ok = True)
        self.db_path = self.vectors_dir / "embeddings.sqlite"

        self.model_name = model_name or os.getenv("EMB_MODEL")

        self._model: Optional[SentenceTransformer] = None
        self._init_db()

# ------------------DATABASE-----------------------
    def _init_db(self):
        con = sqlite3.connect(self.db_path)
        con.executescript(
            """
            PRAGMA journal_mode = WAL;
            CREATE TABLE IF NOT EXISTS msg(
            id_hash TEXT PRIMARY KEY,
            chat TEXT,
            sender TEXT, 
            ts INTEGER,
            text TEXT,
            dim INTEGER NOT NULL,
            vec BLOB NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_msg_ts ON msg(ts);
            CREATE INDEX IF NOT EXISTS idx_msg_chat ON msg(chat);
            
            CREATE TABLE IF NOT EXISTS meta(
                key TEXT PRIMARY KEY, 
                value TEXT
            );
            """
        )
        con.commit();
        con.close()
    
    def _get_meta(self, key: str) -> Optional[str]:
        con = sqlite3.connect(self.db_path)                  # 1. open the SQLite database file
        cur = con.cursor()                                   # 2. create a cursor (object to run SQL queries)
        cur.execute("SELECT value FROM meta WHERE key=?",    # 3. run a SQL query: 
                    (key,))                                     #"give me the value where meta.key = ?"
                                                                #the ? is filled with the given key safely
        row = cur.fetchone()                                 # 4. fetch one row (or None if no match)
        con.close()                                          # 5. close the database connection
                                            
        return row[0] if row else None    
    
    def _set_meta(self, key: str, value: str) -> None:
        con = sqlite3.connect(self.db_path)
        con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (key, value)) # Inserting the model name and dimension used, if neccassary it can replace to new model 
        con.commit();
        con.close()

    def _check_or_write_model_meta(self, dim: int) -> None:
        m_id = self._get_meta("model_name") # Gets model name from the database 'meta'
        m_dim = self._get_meta("model_dim") # Gets model dim from the database 'meta'

        if m_id is None and m_dim is None: # If nothing is stored in the database then append it
            self._set_meta("model_name", self.model_name)
            self._set_meta("model_dim", str(int(dim)))

            return 
        
        if m_id and m_id != self.model_name: # Raises an Error if Model name is different
            raise RuntimeError(
                f"Embedding DB was created with model '{m_id}', but current model is '{self.model_name}'.\n"
                f"Use a different VECTORS_DIR or delete {self.db_path} to re-index"
            )
        
        if m_dim and int(m_dim) != int(dim): # Raises an Error if dim is different
            raise RuntimeError(
                f"Embedding DB dim={m_dim} does not match current model dim={dim}.\n"
                f"Use a different VECTORS_DIR or delete {self.db_path} to re-index."
            )

# ------------------MODEL-----------------------
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device = device_name())
        return self._model

# ------------------INSERT / UPDATE CHATS-----------------------
    def have_ids(self, ids: List[str]) -> Dict[str, bool]:
        if not ids: return {}                                # Return empty dict if no ids provided
        con = sqlite3.connect(self.db_path)                  # Open SQLite database connection
        q = ",".join("?" * len(ids))                         # Create placeholders (?, ?, ...) for SQL query
        seen = {ih: False for ih in ids}                     # Initialize dict marking all ids as not found
        for (ih,) in con.execute(f"SELECT id_hash FROM msg WHERE id_hash IN ({q})", ids):
            seen[ih] = True                                  # Mark id as found if it exists in the table
        con.close()                                          # Close the database connection
        return seen                                          # Return dict with True/False for each id

    def upsert_message(self, rows: List[Dict]) -> int:
        """
        rows: [{chat, sender, ts, text}]
        Only embeds rows whose sha1(text) is not present. Vectors are normalized.
        """

        if not rows:
            return 0
        
        ids = [sha1_text((r.get("text") or "").strip()) for r in rows] # Normalises white spaces
        
        presence = self.have_ids(ids) # Returns {id_hash: True/False}
        
        # Appends only unique messages, if the message already exists its gonna skip
        to_do: List[Tuple[str, Dict]] = [
            (ih, r) for ih, r in zip(ids, rows)
            if ih and not presence.get(ih)
        ]
        if not to_do:
            return 0
        
        texts = [(r.get("text") or "").strip() for _, r in to_do]     # Collect clean message texts from rows to embed
        M = self.model()                                              # Load or reuse the embedding model
        X = M.encode(texts, batch_size=128, convert_to_numpy=True, normalize_embeddings=True)  # Encode texts into normalized numpy vectors
        X = X.astype(np.float32)                                      # Convert vectors to float32 for compact storage
        dim = X.shape[1]                                              # Get the embedding dimension (e.g., 384)

        self._check_or_write_model_meta(dim)                          # Ensure model name and dimension match DB meta

        con = sqlite3.connect(self.db_path)                           # Open SQLite connection
        con.executemany(                                              # Insert or replace message rows with metadata and vectors
            "INSERT OR REPLACE INTO msg(id_hash, chat, sender, ts, text, dim, vec) VALUES (?,?,?,?,?,?,?)",
            [
                (
                    ih,                                               # Unique id_hash (sha1 of text)
                    r.get("chat"),                                    # Chat name
                    r.get("sender"),                                  # Sender name
                    int(r.get("ts") or 0),                            # Timestamp (int, defaults to 0)
                    r.get("text"),                                    # Original message text
                    int(dim),                                         # Embedding dimension
                    X[i].tobytes(),                                   # Vector stored as raw bytes
                )
                for i, (ih, r) in enumerate(to_do)                    # Loop over new rows and pair with vectors
            ],
        )
        con.commit(); con.close()                                     # Save changes and close DB connection
        return len(to_do)                                             # Return count of new messages added

    # Back-compat: plural alias calls the original function
    def upsert_messages(self, rows: List[Dict]) -> int:
        return self.upsert_message(rows)


# ------------------FETCHING-----------------------
    def _fetch_vectors(
        self,
        chat: Optional[str],
        start_ts: Optional[int],
        end_ts: Optional[int],
    ) -> Tuple[np.ndarray, List[Dict]]:
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        sql = "SELECT id_hash, chat, sender, ts, text, dim, vec FROM msg WHERE 1=1"
        params: List = [] # List to hold query parameters
        if chat:
            sql += " AND LOWER(chat)=LOWER(?)" # Add case-insensitive chat condition
            params.append(chat)
        if start_ts is not None:
            sql += " AND ts>=?" # Add lower bound filter on timestamp
            params.append(int(start_ts))
        if end_ts is not None:
            sql += " AND ts<=?" # Add upper bound filter on timestamp
            params.append(int(end_ts))
        sql += " ORDER BY ts ASC"  # Always order results chronologically
        cur.execute(sql, params)
        rows = cur.fetchall()
        con.close()

        """
        This block takes the raw DB rows and turns them into:

        X: a NumPy matrix of embeddings.

        meta: aligned metadata (chat, sender, time, text).

        These are then used in query() to score and rank results.
        """

        meta: List[Dict] = []
        vecs: List[np.ndarray] = []
        dim: Optional[int] = None
        for idh, chat, sender, ts, text, d, blob in rows:
            v = np.frombuffer(blob, dtype = np.float32) # Converts blob into float
            if dim is None:
                dim = int(d)
            vecs.append(v)
            meta.append({"chat": chat, "sender": sender, "ts": int(ts), "text": text})

        if not vecs:
            return np.zeros((0,0), dtype=np.float32), []
        X = np.vstack(vecs)
        return X, meta

    def query(
        self,
        query_text: str,
        top_k: int = 20,
        chat: Optional[str] = None,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> List[Tuple[float, Dict]]:
            """Return top_k list of (score, meta_row) by cosine/dot on normalized vectors."""
            X, meta = self._fetch_vectors(chat, start_ts, end_ts)
            if X.size == 0:
                return []
            qv = self.model().encode([query_text], convert_to_numpy = True, normalize_embeddings = True).astype(np.float32)[0]
            scores = X @ qv
            idx = np.argsort(-scores)[: max(1, int(top_k))]
            return [(float(scores[i]), meta[i]) for i in idx]
