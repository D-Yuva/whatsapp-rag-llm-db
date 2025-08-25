"""
WHATSAPP CHAT CLEANING

Converts messy WhatsApp .txt exports into clean, structured, chronologically ordered message data.

- Defines regex for parsing chat lines (date, time, sender, message).
- Handles timezones (defaults to IST, converts to UTC epoch timestamps).
- Filters out noise (e.g., "image omitted", "This message was deleted", emoji/symbol-only lines).
- Supports continuation lines for multi-line messages.
- parse_whatsapp_txt(path): parses a single WhatsApp export into a list of dicts:
{"chat": name, "sender": sender, "text": text, "ts": timestamp}
- load_exports(patterns): loads multiple .txt exports (via glob), parses them, and returns all messages sorted chronologically.
"""

import re 
from typing import List, Dict, Any, Optional 
from pathlib import Path
from dateutil import parser as dateparser
from datetime import timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))

# Supports dd/mm/yy or dd-mm-yy and AM/PM 
LINE_RE = re.compile(
    r"^\[(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}),\s+"
    r"(\d{1,2}:\d{2}(?::\d{2})?)\s*"
    r"([APap]\.?[Mm]\.?)?\]\s+"
    r"([^:]+):\s(.*)$"
)

# Filters out unwanted messages from the chat 
NOISE_RE = re.compile(
    r"^(?:\u200e)?(sticker omitted|image omitted|video omitted|audio omitted|This message was deleted\.?)$",
    re.IGNORECASE,
)

# Defining what's noise
def _is_noise(text: str) -> bool:
    if not text or len(text.strip()) < 2: #If text is one character its treated as noise
        return True
    s = text.strip()
    if NOISE_RE.match(s): # Anything that is in NOISE_RE
        return True
    
    alnum = sum(ch.isalnum() for ch in s)
    return alnum < max(5, int(len(s)*0.2)) #If the number of alphanumeric characters is too small compared to the length of the string, in this case less than 5 or 20% of length
    # Example: Okay, Hi -> 4 letters and 2 letters, these are treated as noise

def parse_whatsapp_txt(path: Path, chat_name: Optional[str] = None) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = []
    name = chat_name or path.stem
    last = None

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():  # Read the file line by line
        m = LINE_RE.match(raw)  # Using REGEX LINE_RE to match the WhatsApp chat format
        if m:
            # Convert date and time into a string
            d, t, ampm, sender, text = m.groups()
            stamp = f"{d} {t} {ampm or ''}".strip()
            dt_local = dateparser.parse(stamp, dayfirst=True)
            if dt_local.tzinfo is None:
                dt_local = dt_local.replace(tzinfo=IST)
            ts = int(dt_local.astimezone(timezone.utc).timestamp())
            # Record
            rec = {
                "chat": name,
                "sender": (sender or "").strip(),
                "text": (text or "").strip(),
                "ts": ts,
            }
            msgs.append(rec)
            last = rec
        else:
            # Continuation line for multi-line messages
            if last is not None and raw.strip():
                last["text"] += "\n" + raw.strip()

    # Remove noise
    msgs = [r for r in msgs if r.get("text") and not _is_noise(r["text"])]
    return msgs

# Back-compat wrapper (old name)
def parse_whatsapp_text(path: Path, chat_name: Optional[str] = None) -> List[Dict[str, Any]]:
    return parse_whatsapp_txt(path, chat_name)

from glob import glob

def load_exports(patterns: List[str]) -> List[Dict[str, Any]]:
    """Load multiple .txt exports (globs allowed) and return chronologically sorted messages."""
    out: List[Dict[str, Any]] = []
    for pat in patterns:
        for fp in glob(pat, recursive=True):
            p = Path(fp)
            if p.suffix.lower() == ".txt":
                out.extend(parse_whatsapp_txt(p))
    out.sort(key=lambda r: r["ts"])  # chronological
    return out
