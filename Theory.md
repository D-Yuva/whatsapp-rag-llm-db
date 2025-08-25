# Why SQL ?

	•	Local, file-based DB — no server to run, fully offline & private.
	•	One row per message: chat, sender, ts, text, dim, vec(BLOB) → everything needed to print exact [time] sender: message.
	•	Filter-before-score: fast WHERE chat=… AND ts BETWEEN … to shrink the candidate set before vector math.
	•	Dedup & upserts: keyed by sha1(text) so re-runs don’t re-embed the same message.
	•	Crash-safe, atomic writes: WAL mode keeps the DB consistent during incremental indexing.
	•	Incremental indexing: only new messages are embedded and inserted (quick startup after first run).
	•	Scales well for laptops