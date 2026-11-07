import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

PATH = Path(__file__).resolve().parent.parent / "data" / "sessions.jsonl"

def new_session_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:6]

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def append_turn(session_id: str, role: str, content: str, meta: dict | None = None) -> dict:
    record = {"session_id":session_id, "role": role, "content": content, "ts": _now()}
    if meta:
        record["meta"] = meta
    PATH.parent.mkdir(parents=True, exist_ok=True)
    with PATH.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record

def _read_all() -> list[dict]:
    if not PATH.exists():
        return []
    records = []
    for line in PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and {"session_id", "role", "content"} <=record.keys():
            records.append(record)
    return records

def load_session(session_id: str, last_n: int = 8) -> list[dict]:
    turns = [r for r in _read_all() if r["session_id"] == session_id]
    return [{"role": r["role"], "content": r["content"]} for r in turns[-last_n:]]

def list_sessions(limit: int = 20) -> list[dict]:
    grouped: dict[str, dict] = {}
    for r in _read_all():
        g = grouped.setdefault(
            r["session_id"],
            {"session_id": r["session_id"], "turns": 0, "first_ts": r["ts"], "last_ts": r["ts"], "preview": ""},)
        g["turns"] += 1
        g["last_ts"] = r["ts"]
        if r["role"] == "user" and not g["preview"]:
            g["preview"] = r["content"][:60] 
    return sorted(grouped.values(), key=lambda g: g["last_ts"], reverse=True)[:limit]

if __name__ == "__main__":
    sid = "selftest-" + uuid4().hex[:6]
    append_turn(sid, "user", "Hello there")
    append_turn(sid, "assistant", "Hey, what's up", meta={"facts_retrieved": 0})
    assert load_session(sid) == [
        {"role": "user", "content": "Hello there"},
        {"role": "assistant", "content": "Hey, what's up"}
    ], load_session(sid)
    print(f"[store] wrote to {PATH}")
    print(f"[store] verified; sessions={len(list_sessions())}")
