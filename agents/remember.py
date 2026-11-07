import uuid as uuid_mod
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qdrant_client.models import PointStruct

from memory import (AGENT_MEMORY, KNOWLEDGE, MIN_SCORE, client, embed, ensure_collections, expand_query)

def remember(text: str, source: str = "agent", category: str = "learned") -> str:
    clean = " ".join(text.split())
    if len(clean) < 3:
        raise ValueError("Fact is too short to store")
    point_id = str(uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS, f"{category}:{clean}"))
    client.upsert(
        collection_name=AGENT_MEMORY,
        points=[PointStruct(
            id=point_id,
            vector=embed(clean),
            payload={
                "text": clean,
                "category": category,
                "source": source,
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        )],
        wait=True
    )
    return point_id

def recall(query: str, limit: int = 6) -> list[tuple[float, str, str]]:

    vector = embed(expand_query(query))
    best: dict[str, tuple[float, str, str]] = {}
    for collection in (KNOWLEDGE, AGENT_MEMORY):
        for p in client.query_points(collection, query=vector, limit=limit, with_payload=True).points:
            if p.score < MIN_SCORE:
                continue
            key = " ".join(p.payload["text"].split())
            entry = (float(p.score), p.payload["text"], p.payload.get("category", "learned"))
            if best.get(key, (-1.0,))[0] < entry[0]:
                best[key] = entry
    return sorted(best.values(), key=lambda e: e[0], reverse=True)[:limit]

def facts_block(query: str, limit: int = 6) -> str:
    return "\n----\n".join(text for _, text, _ in recall(query, limit))

if __name__ == "__main__":
    ensure_collections()
    pid = remember("Habel's dog is named Jack", source="manual-test")
    print("Stored point id:", pid)
    print("Recall ('What is my dog's name?')")
    for score, text, category in recall("What is my dog's name?", limit=4):
        print(f" {score:.3f} [{category}] {text[:70]}")
