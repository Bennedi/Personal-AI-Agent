"""Ingest agents/knowledge/*.txt into the Qdrant Cloud 'knowledge' collection.

Run from repo root:  python agents/ingest.py
"""
import os
import uuid as uuid_mod
from pathlib import Path

from dotenv import load_dotenv
from fastembed import TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "https://675cb084-207c-468e-9877-5a485523e479.us-west-1-0.aws.cloud.qdrant.io")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "knowledge"
EMBED_MODEL = "BAAI/bge-small-en"  # 384-dim, runs on CPU via fastembed

if not QDRANT_API_KEY:
    raise SystemExit("QDRANT_API_KEY missing in .env")

knowledge_dir = Path(__file__).parent / "knowledge"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)

documents = []
for file in sorted(knowledge_dir.rglob("*.txt")):
    category = file.stem
    text = file.read_text(encoding="utf-8")
    for chunk in splitter.split_text(text):
        documents.append({"category": category, "source": file.name, "text": chunk})
    print(f"[read] {file.name}: chunks so far = {len(documents)}")

if not documents:
    raise SystemExit("No documents found in agents/knowledge")

print(f"[embed] loading {EMBED_MODEL} ...")
embedder = TextEmbedding(model_name=EMBED_MODEL)
dim = len(next(iter(embedder.embed([documents[0]["text"]]))))
print(f"[embed] model loaded, dim = {dim}")
vectors = list(embedder.embed([d["text"] for d in documents]))

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)



# Old 'knowledge' collection has size=4 placeholder vectors; recreate it.
collections = [c.name for c in client.get_collections().collections]
if COLLECTION in collections:
    info = client.get_collection(COLLECTION)
    if info.config.params.vectors.size != dim:
        print(f"[qdrant] recreating '{COLLECTION}' (was size {info.config.params.vectors.size}, need {dim})")
        client.delete_collection(COLLECTION)
if COLLECTION not in collections or COLLECTION not in [c.name for c in client.get_collections().collections]:
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
client.create_payload_index(
    collection_name=COLLECTION,
    field_name="category",
    field_schema="keyword",
    wait=True
)
# Replace previously ingested chunks for these categories so re-runs don't duplicate.
for category in {d["category"] for d in documents}:
    client.delete(
        COLLECTION,
        points_selector=Filter(must=[FieldCondition(key="category", match=MatchValue(value=category))]),
        wait=True,
    )

points = [
    PointStruct(
        id=str(uuid_mod.uuid5(uuid_mod.NAMESPACE_URL, f"{doc['category']}:{i}")),
        vector=vec.tolist(),
        payload={"category": doc["category"], "source": doc["source"], "text": doc["text"]},
    )
    for i, (doc, vec) in enumerate(zip(documents, vectors))
]
client.upload_points(COLLECTION, points, wait=True)
print(f"[done] upserted {len(points)} points into '{COLLECTION}' on Qdrant Cloud")
